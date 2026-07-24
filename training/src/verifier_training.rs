#![allow(missing_docs)]
//! Self-improving verifier: learns from verification failures. Each sample is
//! tagged with a pattern type (`false_positive`, `false_negative`, or
//! `correct`). Patterns are indexed by feature signature and used to bias
//! future verifications.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifierTrainingError { #[error("pattern lookup failed: {0}")] LookupFailed(String) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierTrainingSample {
    pub question: String,
    pub expected_answer: String,
    pub actual_answer: String,
    pub verification_result: bool,
    pub reasoning: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl VerifierTrainingSample {
    #[must_use]
    pub fn new(
        question: impl Into<String>, expected: impl Into<String>, actual: impl Into<String>,
        verified: bool, reasoning: impl Into<String>,
    ) -> Self {
        Self {
            question: question.into(), expected_answer: expected.into(),
            actual_answer: actual.into(), verification_result: verified,
            reasoning: reasoning.into(), timestamp: chrono::Utc::now(),
        }
    }
    #[must_use]
    pub fn pattern_type(&self) -> &'static str {
        let ok = self.actual_answer.trim().eq_ignore_ascii_case(self.expected_answer.trim());
        match (self.verification_result, ok) {
            (true, false) => "false_positive",
            (false, true) => "false_negative",
            _ => "correct",
        }
    }
    #[must_use]
    pub fn features(&self) -> Vec<String> {
        let mut f = Vec::with_capacity(5);
        let q_len = self.question.len();
        f.push(if q_len < 50 { "short_question" } else if q_len < 200 { "medium_question" } else { "long_question" }.into());
        let a_len = self.actual_answer.len();
        f.push(if a_len < 20 { "short_answer" } else if a_len < 100 { "medium_answer" } else { "long_answer" }.into());
        let lower = self.actual_answer.to_lowercase();
        if lower.contains("because") || lower.contains("therefore") { f.push("has_reasoning_markers".into()); }
        if self.question.chars().filter(|c| c.is_ascii_digit()).count() > 3 { f.push("numeric".into()); }
        f.push(self.pattern_type().to_string());
        f
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationPattern { pub pattern_type: String, pub features: Vec<String>, pub frequency: u64, pub confidence: f64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedResult { pub verified: bool, pub confidence: f64, pub matched_pattern: Option<String> }

pub struct VerifierTrainer {
    samples: Arc<RwLock<Vec<VerifierTrainingSample>>>,
    patterns: Arc<RwLock<AHashMap<String, VerificationPattern>>>,
    confidence_threshold: f64,
}

impl Default for VerifierTrainer { fn default() -> Self { Self::new() } }

impl VerifierTrainer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Arc::new(RwLock::new(Vec::new())),
            patterns: Arc::new(RwLock::new(AHashMap::new())),
            confidence_threshold: 0.5,
        }
    }
    #[must_use] pub fn with_confidence_threshold(mut self, t: f64) -> Self { self.confidence_threshold = t.clamp(0.0, 1.0); self }
    fn pattern_key(features: &[String]) -> String { features.join("|") }

    pub fn record(&self, sample: VerifierTrainingSample) {
        let feats = sample.features();
        let ptype = sample.pattern_type().to_string();
        self.samples.write().push(sample);
        let mut patterns = self.patterns.write();
        let key = Self::pattern_key(&feats);
        let entry = patterns.entry(key).or_insert(VerificationPattern {
            pattern_type: ptype.clone(), features: feats, frequency: 0, confidence: 0.5,
        });
        entry.frequency += 1;
        let n = entry.frequency as f64;
        let target = if ptype == "correct" { 1.0 } else { 0.0 };
        entry.confidence = (entry.confidence * (n - 1.0) + target) / n;
    }
    #[must_use]
    pub fn get_pattern(&self, features: &[String]) -> Option<VerificationPattern> {
        self.patterns.read().get(&Self::pattern_key(features)).cloned()
    }
    #[must_use]
    pub fn verify_with_confidence(&self, question: &str, answer: &str, expected: &str) -> VerifiedResult {
        let is_exact = answer.trim().eq_ignore_ascii_case(expected.trim());
        let sample = VerifierTrainingSample::new(question, expected, answer, is_exact, "");
        let feats = sample.features();
        if let Some(p) = self.get_pattern(&feats) {
            let verified = match p.pattern_type.as_str() {
                "false_positive" => is_exact && p.confidence >= self.confidence_threshold,
                "false_negative" => is_exact || p.confidence < self.confidence_threshold,
                _ => is_exact,
            };
            return VerifiedResult { verified, confidence: p.confidence, matched_pattern: Some(p.pattern_type) };
        }
        VerifiedResult { verified: is_exact, confidence: 0.5, matched_pattern: None }
    }
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        let s = self.samples.read();
        if s.is_empty() { 0.0 } else { s.iter().filter(|x| x.pattern_type() == "correct").count() as f64 / s.len() as f64 }
    }
    #[must_use] pub fn len(&self) -> usize { self.samples.read().len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.samples.read().is_empty() }
    #[must_use] pub fn pattern_count(&self) -> usize { self.patterns.read().len() }
}

impl std::fmt::Debug for VerifierTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifierTrainer").field("samples", &self.len())
            .field("patterns", &self.pattern_count()).field("accuracy", &self.accuracy()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pattern_type_classifies_correctly() {
        assert_eq!(VerifierTrainingSample::new("q", "yes", "yes", true, "").pattern_type(), "correct");
        assert_eq!(VerifierTrainingSample::new("q", "yes", "no", true, "").pattern_type(), "false_positive");
        assert_eq!(VerifierTrainingSample::new("q", "yes", "yes", false, "").pattern_type(), "false_negative");
    }
    #[test]
    fn record_creates_and_updates_pattern() {
        let t = VerifierTrainer::new();
        t.record(VerifierTrainingSample::new("What is 2+2?", "4", "4", true, "match"));
        t.record(VerifierTrainingSample::new("What is 3+3?", "6", "6", true, "match"));
        assert_eq!(t.len(), 2);
        assert!(t.pattern_count() >= 1);
        assert!((t.accuracy() - 1.0).abs() < 1e-9);
    }
    #[test]
    fn verify_with_confidence_returns_pattern_when_known() {
        let t = VerifierTrainer::new();
        t.record(VerifierTrainingSample::new("What is 2+2?", "4", "4", true, ""));
        let r = t.verify_with_confidence("What is 2+2?", "4", "4");
        assert!(r.verified);
        assert!(r.matched_pattern.is_some());
    }
    #[test]
    fn verify_with_confidence_returns_default_when_unknown() {
        let t = VerifierTrainer::new();
        let r = t.verify_with_confidence("unknown question", "yes", "yes");
        assert!(r.verified);
        assert!((r.confidence - 0.5).abs() < 1e-9);
        assert!(r.matched_pattern.is_none());
    }
    #[test]
    fn false_positive_pattern_makes_verifier_stricter() {
        let t = VerifierTrainer::new();
        t.record(VerifierTrainingSample::new(
            "What is the capital of France? Tell me the answer", "Paris", "London", true, "guessed"));
        let r = t.verify_with_confidence(
            "What is the capital of France? Tell me the answer", "London", "Paris");
        assert!(!r.verified);
    }
}
