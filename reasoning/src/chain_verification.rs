#![allow(missing_docs)]

//! Chain-of-thought verification: detects contradictions between reasoning
//! steps and scores chain coherence on a [0.0, 1.0] scale.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

/// Errors emitted by the chain verifier.
#[derive(Debug, Error)]
pub enum ChainVerifyError {
    /// Empty chain.
    #[error("empty chain: at least one step required")]
    Empty,
}

/// A single reasoning step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// The premise this step builds on.
    pub premise: String,
    /// The conclusion derived.
    pub conclusion: String,
    /// Supporting evidence (cited facts, prior step ids, etc.).
    #[serde(default)]
    pub evidence: Vec<String>,
}

impl ReasoningStep {
    /// Create a new step.
    #[must_use]
    pub fn new(premise: impl Into<String>, conclusion: impl Into<String>) -> Self {
        Self { premise: premise.into(), conclusion: conclusion.into(), evidence: Vec::new() }
    }
    /// Add an evidence reference (builder).
    pub fn with_evidence(mut self, e: impl Into<String>) -> Self {
        self.evidence.push(e.into()); self
    }
}

/// A detected contradiction between two steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    /// Index of the first conflicting step.
    pub step_a: usize,
    /// Index of the second conflicting step.
    pub step_b: usize,
    /// Severity in [0, 1].
    pub severity: f32,
    /// Human-readable description.
    pub description: String,
}

/// Result of verifying a chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainVerificationResult {
    /// Whether the chain is overall coherent.
    pub coherent: bool,
    /// Coherence score in [0, 1].
    pub coherence_score: f32,
    /// Detected contradictions.
    pub contradictions: Vec<Contradiction>,
}

/// The chain verifier.
#[derive(Debug, Clone)]
pub struct ChainVerifier {
    topic_threshold: f32,
    negation_threshold: f32,
}

impl Default for ChainVerifier { fn default() -> Self { Self::new() } }

impl ChainVerifier {
    /// Create a new verifier with default thresholds.
    #[must_use]
    pub fn new() -> Self { Self { topic_threshold: 0.15, negation_threshold: 0.25 } }

    /// Verify an entire chain.
    pub fn verify_chain(&self, steps: &[ReasoningStep]) -> Result<ChainVerificationResult, ChainVerifyError> {
        if steps.is_empty() { return Err(ChainVerifyError::Empty); }
        let contradictions = self.detect_contradictions(steps);
        let topic = self.topic_continuity(steps);
        let evidence = self.evidence_coverage(steps);
        let penalty = contradictions.iter().map(|c| c.severity).sum::<f32>() / steps.len().max(1) as f32;
        let coherence = (0.5 * topic + 0.3 * evidence + 0.2) * (1.0 - penalty).max(0.0);
        let coherence = coherence.clamp(0.0, 1.0);
        Ok(ChainVerificationResult { coherent: contradictions.is_empty() && coherence > 0.4, coherence_score: coherence, contradictions })
    }

    /// Detect contradictions between any pair of steps.
    pub fn detect_contradictions(&self, steps: &[ReasoningStep]) -> Vec<Contradiction> {
        let mut out = Vec::new();
        for i in 0..steps.len() {
            for j in (i + 1)..steps.len() {
                if let Some(c) = self.check_pair(i, j, &steps[i], &steps[j]) { out.push(c); }
            }
        }
        out
    }

    fn check_pair(&self, i: usize, j: usize, a: &ReasoningStep, b: &ReasoningStep) -> Option<Contradiction> {
        let toks_a = tokenize(&a.conclusion);
        let toks_b = tokenize(&b.conclusion);
        let overlap = jaccard(&toks_a, &toks_b);
        if overlap >= self.negation_threshold && has_negation(&a.conclusion) != has_negation(&b.conclusion) {
            let sev = (overlap - self.negation_threshold) / (1.0 - self.negation_threshold).max(0.01);
            return Some(Contradiction {
                step_a: i, step_b: j, severity: sev.clamp(0.0, 1.0),
                description: format!("step {i} and step {j} reach opposite conclusions (overlap={overlap:.2})"),
            });
        }
        let cap = jaccard(&toks_a, &tokenize(&b.premise));
        if cap >= self.negation_threshold && has_negation(&a.conclusion) != has_negation(&b.premise) {
            return Some(Contradiction {
                step_a: i, step_b: j, severity: cap.clamp(0.0, 1.0),
                description: format!("step {j}'s premise contradicts step {i}'s conclusion"),
            });
        }
        None
    }

    fn topic_continuity(&self, steps: &[ReasoningStep]) -> f32 {
        if steps.len() < 2 { return 1.0; }
        let sum: f32 = steps.windows(2)
            .map(|w| {
                let overlap = jaccard(&tokenize(&w[0].conclusion), &tokenize(&w[1].premise));
                // Discount pairs that fall below the topic continuity threshold —
                // they indicate a context switch rather than a coherent chain.
                if overlap < self.topic_threshold { 0.0 } else { overlap }
            })
            .sum();
        (sum / (steps.len() - 1) as f32).max(0.0)
    }

    fn evidence_coverage(&self, steps: &[ReasoningStep]) -> f32 {
        steps.iter().filter(|s| !s.evidence.is_empty()).count() as f32 / steps.len() as f32
    }
}

fn tokenize(s: &str) -> HashSet<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2).map(String::from).collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() { return 0.0; }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn has_negation(s: &str) -> bool {
    const NEG: &[&str] = &[
        " not ", " no ", " none ", "n't", "cannot", "never", "nothing",
        "neither", "nor", "without", "isn't", "wasn't", "aren't", "weren't",
        "don't", "doesn't", "didn't", "won't", "can't", "couldn't",
    ];
    let lower = format!(" {} ", s.to_lowercase());
    NEG.iter().any(|n| lower.contains(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherent_chain_scores_high() {
        let chain = vec![
            ReasoningStep::new("all humans are mortal", "socrates is mortal").with_evidence("p1"),
            ReasoningStep::new("socrates is mortal", "socrates will die").with_evidence("p2"),
        ];
        let r = ChainVerifier::new().verify_chain(&chain).unwrap();
        assert!(r.coherent && r.coherence_score > 0.3 && r.contradictions.is_empty());
    }

    #[test]
    fn detects_negation_contradiction() {
        let chain = vec![
            ReasoningStep::new("the sky is blue", "the sky is blue today"),
            ReasoningStep::new("the sky is blue", "the sky is not blue today"),
        ];
        let r = ChainVerifier::new().verify_chain(&chain).unwrap();
        assert!(!r.contradictions.is_empty() && !r.coherent);
    }

    #[test]
    fn empty_chain_errors() { assert!(ChainVerifier::new().verify_chain(&[]).is_err()); }

    #[test]
    fn evidence_coverage_affects_score() {
        let no_ev = vec![ReasoningStep::new("a", "b"), ReasoningStep::new("b", "c")];
        let with_ev = vec![
            ReasoningStep::new("a", "b").with_evidence("e1"),
            ReasoningStep::new("b", "c").with_evidence("e2"),
        ];
        let v = ChainVerifier::new();
        let s1 = v.verify_chain(&no_ev).unwrap().coherence_score;
        let s2 = v.verify_chain(&with_ev).unwrap().coherence_score;
        assert!(s2 >= s1);
    }
}
