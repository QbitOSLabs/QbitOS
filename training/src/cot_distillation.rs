#![allow(missing_docs)]
//! Chain-of-thought distillation: compresses verbose reasoning chains into
//! compact procedural knowledge that can be retrieved and applied to similar
//! future situations. The distiller extracts: situation from the first
//! reasoning step, decision and outcome from explicit arguments, lesson by
//! summarising key patterns (step counts, analysis/verification markers),
//! and compression_ratio = 1 - lesson.len() / total_chain_len.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CotDistillationError { #[error("empty reasoning chain")] EmptyChain }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledKnowledge {
    pub id: String,
    pub situation: String,
    pub decision: String,
    pub outcome: String,
    pub lesson: String,
    pub confidence: f64,
    pub compression_ratio: f64,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Compresses reasoning chains into distilled knowledge.
pub struct CotDistiller {
    knowledge_base: Arc<RwLock<Vec<DistilledKnowledge>>>,
}

impl Default for CotDistiller { fn default() -> Self { Self::new() } }

impl CotDistiller {
    #[must_use]
    pub fn new() -> Self { Self { knowledge_base: Arc::new(RwLock::new(Vec::new())) } }

    /// Distill a reasoning chain into compact procedural knowledge.
    pub fn distill(&self, reasoning_chain: &[String], decision: &str, outcome: &str) -> DistilledKnowledge {
        let situation = reasoning_chain.first().cloned().unwrap_or_default();
        let step_count = reasoning_chain.len();
        let total_chain_len: usize = reasoning_chain.iter().map(|s| s.len()).sum();
        let analysis_count = reasoning_chain.iter().filter(|s| {
            let l = s.to_lowercase();
            l.contains("analyze") || l.contains("examine") || l.contains("consider")
        }).count();
        let verification_count = reasoning_chain.iter().filter(|s| {
            let l = s.to_lowercase();
            l.contains("verify") || l.contains("check") || l.contains("confirm")
        }).count();
        let lesson = format!(
            "Steps: {step_count}, Analysis steps: {analysis_count}, Verification steps: \
             {verification_count}. Decision: {decision}. Outcome: {outcome}."
        );
        let compression_ratio = if total_chain_len > 0 {
            1.0 - (lesson.len() as f64 / total_chain_len as f64)
        } else { 0.0 };
        let outcome_lower = outcome.to_lowercase();
        let confidence = if outcome_lower.contains("success") || outcome_lower.contains("correct") {
            0.9
        } else if outcome_lower.contains("fail") || outcome_lower.contains("error") {
            0.4
        } else { 0.6 };
        let knowledge = DistilledKnowledge {
            id: uuid::Uuid::new_v4().to_string(),
            situation, decision: decision.to_string(), outcome: outcome.to_string(),
            lesson, confidence, compression_ratio: compression_ratio.clamp(0.0, 1.0),
            created_at: chrono::Utc::now(),
        };
        self.knowledge_base.write().push(knowledge.clone());
        knowledge
    }

    /// Retrieve the top-k distilled knowledge whose situation overlaps with
    /// the given situation (Jaccard similarity on tokens).
    #[must_use]
    pub fn retrieve(&self, situation: &str, k: usize) -> Vec<DistilledKnowledge> {
        let kb = self.knowledge_base.read();
        let q: HashSet<String> = situation.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|s| !s.is_empty()).collect();
        let mut scored: Vec<(f64, &DistilledKnowledge)> = kb.iter().map(|k| {
            let s: HashSet<String> = k.situation.split_whitespace()
                .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .filter(|s| !s.is_empty()).collect();
            if q.is_empty() || s.is_empty() { return (0.0, k); }
            let inter = q.intersection(&s).count() as f64;
            let union = q.union(&s).count() as f64;
            (inter / union, k)
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().filter(|(s, _)| *s > 0.0).take(k).map(|(_, k)| k.clone()).collect()
    }
    #[must_use] pub fn len(&self) -> usize { self.knowledge_base.read().len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.knowledge_base.read().is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distill_creates_knowledge_with_compression() {
        let d = CotDistiller::new();
        let chain = vec![
            "First I analyzed the problem carefully".to_string(),
            "Then I considered all options".to_string(),
            "Finally I verified my solution".to_string(),
        ];
        let k = d.distill(&chain, "Use option B", "success");
        assert!(!k.id.is_empty());
        assert_eq!(k.situation, "First I analyzed the problem carefully");
        assert!(k.compression_ratio > 0.0);
        assert!((k.confidence - 0.9).abs() < 1e-9);
    }
    #[test]
    fn distill_empty_chain_uses_empty_situation() {
        let d = CotDistiller::new();
        let k = d.distill(&[], "decide", "unknown");
        assert_eq!(k.situation, "");
        assert!((k.compression_ratio - 0.0).abs() < 1e-9);
        assert!((k.confidence - 0.6).abs() < 1e-9);
    }
    #[test]
    fn distill_failure_outcome_yields_low_confidence() {
        let d = CotDistiller::new();
        let k = d.distill(&["tried something".into()], "wrong path", "failure");
        assert!((k.confidence - 0.4).abs() < 1e-9);
    }
    #[test]
    fn retrieve_finds_similar_situations() {
        let d = CotDistiller::new();
        d.distill(&["debug rust null pointer issue".into()], "use safe refs", "success");
        d.distill(&["cook pasta for dinner".into()], "boil water", "success");
        let results = d.retrieve("debug rust safety", 5);
        assert!(!results.is_empty());
        assert!(results[0].situation.contains("rust"));
    }
    #[test]
    fn len_increments_with_distill() {
        let d = CotDistiller::new();
        assert_eq!(d.len(), 0);
        d.distill(&["step1".into()], "d", "o");
        assert_eq!(d.len(), 1);
    }
}
