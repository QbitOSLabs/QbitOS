#![allow(missing_docs)]
//! Causal learning: learns "A → B because C" causal chains from observation.
//! Each observation of (cause, intermediate, effect) either creates a new
//! chain or strengthens an existing one (incrementing evidence count and
//! confidence). An adjacency map supports forward and reverse lookups.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

const CONFIDENCE_INCREMENT: f64 = 0.05;
const CONFIDENCE_MAX: f64 = 1.0;

#[derive(Debug, Error)]
pub enum CausalLearningError { #[error("empty causal component")] EmptyComponent }

/// A causal chain: cause → intermediate → effect, with confidence and evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalChain {
    pub cause: String,
    pub intermediate: String,
    pub effect: String,
    pub confidence: f64,
    pub evidence_count: u32,
    pub last_observed: chrono::DateTime<chrono::Utc>,
}

impl CausalChain {
    #[must_use]
    pub fn new(cause: impl Into<String>, intermediate: impl Into<String>, effect: impl Into<String>) -> Self {
        Self {
            cause: cause.into(), intermediate: intermediate.into(), effect: effect.into(),
            confidence: 0.1, evidence_count: 1, last_observed: chrono::Utc::now(),
        }
    }
}

/// The causal learner.
pub struct CausalLearner {
    chains: Arc<RwLock<Vec<CausalChain>>>,
    adjacency: Arc<RwLock<AHashMap<String, Vec<String>>>>,
    index: Arc<RwLock<AHashMap<String, usize>>>,
}

impl Default for CausalLearner { fn default() -> Self { Self::new() } }

impl CausalLearner {
    #[must_use]
    pub fn new() -> Self {
        Self {
            chains: Arc::new(RwLock::new(Vec::new())),
            adjacency: Arc::new(RwLock::new(AHashMap::new())),
            index: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    /// Observe a (cause, intermediate, effect) triple. Strengthens an existing
    /// chain or records a new one.
    pub fn observe(&self, cause: &str, intermediate: &str, effect: &str) {
        let key = format!("{cause}|{intermediate}|{effect}");
        let mut chains = self.chains.write();
        let mut index = self.index.write();
        if let Some(&i) = index.get(&key) {
            let chain = &mut chains[i];
            chain.evidence_count += 1;
            chain.confidence = (chain.confidence + CONFIDENCE_INCREMENT).min(CONFIDENCE_MAX);
            chain.last_observed = chrono::Utc::now();
            return;
        }
        let chain = CausalChain::new(cause, intermediate, effect);
        let i = chains.len();
        index.insert(key, i);
        let mut adj = self.adjacency.write();
        adj.entry(cause.to_string()).or_default().push(intermediate.to_string());
        adj.entry(intermediate.to_string()).or_default().push(effect.to_string());
        chains.push(chain);
    }

    /// Find all chains that produce this effect.
    #[must_use]
    pub fn query_cause(&self, effect: &str) -> Vec<CausalChain> {
        self.chains.read().iter().filter(|c| c.effect == effect).cloned().collect()
    }
    /// Find all chains that this cause leads to.
    #[must_use]
    pub fn query_effect(&self, cause: &str) -> Vec<CausalChain> {
        self.chains.read().iter().filter(|c| c.cause == cause).cloned().collect()
    }
    /// Get the highest-confidence chain originating from a cause.
    #[must_use]
    pub fn get_strongest_chain(&self, cause: &str) -> Option<CausalChain> {
        self.chains.read().iter().filter(|c| c.cause == cause)
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }
    /// Get the list of nodes adjacent to `node` in the causal graph.
    #[must_use]
    pub fn get_adjacency(&self, node: &str) -> Vec<String> {
        self.adjacency.read().get(node).cloned().unwrap_or_default()
    }
    #[must_use] pub fn len(&self) -> usize { self.chains.read().len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.chains.read().is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observe_creates_new_chain() {
        let c = CausalLearner::new();
        c.observe("rain", "wet_ground", "slippery");
        assert_eq!(c.len(), 1);
        let chains = c.query_effect("rain");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].effect, "slippery");
    }
    #[test]
    fn observe_strengthens_existing_chain() {
        let c = CausalLearner::new();
        c.observe("rain", "wet", "slip");
        let initial = c.get_strongest_chain("rain").unwrap().confidence;
        c.observe("rain", "wet", "slip");
        let after = c.get_strongest_chain("rain").unwrap();
        assert!(after.confidence > initial);
        assert_eq!(after.evidence_count, 2);
        assert_eq!(c.len(), 1);
    }
    #[test]
    fn query_cause_finds_all_producers() {
        let c = CausalLearner::new();
        c.observe("rain", "wet", "slip");
        c.observe("sprinkler", "wet", "slip");
        assert_eq!(c.query_cause("slip").len(), 2);
    }
    #[test]
    fn get_strongest_chain_returns_highest_confidence() {
        let c = CausalLearner::new();
        c.observe("a", "b", "c");
        c.observe("a", "b", "c");
        c.observe("x", "y", "z");
        let strongest = c.get_strongest_chain("a").unwrap();
        assert_eq!(strongest.intermediate, "b");
        assert!(strongest.confidence > 0.1);
    }
    #[test]
    fn adjacency_lists_neighbors() {
        let c = CausalLearner::new();
        c.observe("rain", "wet", "slip");
        assert_eq!(c.get_adjacency("rain"), vec!["wet".to_string()]);
        assert_eq!(c.get_adjacency("wet"), vec!["slip".to_string()]);
        assert!(c.get_adjacency("unknown").is_empty());
    }
}
