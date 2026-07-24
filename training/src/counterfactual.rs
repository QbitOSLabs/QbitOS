#![allow(missing_docs)]
//! Counterfactual reasoning: after execution, asks "what if" about alternative
//! tools, plans, or orderings. For each alternative action, the analyzer
//! estimates the reward delta it would have produced. If the alternative was
//! previously executed in a similar context, the historical reward is used;
//! otherwise a small default penalty (-0.1) is assigned to reflect uncertainty cost.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thiserror::Error;

const DEFAULT_DELTA: f64 = -0.1;
const HISTORICAL_CONFIDENCE: f64 = 0.8;
const DEFAULT_CONFIDENCE: f64 = 0.3;

#[derive(Debug, Error)]
pub enum CounterfactualError { #[error("no alternatives provided")] NoAlternatives }

/// A counterfactual scenario: what would have happened if we had taken `alternative_action`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualScenario {
    pub original_action: String,
    pub alternative_action: String,
    pub estimated_reward_delta: f64,
    pub confidence: f64,
    pub reasoning: String,
    pub context_hash: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl CounterfactualScenario {
    #[must_use]
    pub fn new(
        original: impl Into<String>, alternative: impl Into<String>,
        delta: f64, confidence: f64, reasoning: impl Into<String>,
    ) -> Self {
        Self {
            original_action: original.into(), alternative_action: alternative.into(),
            estimated_reward_delta: delta, confidence, reasoning: reasoning.into(),
            context_hash: String::new(), created_at: chrono::Utc::now(),
        }
    }
}

/// Counterfactual analyzer: tracks historical action outcomes and produces
/// "what if" scenarios for previously taken actions.
pub struct CounterfactualAnalyzer {
    history: Arc<RwLock<Vec<CounterfactualScenario>>>,
    observed: Arc<RwLock<AHashMap<String, f64>>>,
}

impl Default for CounterfactualAnalyzer { fn default() -> Self { Self::new() } }

impl CounterfactualAnalyzer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: Arc::new(RwLock::new(Vec::new())),
            observed: Arc::new(RwLock::new(AHashMap::new())),
        }
    }
    fn context_hash(context: &serde_json::Value) -> String {
        let mut h = DefaultHasher::new();
        context.to_string().hash(&mut h);
        format!("{:016x}", h.finish())
    }

    /// Record an observed (context, action, reward) tuple for future lookup.
    pub fn record_observation(&self, context: &serde_json::Value, action: &str, reward: f64) {
        let key = format!("{}|{}", Self::context_hash(context), action);
        self.observed.write().insert(key, reward);
    }

    /// Analyze alternatives to `original_action` taken in `context` that yielded
    /// `original_reward`. Returns one scenario per alternative.
    pub fn analyze(
        &self, context: &serde_json::Value, original_action: &str,
        original_reward: f64, alternatives: &[String],
    ) -> Vec<CounterfactualScenario> {
        let ctx_hash = Self::context_hash(context);
        let observed = self.observed.read();
        let mut scenarios = Vec::with_capacity(alternatives.len());
        for alt in alternatives {
            if alt == original_action { continue; }
            let key = format!("{}|{}", ctx_hash, alt);
            let (delta, confidence, reasoning) = if let Some(&past_reward) = observed.get(&key) {
                (past_reward - original_reward, HISTORICAL_CONFIDENCE,
                    format!("Historical observation: '{alt}' yielded reward {past_reward:.4}"))
            } else {
                (DEFAULT_DELTA, DEFAULT_CONFIDENCE,
                    "No historical data; default pessimistic estimate".to_string())
            };
            let mut s = CounterfactualScenario::new(original_action, alt, delta, confidence, reasoning);
            s.context_hash = ctx_hash.clone();
            scenarios.push(s);
        }
        self.history.write().extend(scenarios.clone());
        scenarios
    }

    /// Return the alternative action with the highest positive estimated delta
    /// for the given original action, or `None` if no beneficial alternative exists.
    #[must_use]
    pub fn get_best_alternative(&self, action: &str) -> Option<String> {
        let history = self.history.read();
        let best = history.iter().filter(|s| s.original_action == action)
            .max_by(|a, b| a.estimated_reward_delta.partial_cmp(&b.estimated_reward_delta).unwrap_or(std::cmp::Ordering::Equal))?;
        if best.estimated_reward_delta > 0.0 { Some(best.alternative_action.clone()) } else { None }
    }
    #[must_use] pub fn history(&self) -> Vec<CounterfactualScenario> { self.history.read().clone() }
    #[must_use] pub fn len(&self) -> usize { self.history.read().len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.history.read().is_empty() }
    #[must_use] pub fn observation_count(&self) -> usize { self.observed.read().len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn analyze_uses_default_delta_without_history() {
        let a = CounterfactualAnalyzer::new();
        let scenarios = a.analyze(&json!({"s": 1}), "A", 0.5, &["B".to_string(), "C".to_string()]);
        assert_eq!(scenarios.len(), 2);
        for s in &scenarios {
            assert!((s.estimated_reward_delta - DEFAULT_DELTA).abs() < 1e-9);
            assert!((s.confidence - DEFAULT_CONFIDENCE).abs() < 1e-9);
        }
    }
    #[test]
    fn analyze_uses_historical_data_when_available() {
        let a = CounterfactualAnalyzer::new();
        let ctx = json!({"state": "x"});
        a.record_observation(&ctx, "B", 1.0);
        let scenarios = a.analyze(&ctx, "A", 0.5, &["B".to_string()]);
        assert_eq!(scenarios.len(), 1);
        assert!((scenarios[0].estimated_reward_delta - 0.5).abs() < 1e-9);
        assert!((scenarios[0].confidence - HISTORICAL_CONFIDENCE).abs() < 1e-9);
    }
    #[test]
    fn analyze_skips_original_action() {
        let a = CounterfactualAnalyzer::new();
        let scenarios = a.analyze(&json!({}), "A", 0.5, &["A".to_string(), "B".to_string()]);
        assert_eq!(scenarios.len(), 1);
        assert_eq!(scenarios[0].alternative_action, "B");
    }
    #[test]
    fn get_best_alternative_returns_positive_delta_action() {
        let a = CounterfactualAnalyzer::new();
        let ctx = json!({});
        a.record_observation(&ctx, "B", 1.0);
        a.analyze(&ctx, "A", 0.2, &["B".to_string()]);
        assert_eq!(a.get_best_alternative("A"), Some("B".to_string()));
    }
    #[test]
    fn get_best_alternative_none_when_no_positive() {
        let a = CounterfactualAnalyzer::new();
        a.analyze(&json!({}), "A", 1.0, &["B".to_string()]);
        assert!(a.get_best_alternative("A").is_none());
    }
    #[test]
    fn history_returns_all_scenarios() {
        let a = CounterfactualAnalyzer::new();
        a.analyze(&json!({}), "A", 0.5, &["B".to_string(), "C".to_string(), "D".to_string()]);
        assert_eq!(a.history().len(), 3);
    }
}
