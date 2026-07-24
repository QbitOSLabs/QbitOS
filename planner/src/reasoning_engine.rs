//! Reasoning engine.
//!
//! Applies a set of inference rules to a fact base, producing new facts and
//! a trace of how each was derived. Supports forward-chaining only — backward
//! chaining is the responsibility of the decision engine.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// A logical rule: `IF antecedents THEN consequent`.
#[derive(Clone)]
pub struct InferenceRule {
    /// Human-readable name.
    pub name: String,
    /// Antecedents: each must evaluate to `true` for the rule to fire.
    pub antecedents: Vec<Predicate>,
    /// Consequents: facts to add when the rule fires.
    pub consequents: Vec<Fact>,
}

impl std::fmt::Debug for InferenceRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceRule")
            .field("name", &self.name)
            .field("antecedents_count", &self.antecedents.len())
            .field("consequents_count", &self.consequents.len())
            .finish()
    }
}

/// A predicate over a fact base.
pub type Predicate = Arc<dyn Fn(&FactBase) -> bool + Send + Sync>;

/// A fact: a typed key-value pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    /// Fact key (e.g. "weather.is_raining").
    pub key: String,
    /// Fact value (any JSON).
    pub value: serde_json::Value,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// A fact base: a map from key to fact.
#[derive(Debug, Default, Clone)]
pub struct FactBase {
    facts: AHashMap<String, Fact>,
}

impl FactBase {
    /// Create a new empty fact base.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a fact.
    pub fn insert(&mut self, fact: Fact) {
        self.facts.insert(fact.key.clone(), fact);
    }

    /// Look up a fact by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Fact> {
        self.facts.get(key)
    }

    /// Returns `true` if a fact with the given key exists.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.facts.contains_key(key)
    }

    /// Iterate over all facts.
    pub fn iter(&self) -> impl Iterator<Item = &Fact> {
        self.facts.values()
    }

    /// Number of facts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Returns `true` if the fact base is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

/// A reasoning trace: the derivation history of one rule firing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// Unique id.
    pub id: Uuid,
    /// Rule that fired.
    pub rule_name: String,
    /// Facts that were used as antecedents.
    pub premises: Vec<String>,
    /// Facts that were derived.
    pub conclusions: Vec<String>,
    /// When this rule fired.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// The reasoning engine: forward-chains over a set of rules until no more
/// rules fire.
pub struct ReasoningEngine {
    rules: Vec<InferenceRule>,
}

impl ReasoningEngine {
    /// Create a new engine with the given rules.
    #[must_use]
    pub fn new(rules: Vec<InferenceRule>) -> Self {
        Self { rules }
    }

    /// Add a rule to the engine.
    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.rules.push(rule);
    }

    /// Run forward chaining until no more rules fire. Returns the trace.
    pub fn forward_chain(&self, base: &mut FactBase) -> Vec<ReasoningTrace> {
        let mut traces = Vec::new();
        let mut changed = true;
        let max_iterations = 100;
        let mut iteration = 0;

        while changed && iteration < max_iterations {
            changed = false;
            iteration += 1;
            for rule in &self.rules {
                // Skip if any consequent is already in the base (rule already fired)
                let already_fired = rule.consequents.iter().all(|c| base.contains(&c.key));
                if already_fired {
                    continue;
                }
                if rule.antecedents.iter().all(|p| p(base)) {
                    let premises: Vec<String> = rule.antecedents.iter()
                        .flat_map(|_| base.iter().map(|f| f.key.clone()))
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    let conclusions: Vec<String> = rule.consequents.iter().map(|c| c.key.clone()).collect();
                    for fact in &rule.consequents {
                        base.insert(fact.clone());
                    }
                    traces.push(ReasoningTrace {
                        id: Uuid::now_v7(),
                        rule_name: rule.name.clone(),
                        premises,
                        conclusions,
                        timestamp: chrono::Utc::now(),
                    });
                    changed = true;
                }
            }
        }
        traces
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_chaining() {
        let mut base = FactBase::new();
        base.insert(Fact { key: "is_raining".into(), value: serde_json::json!(true), confidence: 0.9 });
        base.insert(Fact { key: "is_cold".into(), value: serde_json::json!(true), confidence: 0.8 });

        let rules = vec![InferenceRule {
            name: "raining_and_cold_implies_snow".into(),
            antecedents: vec![
                Arc::new(|b: &FactBase| b.get("is_raining").map_or(false, |f| f.value.as_bool() == Some(true))),
                Arc::new(|b: &FactBase| b.get("is_cold").map_or(false, |f| f.value.as_bool() == Some(true))),
            ],
            consequents: vec![Fact { key: "is_snowing".into(), value: serde_json::json!(true), confidence: 0.7 }],
        }];

        let engine = ReasoningEngine::new(rules);
        let traces = engine.forward_chain(&mut base);
        assert_eq!(traces.len(), 1);
        assert!(base.contains("is_snowing"));
    }
}
