#![allow(missing_docs)]
//! Self-Improving Harness Engineering — Lilian Weng, July 2026.
//!
//! Implements the "Harness Engineering for Self-Improvement" framework.
//! The agent's operating harness — the collection of prompts, tool
//! configurations, and memory strategies that wrap the base model — is
//! treated as a first-class, tunable artifact. The agent iteratively
//! proposes changes to its own harness, evaluates them against a
//! performance metric, and accepts the change only if it improves
//! performance without regressing other components.
//!
//! Reference:
//! - Lilian Weng, "Harness Engineering for Self-Improvement" (July 2026).
//! - "Self-Harness: Agents that Tune Their Own Operating Context" (2026).

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Errors that can occur during harness engineering.
#[derive(Debug, Error)]
pub enum HarnessError {
    /// The requested component was not registered.
    #[error("component '{0}' not found")]
    NotFound(String),
    /// A regression was detected that blocks the improvement.
    #[error("regression detected on component '{component}': {delta}")]
    Regression {
        /// Name of the regressing component.
        component: String,
        /// Magnitude of the regression.
        delta: f64,
    },
}

/// Configuration for the harness engineer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    /// Maximum number of self-improvement iterations.
    pub max_iterations: usize,
    /// Minimum performance improvement required to accept a proposal.
    pub improvement_threshold: f64,
    /// Whether the regression guard is enabled.
    pub regression_guard_enabled: bool,
    /// Maximum allowed regression on any other component when a proposal
    /// is applied.
    pub max_regression: f64,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            improvement_threshold: 0.02,
            regression_guard_enabled: true,
            max_regression: 0.05,
        }
    }
}

/// A single tunable harness component (prompt, tool config, memory
/// strategy, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessComponent {
    /// Human-readable name (e.g. `"system_prompt"`, `"tool:search.top_k"`).
    pub name: String,
    /// Current value of the component.
    pub current_value: Value,
    /// Most recent performance score for this component (0..1).
    pub performance_score: f64,
    /// History of past performance scores (oldest first).
    pub history: Vec<f64>,
}

impl HarnessComponent {
    /// Construct a new component with an initial value and a baseline
    /// performance score of `0.5`.
    #[must_use]
    pub fn new(name: impl Into<String>, initial_value: Value) -> Self {
        Self {
            name: name.into(),
            current_value: initial_value,
            performance_score: 0.5,
            history: vec![0.5],
        }
    }

    /// Record a new performance score and update the running estimate.
    pub fn record_score(&mut self, score: f64) {
        let alpha = 0.3;
        self.performance_score = (1.0 - alpha) * self.performance_score + alpha * score;
        self.history.push(self.performance_score);
        // Cap history length to keep memory bounded.
        if self.history.len() > 100 {
            let drain_n = self.history.len() - 100;
            self.history.drain(0..drain_n);
        }
    }

    /// Trend (positive = improving, negative = degrading) over the last
    /// few observations.
    #[must_use]
    pub fn trend(&self) -> f64 {
        let n = self.history.len();
        if n < 2 {
            return 0.0;
        }
        let window: Vec<f64> = self.history.iter().rev().take(5).copied().collect();
        let first = *window.last().unwrap();
        let last = *window.first().unwrap();
        last - first
    }
}

/// Overall evaluation of the harness across all components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessEvaluation {
    /// Mean performance across all components.
    pub overall_score: f64,
    /// Per-component performance scores.
    pub component_scores: AHashMap<String, f64>,
}

/// A proposed change to a single harness component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessProposal {
    /// Name of the target component.
    pub component_name: String,
    /// Proposed new value.
    pub proposed_value: Value,
    /// Expected improvement (in performance score).
    pub expected_improvement: f64,
    /// Human-readable rationale for the change.
    pub rationale: String,
}

/// The harness engineer: iteratively proposes, evaluates, and applies
/// improvements to its own operating harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessEngineer {
    /// Configured components.
    pub components: Vec<HarnessComponent>,
    /// Engineer configuration.
    pub config: HarnessConfig,
    /// Number of iterations performed so far.
    pub iterations: usize,
    /// Number of accepted proposals.
    pub accepted: usize,
    /// Number of rejected proposals.
    pub rejected: usize,
}

impl HarnessEngineer {
    /// Construct a new harness engineer.
    #[must_use]
    pub fn new(config: HarnessConfig) -> Self {
        Self {
            components: Vec::new(),
            config,
            iterations: 0,
            accepted: 0,
            rejected: 0,
        }
    }

    /// Register a new harness component.
    pub fn register_component(&mut self, name: impl Into<String>, initial_value: Value) {
        self.components.push(HarnessComponent::new(name, initial_value));
    }

    /// Evaluate the current harness.
    #[must_use]
    pub fn evaluate(&self) -> HarnessEvaluation {
        let mut component_scores = AHashMap::new();
        let mut sum = 0.0;
        for c in &self.components {
            component_scores.insert(c.name.clone(), c.performance_score);
            sum += c.performance_score;
        }
        let overall_score = if self.components.is_empty() {
            0.0
        } else {
            sum / self.components.len() as f64
        };
        HarnessEvaluation {
            overall_score,
            component_scores,
        }
    }

    /// Propose an improvement for a specific component.
    ///
    /// Returns `None` if the component does not exist or its trend is
    /// already positive (i.e. it is improving on its own).
    pub fn propose_improvement(&self, component_name: &str) -> Option<HarnessProposal> {
        let comp = self
            .components
            .iter()
            .find(|c| c.name == component_name)?;
        // If the component is already trending up beyond the threshold,
        // don't propose changes.
        if comp.trend() > self.config.improvement_threshold {
            return None;
        }
        let (proposed_value, rationale, expected) = self.synthesize_proposal(comp);
        Some(HarnessProposal {
            component_name: component_name.to_string(),
            proposed_value,
            expected_improvement: expected,
            rationale,
        })
    }

    /// Synthesize a proposed improvement for a component.
    ///
    /// The strategy depends on the value's JSON type:
    /// - **String**: append a precision/verification directive.
    /// - **Number**: nudge towards a better-known default.
    /// - **Array**: append a missing element if the array is short.
    /// - **Object**: add a `"strict": true` flag.
    fn synthesize_proposal(&self, comp: &HarnessComponent) -> (Value, String, f64) {
        let v = &comp.current_value;
        match v {
            Value::String(s) => {
                let new_s = format!("{s} Be precise and verify your answer.");
                let expected = 0.1;
                let rationale = format!(
                    "Appending a precision/verification directive to '{}'.",
                    comp.name
                );
                (Value::String(new_s), rationale, expected)
            }
            Value::Number(n) => {
                let cur = n.as_f64().unwrap_or(0.5);
                let target = if cur < 0.5 { cur + 0.1 } else { cur - 0.05 };
                let expected = 0.05;
                let rationale = format!(
                    "Tuning numeric parameter '{}' from {:.3} to {:.3}.",
                    comp.name, cur, target
                );
                (Value::from(target), rationale, expected)
            }
            Value::Array(a) => {
                let mut new_a = a.clone();
                if new_a.len() < 5 {
                    new_a.push(Value::String("additional_context".into()));
                }
                let expected = 0.08;
                let rationale = format!(
                    "Adding an extra context entry to array '{}'.",
                    comp.name
                );
                (Value::Array(new_a), rationale, expected)
            }
            Value::Object(o) => {
                let mut new_o = o.clone();
                new_o.insert("strict".into(), Value::Bool(true));
                let expected = 0.07;
                let rationale = format!(
                    "Enabling 'strict' mode on object '{}'.",
                    comp.name
                );
                (Value::Object(new_o), rationale, expected)
            }
            _ => (v.clone(), "No change proposed.".into(), 0.0),
        }
    }

    /// Apply a proposed improvement.
    ///
    /// Returns `true` if the proposal was accepted, `false` otherwise.
    /// The regression guard rejects proposals that would cause any other
    /// component's score to drop by more than `max_regression`.
    pub fn apply_improvement(&mut self, proposal: &HarnessProposal) -> bool {
        if self.iterations >= self.config.max_iterations {
            return false;
        }
        let Some(idx) = self
            .components
            .iter()
            .position(|c| c.name == proposal.component_name)
        else {
            self.rejected += 1;
            return false;
        };

        // Compute a synthetic post-application score: current score + expected
        // improvement, clamped to [0, 1].
        let cur = self.components[idx].performance_score;
        let new_score = (cur + proposal.expected_improvement).clamp(0.0, 1.0);

        // Regression guard: ensure other components do not regress more than
        // the configured threshold. In this surrogate we model the
        // side-effect as a small uniform dip proportional to the magnitude
        // of the change.
        if self.config.regression_guard_enabled {
            let side_effect = proposal.expected_improvement * 0.4;
            if side_effect > self.config.max_regression {
                self.rejected += 1;
                self.iterations += 1;
                return false;
            }
            // Apply the dip to all other components.
            for (i, c) in self.components.iter_mut().enumerate() {
                if i != idx {
                    let dipped = (c.performance_score - side_effect).max(0.0);
                    c.record_score(dipped);
                }
            }
        }

        // Apply the change.
        self.components[idx].current_value = proposal.proposed_value.clone();
        self.components[idx].record_score(new_score);
        self.accepted += 1;
        self.iterations += 1;
        true
    }

    /// Run a full self-improvement pass: evaluate, propose for the weakest
    /// component, apply if the proposal is acceptable.
    ///
    /// Returns the accepted proposal, or `None` if no proposal was
    /// accepted this pass.
    pub fn run_pass(&mut self) -> Option<HarnessProposal> {
        let _eval = self.evaluate();
        // Pick the weakest component.
        let weakest = self
            .components
            .iter()
            .min_by(|a, b| {
                a.performance_score
                    .partial_cmp(&b.performance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.name.clone())?;
        let proposal = self.propose_improvement(&weakest)?;
        if self.apply_improvement(&proposal) {
            Some(proposal)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_component_adds_to_list() {
        let mut he = HarnessEngineer::new(HarnessConfig::default());
        he.register_component("system_prompt", serde_json::json!("You are an assistant."));
        assert_eq!(he.components.len(), 1);
    }

    #[test]
    fn evaluate_returns_mean_score() {
        let mut he = HarnessEngineer::new(HarnessConfig::default());
        he.register_component("a", serde_json::json!(1));
        he.register_component("b", serde_json::json!(2));
        he.components[0].performance_score = 0.4;
        he.components[1].performance_score = 0.6;
        let e = he.evaluate();
        assert!((e.overall_score - 0.5).abs() < 1e-9);
        assert_eq!(e.component_scores.len(), 2);
    }

    #[test]
    fn propose_improvement_for_string_component() {
        let mut he = HarnessEngineer::new(HarnessConfig::default());
        he.register_component("system_prompt", serde_json::json!("You are an assistant."));
        let p = he.propose_improvement("system_prompt").unwrap();
        assert_eq!(p.component_name, "system_prompt");
        assert!(p.expected_improvement > 0.0);
        assert!(p.proposed_value.is_string());
    }

    #[test]
    fn apply_improvement_accepts_small_change() {
        let mut he = HarnessEngineer::new(HarnessConfig::default());
        he.register_component("system_prompt", serde_json::json!("Hi"));
        let p = he.propose_improvement("system_prompt").unwrap();
        let accepted = he.apply_improvement(&p);
        assert!(accepted);
        assert_eq!(he.accepted, 1);
    }

    #[test]
    fn apply_improvement_rejects_large_change() {
        let mut he = HarnessEngineer::new(HarnessConfig {
            max_regression: 0.001,
            ..HarnessConfig::default()
        });
        he.register_component("system_prompt", serde_json::json!("Hi"));
        let p = HarnessProposal {
            component_name: "system_prompt".into(),
            proposed_value: serde_json::json!("Hi Be precise."),
            expected_improvement: 0.5, // large
            rationale: "test".into(),
        };
        let accepted = he.apply_improvement(&p);
        assert!(!accepted);
        assert_eq!(he.rejected, 1);
    }

    #[test]
    fn apply_improvement_rejects_unknown_component() {
        let mut he = HarnessEngineer::new(HarnessConfig::default());
        let p = HarnessProposal {
            component_name: "does_not_exist".into(),
            proposed_value: serde_json::json!(null),
            expected_improvement: 0.1,
            rationale: "test".into(),
        };
        assert!(!he.apply_improvement(&p));
    }

    #[test]
    fn run_pass_picks_weakest_component() {
        let mut he = HarnessEngineer::new(HarnessConfig::default());
        he.register_component("a", serde_json::json!(1));
        he.register_component("b", serde_json::json!("hello"));
        he.components[0].performance_score = 0.9;
        he.components[1].performance_score = 0.2;
        let p = he.run_pass();
        assert!(p.is_some());
        assert_eq!(p.unwrap().component_name, "b");
    }

    #[test]
    fn component_trend_detects_improvement() {
        let mut c = HarnessComponent::new("x", serde_json::json!(0));
        c.record_score(0.6);
        c.record_score(0.7);
        c.record_score(0.8);
        assert!(c.trend() > 0.0);
    }

    #[test]
    fn component_trend_zero_with_few_observations() {
        let c = HarnessComponent::new("x", serde_json::json!(0));
        assert_eq!(c.trend(), 0.0);
    }
}
