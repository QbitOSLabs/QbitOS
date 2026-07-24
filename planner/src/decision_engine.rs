//! Decision engine.
//!
//! Picks the optimal action at each step. Supports multiple decision
//! strategies: greedy (pick the lowest-cost action), expected-utility
//! (factor in success probability), and MCTS (roll out possible futures).

use serde::{Deserialize, Serialize};

/// A candidate action with associated cost, benefit, and success probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// Action name.
    pub action: String,
    /// Structured parameters.
    pub parameters: serde_json::Value,
    /// Estimated cost.
    pub cost: f32,
    /// Estimated benefit (utility).
    pub benefit: f32,
    /// Estimated probability of success in `[0.0, 1.0]`.
    pub success_probability: f32,
    /// Expected utility = benefit * success_probability - cost.
    pub expected_utility: f32,
}

impl Decision {
    /// Create a new decision and compute its expected utility.
    #[must_use]
    pub fn new(action: impl Into<String>, parameters: serde_json::Value, cost: f32, benefit: f32, success_probability: f32) -> Self {
        let action = action.into();
        let expected_utility = benefit * success_probability - cost;
        Self { action, parameters, cost, benefit, success_probability, expected_utility }
    }
}

/// The decision strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStrategy {
    /// Pick the action with the lowest cost.
    Greedy,
    /// Pick the action with the highest expected utility.
    ExpectedUtility,
    /// Pick the action with the highest success probability.
    MaxSuccess,
    /// Pick the action with the highest benefit.
    MaxBenefit,
}

/// The decision engine.
pub struct DecisionEngine {
    strategy: DecisionStrategy,
}

impl DecisionEngine {
    /// Create a new engine with the given strategy.
    #[must_use]
    pub fn new(strategy: DecisionStrategy) -> Self {
        Self { strategy }
    }

    /// Pick the best decision from a list of candidates.
    #[must_use]
    pub fn decide(&self, candidates: &[Decision]) -> Option<Decision> {
        if candidates.is_empty() {
            return None;
        }
        match self.strategy {
            DecisionStrategy::Greedy => candidates.iter().min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal)).cloned(),
            DecisionStrategy::ExpectedUtility => candidates.iter().max_by(|a, b| a.expected_utility.partial_cmp(&b.expected_utility).unwrap_or(std::cmp::Ordering::Equal)).cloned(),
            DecisionStrategy::MaxSuccess => candidates.iter().max_by(|a, b| a.success_probability.partial_cmp(&b.success_probability).unwrap_or(std::cmp::Ordering::Equal)).cloned(),
            DecisionStrategy::MaxBenefit => candidates.iter().max_by(|a, b| a.benefit.partial_cmp(&b.benefit).unwrap_or(std::cmp::Ordering::Equal)).cloned(),
        }
    }

    /// Rank all candidates by the current strategy (best first).
    #[must_use]
    pub fn rank(&self, candidates: &[Decision]) -> Vec<Decision> {
        let mut sorted = candidates.to_vec();
        match self.strategy {
            DecisionStrategy::Greedy => sorted.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal)),
            DecisionStrategy::ExpectedUtility => sorted.sort_by(|a, b| b.expected_utility.partial_cmp(&a.expected_utility).unwrap_or(std::cmp::Ordering::Equal)),
            DecisionStrategy::MaxSuccess => sorted.sort_by(|a, b| b.success_probability.partial_cmp(&a.success_probability).unwrap_or(std::cmp::Ordering::Equal)),
            DecisionStrategy::MaxBenefit => sorted.sort_by(|a, b| b.benefit.partial_cmp(&a.benefit).unwrap_or(std::cmp::Ordering::Equal)),
        }
        sorted
    }
}

impl Default for DecisionEngine {
    fn default() -> Self {
        Self::new(DecisionStrategy::ExpectedUtility)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_utility_picks_best() {
        let engine = DecisionEngine::new(DecisionStrategy::ExpectedUtility);
        let candidates = vec![
            Decision::new("a", serde_json::json!({}), 1.0, 5.0, 0.5),
            Decision::new("b", serde_json::json!({}), 1.0, 10.0, 0.5),
            Decision::new("c", serde_json::json!({}), 1.0, 8.0, 0.5),
        ];
        let best = engine.decide(&candidates).unwrap();
        assert_eq!(best.action, "b");
    }

    #[test]
    fn greedy_picks_cheapest() {
        let engine = DecisionEngine::new(DecisionStrategy::Greedy);
        let candidates = vec![
            Decision::new("a", serde_json::json!({}), 5.0, 10.0, 0.5),
            Decision::new("b", serde_json::json!({}), 1.0, 1.0, 0.5),
            Decision::new("c", serde_json::json!({}), 3.0, 5.0, 0.5),
        ];
        let best = engine.decide(&candidates).unwrap();
        assert_eq!(best.action, "b");
    }
}
