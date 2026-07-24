//! Markov Decision Process (MDP) solver.
//!
//! Implements value iteration and policy iteration for finding
//! optimal policies in sequential decision-making problems.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

/// An MDP state identifier.
pub type MdpState = String;

/// An MDP action identifier.
pub type MdpAction = String;

/// A transition: (state, action) → (next_state, probability, reward).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Next state.
    pub next_state: MdpState,
    /// Probability of this transition.
    pub probability: f64,
    /// Reward received.
    pub reward: f64,
}

/// A policy: state → action.
pub type Policy = AHashMap<MdpState, MdpAction>;

/// Result of MDP solving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdpResult {
    /// Optimal policy.
    pub policy: Policy,
    /// Value function: state → expected cumulative reward.
    pub values: AHashMap<MdpState, f64>,
    /// Number of iterations.
    pub iterations: usize,
}

/// MDP solver using value iteration.
pub struct MdpSolver {
    /// Discount factor (gamma).
    pub discount: f64,
    /// Convergence threshold.
    pub threshold: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl Default for MdpSolver {
    fn default() -> Self {
        Self { discount: 0.9, threshold: 1e-6, max_iterations: 1000 }
    }
}

impl MdpSolver {
    /// Create a new solver.
    #[must_use]
    pub fn new(discount: f64) -> Self {
        Self { discount, ..Default::default() }
    }

    /// Solve an MDP using value iteration.
    ///
    /// - `states`: list of all states
    /// - `actions`: list of all actions
    /// - `transitions`: (state, action) → list of possible transitions
    /// - `terminal_states`: states with no outgoing transitions
    #[must_use]
    pub fn value_iteration(
        &self,
        states: &[MdpState],
        actions: &[MdpAction],
        transitions: &AHashMap<(MdpState, MdpAction), Vec<Transition>>,
        terminal_states: &std::collections::HashSet<MdpState>,
    ) -> MdpResult {
        let mut values: AHashMap<MdpState, f64> = states.iter().map(|s| (s.clone(), 0.0)).collect();
        let mut iterations = 0;

        loop {
            iterations += 1;
            if iterations > self.max_iterations { break; }

            let mut max_delta = 0.0_f64;
            let mut new_values = values.clone();

            for state in states {
                if terminal_states.contains(state) { continue; }

                let best_value = actions.iter()
                    .filter_map(|action| {
                        let trans = transitions.get(&(state.clone(), action.clone()))?;
                        let q: f64 = trans.iter()
                            .map(|t| t.probability * (t.reward + self.discount * values.get(&t.next_state).copied().unwrap_or(0.0)))
                            .sum();
                        Some(q)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);

                if best_value.is_finite() {
                    new_values.insert(state.clone(), best_value);
                    let delta = (best_value - values.get(state).copied().unwrap_or(0.0)).abs();
                    max_delta = max_delta.max(delta);
                }
            }

            values = new_values;
            if max_delta < self.threshold { break; }
        }

        // Extract policy
        let policy = self.extract_policy(states, actions, transitions, &values);

        MdpResult { policy, values, iterations }
    }

    /// Extract the optimal policy from a value function.
    fn extract_policy(
        &self,
        states: &[MdpState],
        actions: &[MdpAction],
        transitions: &AHashMap<(MdpState, MdpAction), Vec<Transition>>,
        values: &AHashMap<MdpState, f64>,
    ) -> Policy {
        let mut policy = Policy::new();

        for state in states {
            let best_action = actions.iter()
                .filter_map(|action| {
                    let trans = transitions.get(&(state.clone(), action.clone()))?;
                    let q: f64 = trans.iter()
                        .map(|t| t.probability * (t.reward + self.discount * values.get(&t.next_state).copied().unwrap_or(0.0)))
                        .sum();
                    Some((action.clone(), q))
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(a, _)| a);

            if let Some(action) = best_action {
                policy.insert(state.clone(), action);
            }
        }

        policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn solve_simple_mdp() {
        // Simple MDP: 3 states (A, B, C)
        // A: go_right → B (reward 0), go_left → C (reward 0)
        // B: terminal (reward 10)
        // C: terminal (reward -10)
        let states = vec!["A".into(), "B".into(), "C".into()];
        let actions = vec!["go_right".into(), "go_left".into()];
        let mut transitions = AHashMap::new();
        transitions.insert(("A".into(), "go_right".into()), vec![Transition { next_state: "B".into(), probability: 1.0, reward: 0.0 }]);
        transitions.insert(("A".into(), "go_left".into()), vec![Transition { next_state: "C".into(), probability: 1.0, reward: 0.0 }]);
        transitions.insert(("B".into(), "go_right".into()), vec![Transition { next_state: "B".into(), probability: 1.0, reward: 10.0 }]);
        transitions.insert(("C".into(), "go_left".into()), vec![Transition { next_state: "C".into(), probability: 1.0, reward: -10.0 }]);

        let terminal: HashSet<MdpState> = HashSet::new(); // no terminal — let it converge
        let solver = MdpSolver::new(0.9);
        let result = solver.value_iteration(&states, &actions, &transitions, &terminal);

        // Optimal policy for A should be "go_right" (leads to B with reward 10)
        assert_eq!(result.policy.get("A"), Some(&"go_right".to_string()));
        // Value of A should be positive (discounted sum of rewards from B)
        let val_a = result.values.get("A").copied().unwrap_or(0.0);
        assert!(val_a > 0.0, "value of A should be positive, got {val_a}");
    }
}
