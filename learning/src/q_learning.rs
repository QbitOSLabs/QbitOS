//! Q-learning for sequential decision-making.
//!
//! Implements tabular Q-learning with epsilon-greedy exploration
//! and temporal difference (TD) updates.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// A Q-learning state.
pub type QState = String;
/// A Q-learning action.
pub type QAction = String;

/// Q-table entry: (state, action) → Q-value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QTable {
    /// Q-values keyed by "state|action".
    values: AHashMap<String, f64>,
    /// Learning rate (alpha).
    pub learning_rate: f64,
    /// Discount factor (gamma).
    pub discount: f64,
    /// Exploration rate (epsilon).
    pub epsilon: f64,
}

impl Default for QTable {
    fn default() -> Self {
        Self { values: AHashMap::new(), learning_rate: 0.1, discount: 0.95, epsilon: 0.1 }
    }
}

impl QTable {
    /// Create a new Q-table.
    #[must_use]
    pub fn new(learning_rate: f64, discount: f64, epsilon: f64) -> Self {
        Self { values: AHashMap::new(), learning_rate, discount, epsilon }
    }

    fn key(state: &str, action: &str) -> String {
        format!("{state}|{action}")
    }

    /// Get Q(s, a).
    #[must_use]
    pub fn get(&self, state: &str, action: &str) -> f64 {
        self.values.get(&Self::key(state, action)).copied().unwrap_or(0.0)
    }

    /// Set Q(s, a).
    pub fn set(&mut self, state: &str, action: &str, value: f64) {
        self.values.insert(Self::key(state, action), value);
    }

    /// Get all actions tried in a state.
    #[must_use]
    pub fn actions_for(&self, state: &str) -> Vec<String> {
        let prefix = format!("{state}|");
        self.values.keys()
            .filter_map(|k| {
                if k.starts_with(&prefix) {
                    k[prefix.len()..].to_string().into()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the best action for a state (greedy).
    #[must_use]
    pub fn best_action(&self, state: &str, available_actions: &[String]) -> Option<String> {
        if available_actions.is_empty() { return None; }
        available_actions.iter()
            .max_by(|a, b| {
                let qa = self.get(state, a);
                let qb = self.get(state, b);
                qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Get the max Q-value for a state.
    #[must_use]
    pub fn max_q(&self, state: &str, available_actions: &[String]) -> f64 {
        available_actions.iter()
            .map(|a| self.get(state, a))
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0) // Default to 0 if no actions
    }

    /// Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    pub fn update(&mut self, state: &str, action: &str, reward: f64, next_state: &str, next_actions: &[String]) {
        let current_q = self.get(state, action);
        let max_next_q = self.max_q(next_state, next_actions);
        let td_target = reward + self.discount * max_next_q;
        let td_error = td_target - current_q;
        let new_q = current_q + self.learning_rate * td_error;
        self.set(state, action, new_q);
    }

    /// Number of Q-values learned.
    #[must_use]
    pub fn len(&self) -> usize { self.values.len() }

    /// Returns `true` if the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.values.is_empty() }
}

/// Q-learning agent.
pub struct QLearner {
    table: RwLock<QTable>,
}

impl Default for QLearner {
    fn default() -> Self { Self { table: RwLock::new(QTable::default()) } }
}

impl QLearner {
    /// Create a new Q-learning agent.
    #[must_use]
    pub fn new(learning_rate: f64, discount: f64, epsilon: f64) -> Self {
        Self { table: RwLock::new(QTable::new(learning_rate, discount, epsilon)) }
    }

    /// Select an action using epsilon-greedy.
    ///
    /// With probability ε, explore (random); otherwise exploit (best Q).
    #[must_use]
    pub fn select_action(&self, state: &str, available_actions: &[String]) -> Option<String> {
        if available_actions.is_empty() { return None; }

        // Simple pseudo-random based on state hash
        let hash = state.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let rand_val = (hash as f64 / u64::MAX as f64).fract();

        let table = self.table.read();
        if rand_val < table.epsilon {
            // Explore: random action
            let idx = (hash as usize) % available_actions.len();
            Some(available_actions[idx].clone())
        } else {
            // Exploit: best action
            table.best_action(state, available_actions)
        }
    }

    /// Learn from a transition: (state, action, reward, next_state, next_actions).
    pub fn learn(&self, state: &str, action: &str, reward: f64, next_state: &str, next_actions: &[String]) {
        self.table.write().update(state, action, reward, next_state, next_actions);
    }

    /// Get the Q-table.
    #[must_use]
    pub fn table(&self) -> QTable {
        self.table.read().clone()
    }

    /// Decay epsilon (reduce exploration over time).
    pub fn decay_epsilon(&self, factor: f64) {
        let mut table = self.table.write();
        table.epsilon *= factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_table_update() {
        let mut q = QTable::new(0.1, 0.9, 0.1);
        q.update("s1", "a1", 10.0, "s2", &["a1".into(), "a2".into()]);
        // Q(s1,a1) should have increased from 0
        let val = q.get("s1", "a1");
        assert!(val > 0.0, "Q-value should be positive after positive reward, got {val}");
    }

    #[test]
    fn q_learner_learns_optimal() {
        let learner = QLearner::new(0.5, 0.9, 0.0); // No exploration — pure exploitation

        // Simple environment: state A has actions "go_right" (reward 10) and "go_left" (reward -10)
        // Both lead to terminal state T.
        for _ in 0..100 {
            learner.learn("A", "go_right", 10.0, "T", &[]);
            learner.learn("A", "go_left", -10.0, "T", &[]);
        }

        let action = learner.select_action("A", &["go_right".into(), "go_left".into()]);
        assert_eq!(action, Some("go_right".into()));

        let q = learner.table();
        assert!(q.get("A", "go_right") > q.get("A", "go_left"));
    }

    #[test]
    fn q_table_best_action() {
        let mut q = QTable::default();
        q.set("s1", "a1", 5.0);
        q.set("s1", "a2", 10.0);
        q.set("s1", "a3", 3.0);
        let best = q.best_action("s1", &["a1".into(), "a2".into(), "a3".into()]);
        assert_eq!(best, Some("a2".into()));
    }
}
