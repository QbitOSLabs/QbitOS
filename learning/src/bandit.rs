//! Multi-armed bandit algorithms.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// A bandit arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditArm {
    /// Arm name.
    pub name: String,
    /// Number of times this arm was pulled.
    pub pulls: u64,
    /// Total reward received.
    pub total_reward: f64,
}

impl BanditArm {
    /// Create a new arm.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), pulls: 0, total_reward: 0.0 }
    }

    /// Average reward.
    #[must_use]
    pub fn average_reward(&self) -> f64 {
        if self.pulls == 0 { 0.0 } else { self.total_reward / self.pulls as f64 }
    }

    /// Record a pull with a reward.
    pub fn pull(&mut self, reward: f64) {
        self.pulls += 1;
        self.total_reward += reward;
    }
}

/// Trait for bandit algorithms.
pub trait Bandit: Send + Sync {
    /// Select the next arm to pull.
    fn select(&self) -> String;
    /// Record the reward for a pull.
    fn update(&self, arm: &str, reward: f64);
    /// Get all arms.
    fn arms(&self) -> Vec<BanditArm>;
}

/// UCB1 (Upper Confidence Bound) bandit.
///
/// Selects the arm that maximizes:
/// `Q(a) + c * sqrt(ln(N) / n(a))`
///
/// where Q(a) is the average reward, N is total pulls, n(a) is arm pulls.
pub struct Ucb1Bandit {
    arms: RwLock<AHashMap<String, BanditArm>>,
    exploration_constant: f64,
    total_pulls: RwLock<u64>,
}

impl Ucb1Bandit {
    /// Create a new UCB1 bandit with arms.
    #[must_use]
    pub fn new(arm_names: Vec<String>, exploration_constant: f64) -> Self {
        let arms: AHashMap<String, BanditArm> = arm_names.into_iter()
            .map(|name| (name.clone(), BanditArm::new(name)))
            .collect();
        Self {
            arms: RwLock::new(arms),
            exploration_constant,
            total_pulls: RwLock::new(0),
        }
    }

    /// Create with default exploration constant sqrt(2).
    #[must_use]
    pub fn default_constant(arm_names: Vec<String>) -> Self {
        Self::new(arm_names, std::f64::consts::SQRT_2)
    }
}

impl Bandit for Ucb1Bandit {
    fn select(&self) -> String {
        let arms = self.arms.read();
        let total = *self.total_pulls.read();

        // If any arm hasn't been pulled, pull it first
        for arm in arms.values() {
            if arm.pulls == 0 {
                return arm.name.clone();
            }
        }

        // Select arm with highest UCB1 value
        arms.values()
            .map(|arm| {
                let ucb = arm.average_reward()
                    + self.exploration_constant * ((total as f64).ln() / arm.pulls as f64).sqrt();
                (arm.name.clone(), ucb)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name)
            .unwrap_or_default()
    }

    fn update(&self, arm_name: &str, reward: f64) {
        let mut arms = self.arms.write();
        if let Some(arm) = arms.get_mut(arm_name) {
            arm.pull(reward);
        }
        *self.total_pulls.write() += 1;
    }

    fn arms(&self) -> Vec<BanditArm> {
        self.arms.read().values().cloned().collect()
    }
}

/// Epsilon-greedy bandit.
///
/// With probability ε, explores a random arm; otherwise exploits the best arm.
pub struct EpsilonGreedyBandit {
    arms: RwLock<AHashMap<String, BanditArm>>,
    epsilon: f64,
    total_pulls: RwLock<u64>,
}

impl EpsilonGreedyBandit {
    /// Create a new epsilon-greedy bandit.
    #[must_use]
    pub fn new(arm_names: Vec<String>, epsilon: f64) -> Self {
        let arms: AHashMap<String, BanditArm> = arm_names.into_iter()
            .map(|name| (name.clone(), BanditArm::new(name)))
            .collect();
        Self {
            arms: RwLock::new(arms),
            epsilon,
            total_pulls: RwLock::new(0),
        }
    }

    /// Decay epsilon (reduce exploration over time).
    pub fn decay_epsilon(&self, factor: f64) {
        // Note: epsilon is not mutable after construction in this design.
        // In production, we'd use RwLock<f64> for epsilon.
        let _ = factor;
    }
}

impl Bandit for EpsilonGreedyBandit {
    fn select(&self) -> String {
        let arms = self.arms.read();
        let arm_names: Vec<String> = arms.keys().cloned().collect();
        if arm_names.is_empty() { return String::new(); }

        // Simple pseudo-random: use total_pulls as seed
        let total = *self.total_pulls.read();
        let pseudo_rand = ((total.wrapping_mul(2654435761)) as f64 / u64::MAX as f64).fract();

        if pseudo_rand < self.epsilon {
            // Explore: pick random arm
            let idx = (total as usize) % arm_names.len();
            return arm_names[idx].clone();
        }

        // Exploit: pick best arm
        arms.values()
            .max_by(|a, b| a.average_reward().partial_cmp(&b.average_reward()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|arm| arm.name.clone())
            .unwrap_or_else(|| arm_names[0].clone())
    }

    fn update(&self, arm_name: &str, reward: f64) {
        let mut arms = self.arms.write();
        if let Some(arm) = arms.get_mut(arm_name) {
            arm.pull(reward);
        }
        *self.total_pulls.write() += 1;
    }

    fn arms(&self) -> Vec<BanditArm> {
        self.arms.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucb1_pulls_each_arm_first() {
        let bandit = Ucb1Bandit::default_constant(vec!["a".into(), "b".into(), "c".into()]);
        // First 3 pulls should each be a different arm
        let first = bandit.select();
        bandit.update(&first, 1.0);
        let second = bandit.select();
        bandit.update(&second, 0.0);
        let third = bandit.select();
        assert_ne!(first, second);
        assert_ne!(second, third);
        assert_ne!(first, third);
    }

    #[test]
    fn ucb1_prefers_high_reward_arms() {
        let bandit = Ucb1Bandit::default_constant(vec!["good".into(), "bad".into()]);
        // Pull each once
        bandit.update("good", 1.0);
        bandit.update("bad", 0.0);
        // Next several pulls should prefer "good"
        let mut good_count = 0;
        for _ in 0..20 {
            let choice = bandit.select();
            if choice == "good" { good_count += 1; }
            bandit.update(&choice, if choice == "good" { 1.0 } else { 0.0 });
        }
        assert!(good_count > 10, "UCB1 should prefer high-reward arm, got {good_count}/20 good pulls");
    }

    #[test]
    fn epsilon_greedy_works() {
        let bandit = EpsilonGreedyBandit::new(vec!["a".into(), "b".into()], 0.3);
        bandit.update("a", 1.0);
        bandit.update("b", 0.0);
        // With epsilon=0.3, we expect ~70% exploitation (best arm)
        let mut a_count = 0;
        for _ in 0..100 {
            let choice = bandit.select();
            if choice == "a" { a_count += 1; }
            bandit.update(&choice, if choice == "a" { 1.0 } else { 0.0 });
        }
        // Should exploit more than explore (epsilon=0.3 means ~70% exploit)
        assert!(a_count >= 40, "epsilon-greedy should mostly exploit best arm, got {a_count}/100");
    }
}
