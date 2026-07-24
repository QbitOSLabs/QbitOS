//! Meta-learning: learn which learning strategies work best.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A learning strategy name (e.g. "curriculum", "self_play", "experience_replay").
pub type StrategyName = String;

/// Evaluation of a learning strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStrategyEval {
    /// Strategy name.
    pub strategy: StrategyName,
    /// Total times this strategy was used.
    pub uses: u64,
    /// Total reward accumulated.
    pub total_reward: f64,
    /// Mean reward per use.
    pub mean_reward: f64,
    /// Last reward observed.
    pub last_reward: f64,
    /// Confidence interval (Wilson 95%).
    pub confidence: f64,
    /// Last time this strategy was used (Unix timestamp).
    pub last_used: chrono::DateTime<chrono::Utc>,
}

impl LearningStrategyEval {
    /// Construct a new evaluation entry.
    #[must_use]
    pub fn new(strategy: impl Into<StrategyName>) -> Self {
        Self {
            strategy: strategy.into(),
            uses: 0,
            total_reward: 0.0,
            mean_reward: 0.0,
            last_reward: 0.0,
            confidence: 0.0,
            last_used: chrono::Utc::now(),
        }
    }

    /// Record a use of this strategy.
    pub fn record(&mut self, reward: f64) {
        self.uses += 1;
        self.total_reward += reward;
        self.last_reward = reward;
        self.mean_reward = self.total_reward / self.uses as f64;
        self.last_used = chrono::Utc::now();
        // Wilson 95% confidence interval width approximation.
        let n = self.uses as f64;
        let z = 1.96;
        let p = ((self.mean_reward + 1.0) / 2.0).clamp(0.001, 0.999);
        let denom = 1.0 + z * z / n;
        let _centre = p + z * z / (2.0 * n);
        let margin = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt()) / denom;
        self.confidence = 1.0 - 2.0 * margin;
    }
}

/// The meta-learner: tracks which learning strategies work best and selects
/// the best one for the next training round.
pub struct MetaLearner {
    evaluations: Arc<RwLock<AHashMap<StrategyName, LearningStrategyEval>>>,
    /// Exploration rate (epsilon in epsilon-greedy).
    pub epsilon: f64,
    /// Whether to use UCB1 for selection (vs epsilon-greedy).
    pub use_ucb1: bool,
    /// UCB1 exploration constant.
    pub ucb1_c: f64,
}

impl Default for MetaLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaLearner {
    /// Construct a new meta-learner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            evaluations: Arc::new(RwLock::new(AHashMap::new())),
            epsilon: 0.1,
            use_ucb1: true,
            ucb1_c: 2.0_f64.sqrt(),
        }
    }

    /// Register a strategy (initialize its evaluation entry).
    pub fn register(&self, strategy: impl Into<StrategyName>) {
        let name = strategy.into();
        self.evaluations
            .write()
            .entry(name.clone())
            .or_insert_with(|| LearningStrategyEval::new(name));
    }

    /// Record the reward of using a strategy.
    pub fn record(&self, strategy: &str, reward: f64) {
        let mut evals = self.evaluations.write();
        let entry = evals
            .entry(strategy.to_string())
            .or_insert_with(|| LearningStrategyEval::new(strategy));
        entry.record(reward);
    }

    /// Select the best strategy using UCB1 or epsilon-greedy.
    ///
    /// If no strategies have been registered, returns `None`.
    /// If any strategy has zero uses, returns that one first (exploration).
    #[must_use]
    pub fn select(&self) -> Option<StrategyName> {
        let evals = self.evaluations.read();
        if evals.is_empty() {
            return None;
        }
        // First, try any strategy with zero uses.
        for (name, e) in evals.iter() {
            if e.uses == 0 {
                return Some(name.clone());
            }
        }
        let total_uses: u64 = evals.values().map(|e| e.uses).sum();
        if total_uses == 0 {
            return Some(evals.keys().next().cloned()?);
        }
        // Epsilon-greedy: with probability epsilon, pick a random strategy.
        if !self.use_ucb1 && rand::random::<f64>() < self.epsilon {
            let names: Vec<_> = evals.keys().collect();
            let i = rand::random::<usize>() % names.len();
            return Some(names[i].clone());
        }
        // UCB1 selection.
        let mut best: Option<(StrategyName, f64)> = None;
        for (name, e) in evals.iter() {
            let exploit = e.mean_reward;
            let explore = self.ucb1_c * ((total_uses as f64).ln() / e.uses as f64).sqrt();
            let score = exploit + explore;
            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((name.clone(), score));
            }
        }
        best.map(|(n, _)| n)
    }

    /// Get the evaluation record for a strategy.
    #[must_use]
    pub fn get(&self, strategy: &str) -> Option<LearningStrategyEval> {
        self.evaluations.read().get(strategy).cloned()
    }

    /// List all registered strategies.
    #[must_use]
    pub fn strategies(&self) -> Vec<StrategyName> {
        self.evaluations.read().keys().cloned().collect()
    }

    /// Best strategy by mean reward (ignoring exploration).
    #[must_use]
    pub fn best_strategy(&self) -> Option<StrategyName> {
        let evals = self.evaluations.read();
        evals
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.mean_reward
                    .partial_cmp(&b.mean_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(n, _)| n.clone())
    }

    /// Number of registered strategies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.evaluations.read().len()
    }

    /// Whether the meta-learner has any strategies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.evaluations.read().is_empty()
    }
}

impl std::fmt::Debug for MetaLearner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaLearner")
            .field("strategies", &self.len())
            .field("epsilon", &self.epsilon)
            .field("use_ucb1", &self.use_ucb1)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_initializes_entry() {
        let m = MetaLearner::new();
        m.register("curriculum");
        assert_eq!(m.len(), 1);
        let e = m.get("curriculum").unwrap();
        assert_eq!(e.uses, 0);
    }

    #[test]
    fn record_updates_mean_reward() {
        let m = MetaLearner::new();
        m.record("a", 1.0);
        m.record("a", 0.5);
        m.record("a", 0.0);
        let e = m.get("a").unwrap();
        assert_eq!(e.uses, 3);
        assert!((e.mean_reward - 0.5).abs() < 1e-9);
    }

    #[test]
    fn select_returns_untried_strategy_first() {
        let m = MetaLearner::new();
        m.register("a");
        m.register("b");
        m.register("c");
        // 'a' should be returned first (zero uses).
        let s = m.select().unwrap();
        assert!(s == "a" || s == "b" || s == "c");
    }

    #[test]
    fn best_strategy_picks_highest_mean() {
        let m = MetaLearner::new();
        for _ in 0..10 {
            m.record("good", 0.9);
        }
        for _ in 0..10 {
            m.record("bad", 0.1);
        }
        let best = m.best_strategy().unwrap();
        assert_eq!(best, "good");
    }

    #[test]
    fn ucb1_balances_explore_exploit() {
        let m = MetaLearner::new();
        // Give 'a' many uses with mediocre reward.
        for _ in 0..100 {
            m.record("a", 0.5);
        }
        // 'b' has fewer uses but slightly higher mean.
        for _ in 0..5 {
            m.record("b", 0.6);
        }
        // UCB1 should sometimes pick 'b' for exploration.
        let mut b_count = 0;
        for _ in 0..20 {
            if let Some(s) = m.select() {
                if s == "b" {
                    b_count += 1;
                }
            }
        }
        // With UCB1, 'b' should be selected sometimes.
        assert!(b_count > 0);
    }
}
