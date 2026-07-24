//! Experience replay: sample from past experiences for training.

use parking_lot::RwLock;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A single experience tuple: (state, action, reward, next_state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// The state observed before the action.
    pub state: serde_json::Value,
    /// The action taken.
    pub action: serde_json::Value,
    /// The reward received.
    pub reward: f64,
    /// The next state observed after the action.
    pub next_state: serde_json::Value,
    /// Whether the next state is terminal.
    #[serde(default)]
    pub terminal: bool,
    /// Priority weight (for prioritized replay). Higher = more important.
    #[serde(default = "default_priority")]
    pub priority: f64,
    /// Timestamp of the experience.
    #[serde(default)]
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

fn default_priority() -> f64 {
    1.0
}

impl Experience {
    /// Construct a new experience.
    #[must_use]
    pub fn new(
        state: serde_json::Value,
        action: serde_json::Value,
        reward: f64,
        next_state: serde_json::Value,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            terminal: false,
            priority: 1.0,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Mark the experience as terminal.
    #[must_use]
    pub fn terminal(mut self) -> Self {
        self.terminal = true;
        self
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority;
        self
    }
}

/// The experience replay buffer.
pub struct ExperienceReplay {
    buffer: Arc<RwLock<Vec<Experience>>>,
    capacity: usize,
    alpha: f64, // Priority exponent (0 = uniform, 1 = full prioritization)
}

impl ExperienceReplay {
    /// Construct a new replay buffer with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(Vec::with_capacity(capacity))),
            capacity,
            alpha: 0.6,
        }
    }

    /// Set the priority exponent (0 = uniform, 1 = full prioritization).
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Add an experience to the buffer.
    pub fn add(&self, experience: Experience) {
        let mut buf = self.buffer.write();
        if buf.len() >= self.capacity {
            buf.remove(0);
        }
        buf.push(experience);
    }

    /// Add multiple experiences.
    pub fn extend<I: IntoIterator<Item = Experience>>(&self, experiences: I) {
        let mut buf = self.buffer.write();
        for exp in experiences {
            if buf.len() >= self.capacity {
                buf.remove(0);
            }
            buf.push(exp);
        }
    }

    /// Sample a uniform random batch of experiences.
    #[must_use]
    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let buf = self.buffer.read();
        let n = buf.len().min(batch_size);
        let mut rng = rand::thread_rng();
        buf.choose_multiple(&mut rng, n).cloned().collect()
    }

    /// Sample a prioritized batch (favors high-priority experiences).
    #[must_use]
    pub fn sample_prioritized(&self, batch_size: usize) -> Vec<Experience> {
        let buf = self.buffer.read();
        let n = buf.len().min(batch_size);
        if n == 0 {
            return Vec::new();
        }
        let total_priority: f64 = buf.iter().map(|e| e.priority.powf(self.alpha)).sum();
        if total_priority <= 0.0 {
            return self.sample(batch_size);
        }
        let mut rng = rand::thread_rng();
        let mut out = Vec::with_capacity(n);
        let mut chosen_indices = std::collections::HashSet::new();
        while out.len() < n {
            let r: f64 = rng.gen_range(0.0..total_priority);
            let mut acc = 0.0;
            for (i, exp) in buf.iter().enumerate() {
                acc += exp.priority.powf(self.alpha);
                if acc >= r && chosen_indices.insert(i) {
                    out.push(exp.clone());
                    break;
                }
            }
        }
        out
    }

    /// Number of experiences in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.read().len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.read().is_empty()
    }

    /// Clear the buffer.
    pub fn clear(&self) {
        self.buffer.write().clear();
    }

    /// Average reward across all experiences.
    #[must_use]
    pub fn mean_reward(&self) -> f64 {
        let buf = self.buffer.read();
        if buf.is_empty() {
            return 0.0;
        }
        buf.iter().map(|e| e.reward).sum::<f64>() / buf.len() as f64
    }

    /// Update the priority of an experience by index.
    pub fn update_priority(&self, index: usize, priority: f64) {
        let mut buf = self.buffer.write();
        if let Some(e) = buf.get_mut(index) {
            e.priority = priority;
        }
    }
}

impl std::fmt::Debug for ExperienceReplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExperienceReplay")
            .field("len", &self.len())
            .field("capacity", &self.capacity)
            .field("alpha", &self.alpha)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exp(reward: f64) -> Experience {
        Experience::new(
            serde_json::json!({"s": 1}),
            serde_json::json!({"a": "go"}),
            reward,
            serde_json::json!({"s": 2}),
        )
    }

    #[test]
    fn add_and_len() {
        let r = ExperienceReplay::new(10);
        assert_eq!(r.len(), 0);
        r.add(make_exp(1.0));
        r.add(make_exp(2.0));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let r = ExperienceReplay::new(3);
        for i in 0..5 {
            r.add(make_exp(i as f64));
        }
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn sample_returns_at_most_batch() {
        let r = ExperienceReplay::new(10);
        for i in 0..5 {
            r.add(make_exp(i as f64));
        }
        let batch = r.sample(3);
        assert_eq!(batch.len(), 3);
        let batch = r.sample(10);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn mean_reward_correct() {
        let r = ExperienceReplay::new(10);
        r.add(make_exp(1.0));
        r.add(make_exp(2.0));
        r.add(make_exp(3.0));
        assert!((r.mean_reward() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn prioritized_sample_prefers_high_priority() {
        let r = ExperienceReplay::new(10).with_alpha(1.0);
        r.add(make_exp(1.0).with_priority(0.01));
        r.add(make_exp(2.0).with_priority(0.01));
        r.add(make_exp(3.0).with_priority(100.0));
        // Sample many times; the high-priority one should dominate.
        let mut high_count = 0;
        for _ in 0..30 {
            let batch = r.sample_prioritized(1);
            if let Some(e) = batch.first() {
                if (e.reward - 3.0).abs() < 1e-9 {
                    high_count += 1;
                }
            }
        }
        assert!(high_count > 20);
    }
}
