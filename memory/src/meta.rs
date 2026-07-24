//! L5 Meta Memory.
//!
//! Stores learning strategies, confidence calibration, and user preferences.
//! This layer drives the reflection engine: it tracks which learning
//! strategies worked well in the past and adjusts the kernel's behavior
//! accordingly.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::object::{Importance, MemoryId, MemoryObject};

/// A learning strategy: a named recipe for how to learn from a cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStrategy {
    /// Memory envelope.
    pub object: MemoryObject<serde_json::Value>,
    /// Strategy name (e.g. "rehearsal", "spaced_repetition", "analogical_transfer").
    pub name: String,
    /// Strategy parameters.
    pub parameters: serde_json::Value,
    /// Effectiveness in `[0.0, 1.0]`, updated by reflection.
    pub effectiveness: f32,
}

/// A user preference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    /// Memory envelope.
    pub object: MemoryObject<serde_json::Value>,
    /// Preference key (e.g. "response_tone", "language", "verbosity").
    pub key: String,
    /// Preference value.
    pub value: serde_json::Value,
}

/// L5 meta memory.
pub struct MetaMemory {
    strategies: Arc<RwLock<HashMap<String, LearningStrategy>>>,
    preferences: Arc<RwLock<HashMap<String, Preference>>>,
    /// Confidence calibration: how well-calibrated the kernel's own
    /// confidence estimates are. Values close to 1.0 mean well-calibrated.
    calibration: Arc<RwLock<f32>>,
}

impl MetaMemory {
    /// Create a new meta memory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategies: Arc::new(RwLock::new(HashMap::new())),
            preferences: Arc::new(RwLock::new(HashMap::new())),
            calibration: Arc::new(RwLock::new(0.5)),
        }
    }

    /// Register or update a learning strategy.
    pub fn upsert_strategy(&self, name: impl Into<String>, parameters: serde_json::Value, effectiveness: f32) -> MemoryId {
        let name = name.into();
        let mut object = MemoryObject::new(
            crate::object::MemoryType::Meta,
            parameters.clone(),
            crate::embedding::Embedding::new(vec![]),
        );
        object.importance = Importance(effectiveness);
        let id = object.id;
        let strategy = LearningStrategy { object, name: name.clone(), parameters, effectiveness };
        self.strategies.write().insert(name, strategy);
        id
    }

    /// Look up a strategy by name.
    #[must_use]
    pub fn strategy(&self, name: &str) -> Option<LearningStrategy> {
        self.strategies.read().get(name).cloned()
    }

    /// Get the most effective strategies.
    #[must_use]
    pub fn top_strategies(&self, k: usize) -> Vec<LearningStrategy> {
        let mut strategies: Vec<_> = self.strategies.read().values().cloned().collect();
        strategies.sort_by(|a, b| b.effectiveness.partial_cmp(&a.effectiveness).unwrap_or(std::cmp::Ordering::Equal));
        strategies.truncate(k);
        strategies
    }

    /// Set a user preference.
    pub fn set_preference(&self, key: impl Into<String>, value: serde_json::Value) -> MemoryId {
        let key = key.into();
        let mut object = MemoryObject::new(
            crate::object::MemoryType::Meta,
            value.clone(),
            crate::embedding::Embedding::new(vec![]),
        );
        object.importance = Importance(0.9);
        let id = object.id;
        let pref = Preference { object, key: key.clone(), value };
        self.preferences.write().insert(key, pref);
        id
    }

    /// Get a preference by key.
    #[must_use]
    pub fn preference(&self, key: &str) -> Option<serde_json::Value> {
        self.preferences.read().get(key).map(|p| p.value.clone())
    }

    /// Update the confidence calibration based on a new (predicted, actual)
    /// pair. Uses exponential moving average.
    pub fn update_calibration(&self, predicted: f32, actual: f32) {
        let error = (predicted - actual).abs();
        let current = *self.calibration.read();
        // Lower error → higher calibration
        let observation = 1.0 - error;
        let alpha = 0.05;
        let updated = (1.0 - alpha) * current + alpha * observation;
        *self.calibration.write() = updated;
    }

    /// Current calibration score.
    #[must_use]
    pub fn calibration(&self) -> f32 {
        *self.calibration.read()
    }
}

impl Default for MetaMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_lifecycle() {
        let mem = MetaMemory::new();
        mem.upsert_strategy("rehearsal", serde_json::json!({"repeats": 3}), 0.7);
        mem.upsert_strategy("analogical_transfer", serde_json::json!({"k": 5}), 0.9);
        let top = mem.top_strategies(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "analogical_transfer");
    }

    #[test]
    fn preference_set_get() {
        let mem = MetaMemory::new();
        mem.set_preference("language", serde_json::json!("en"));
        assert_eq!(mem.preference("language").unwrap(), serde_json::json!("en"));
    }

    #[test]
    fn calibration_ema() {
        let mem = MetaMemory::new();
        let initial = mem.calibration();
        for _ in 0..100 {
            mem.update_calibration(0.9, 0.9);
        }
        let final_cal = mem.calibration();
        assert!(final_cal > initial);
    }
}
