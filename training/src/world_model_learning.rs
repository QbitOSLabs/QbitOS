#![allow(missing_docs)]
//! Predictive world model: learns (state, action) → predicted_next_state.
//! The prediction error becomes a training signal; predictions are refined
//! via an exponential moving average when the actual next state is observed.
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorldModelError { #[error("invalid state representation: {0}")] InvalidState(String) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub state_hash: String, pub action: String, pub predicted_next: serde_json::Value,
    pub actual_next: Option<serde_json::Value>, pub prediction_error: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct WorldModelLearner {
    predictions_made: AtomicU64,
    correct_predictions: AtomicU64,
    model: Arc<RwLock<AHashMap<String, StateTransition>>>,
    ema_alpha: f64,
    similarity_threshold: f64,
}

impl Default for WorldModelLearner { fn default() -> Self { Self::new() } }

impl WorldModelLearner {
    #[must_use]
    pub fn new() -> Self {
        Self {
            predictions_made: AtomicU64::new(0), correct_predictions: AtomicU64::new(0),
            model: Arc::new(RwLock::new(AHashMap::new())),
            ema_alpha: 0.3, similarity_threshold: 0.9,
        }
    }
    #[must_use] pub fn with_ema_alpha(mut self, a: f64) -> Self { self.ema_alpha = a.clamp(0.0, 1.0); self }
    #[must_use] pub fn with_similarity_threshold(mut self, t: f64) -> Self { self.similarity_threshold = t.clamp(0.0, 1.0); self }

    fn hash_key(state: &serde_json::Value, action: &str) -> String {
        let mut h = DefaultHasher::new();
        state.to_string().hash(&mut h);
        action.hash(&mut h);
        format!("{:016x}", h.finish())
    }

    #[must_use]
    pub fn predict(&self, state: &serde_json::Value, action: &str) -> Option<serde_json::Value> {
        self.predictions_made.fetch_add(1, Ordering::Relaxed);
        self.model.read().get(&Self::hash_key(state, action)).map(|t| t.predicted_next.clone())
    }

    pub fn update(&self, state: &serde_json::Value, action: &str, actual_next: serde_json::Value) -> f64 {
        let key = Self::hash_key(state, action);
        let mut model = self.model.write();
        let now = chrono::Utc::now();
        match model.get(&key).map(|t| t.predicted_next.clone()) {
            Some(prev) => {
                let sim = similarity(&prev, &actual_next);
                let err = (1.0 - sim).max(0.0);
                if sim >= self.similarity_threshold {
                    self.correct_predictions.fetch_add(1, Ordering::Relaxed);
                }
                let blended = ema_blend(&prev, &actual_next, self.ema_alpha);
                model.insert(key.clone(), StateTransition {
                    state_hash: key, action: action.to_string(), predicted_next: blended,
                    actual_next: Some(actual_next), prediction_error: err, timestamp: now,
                });
                err
            }
            None => {
                model.insert(key.clone(), StateTransition {
                    state_hash: key, action: action.to_string(),
                    predicted_next: actual_next.clone(), actual_next: Some(actual_next),
                    prediction_error: 0.0, timestamp: now,
                });
                0.0
            }
        }
    }

    #[must_use] pub fn accuracy(&self) -> f64 {
        let made = self.predictions_made.load(Ordering::Relaxed);
        if made == 0 { 0.0 } else { self.correct_predictions.load(Ordering::Relaxed) as f64 / made as f64 }
    }
    #[must_use] pub fn model_size(&self) -> usize { self.model.read().len() }
    #[must_use] pub fn predictions_made(&self) -> u64 { self.predictions_made.load(Ordering::Relaxed) }
    #[must_use] pub fn correct_predictions(&self) -> u64 { self.correct_predictions.load(Ordering::Relaxed) }
}

impl std::fmt::Debug for WorldModelLearner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorldModelLearner")
            .field("predictions_made", &self.predictions_made())
            .field("correct_predictions", &self.correct_predictions())
            .field("model_size", &self.model_size())
            .field("accuracy", &self.accuracy())
            .finish()
    }
}

fn ema_blend(prev: &serde_json::Value, obs: &serde_json::Value, alpha: f64) -> serde_json::Value {
    match (prev, obs) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) =>
            match (a.as_f64(), b.as_f64()) {
                (Some(af), Some(bf)) => serde_json::json!(alpha * bf + (1.0 - alpha) * af),
                _ => obs.clone(),
            },
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            let mut out = serde_json::Map::new();
            for (k, v) in a {
                if let Some(bv) = b.get(k) { out.insert(k.clone(), ema_blend(v, bv, alpha)); }
                else { out.insert(k.clone(), v.clone()); }
            }
            for (k, v) in b { if !out.contains_key(k) { out.insert(k.clone(), v.clone()); } }
            serde_json::Value::Object(out)
        }
        _ => obs.clone(),
    }
}

fn similarity(a: &serde_json::Value, b: &serde_json::Value) -> f64 {
    if a == b { return 1.0; }
    match (a, b) {
        (serde_json::Value::Object(ma), serde_json::Value::Object(mb)) => {
            let total = ma.len().max(mb.len()) as f64;
            if total == 0.0 { 1.0 } else {
                ma.iter().map(|(k, va)| mb.get(k).map(|vb| similarity(va, vb) / total).unwrap_or(0.0)).sum()
            }
        }
        (serde_json::Value::Array(aa), serde_json::Value::Array(ab)) => {
            let n = aa.len().max(ab.len()) as f64;
            if n == 0.0 { 1.0 } else { (0..aa.len().min(ab.len())).map(|i| similarity(&aa[i], &ab[i])).sum::<f64>() / n }
        }
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            match (a.as_f64(), b.as_f64()) {
                (Some(af), Some(bf)) => 1.0 - ((af - bf).abs() / af.abs().max(bf.abs()).max(1e-9)).min(1.0),
                _ => 0.0,
            }
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn predict_returns_none_for_unseen_state() {
        let m = WorldModelLearner::new();
        assert!(m.predict(&json!({"x": 1}), "go").is_none());
        assert_eq!(m.predictions_made(), 1);
        assert!((m.accuracy() - 0.0).abs() < 1e-9);
    }
    #[test]
    fn update_stores_first_observation() {
        let m = WorldModelLearner::new();
        let err = m.update(&json!({"x": 1}), "go", json!({"y": 2}));
        assert!((err - 0.0).abs() < 1e-9);
        assert_eq!(m.model_size(), 1);
        assert_eq!(m.predict(&json!({"x": 1}), "go"), Some(json!({"y": 2})));
    }
    #[test]
    fn update_refines_numeric_prediction_with_ema() {
        let m = WorldModelLearner::new().with_ema_alpha(0.5);
        m.update(&json!({"s": 1}), "a", json!({"v": 10.0}));
        let err = m.update(&json!({"s": 1}), "a", json!({"v": 20.0}));
        assert!((err - 0.5).abs() < 1e-6);
        let p = m.predict(&json!({"s": 1}), "a").unwrap();
        assert!((p["v"].as_f64().unwrap() - 15.0).abs() < 1e-6);
    }
    #[test]
    fn accuracy_increases_with_matching_observations() {
        let m = WorldModelLearner::new();
        m.update(&json!({"s": 1}), "a", json!({"v": 10}));
        let _ = m.predict(&json!({"s": 1}), "a");
        m.update(&json!({"s": 1}), "a", json!({"v": 10}));
        assert!(m.accuracy() > 0.0);
    }
    #[test]
    fn distinct_state_action_pairs_get_distinct_entries() {
        let m = WorldModelLearner::new();
        m.update(&json!({"s": 1}), "a", json!({}));
        m.update(&json!({"s": 2}), "a", json!({}));
        m.update(&json!({"s": 1}), "b", json!({}));
        assert_eq!(m.model_size(), 3);
    }
}
