#![allow(missing_docs)]
//! Learned memory importance: instead of static importance scores, learns
//! importance as a weighted combination of memory features. Weights are
//! updated via gradient descent on the squared error between predicted and
//! observed importance.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Scale used to encode the `retrieval_frequency` weight (a fractional value)
/// into the `u32` field of `MemoryFeatures` when used as a weights container.
const WEIGHT_SCALE: f64 = 10_000.0;
const DEFAULT_WEIGHT: f64 = 0.1667;
const LEARNING_RATE: f64 = 0.01;

#[derive(Debug, Error)]
pub enum ImportanceError { #[error("feature value out of range: {0}")] OutOfRange(String) }

/// Features describing a memory that predict its importance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFeatures {
    pub retrieval_frequency: u32,
    pub future_usefulness: f64,
    pub novelty: f64,
    pub recency_score: f64,
    pub user_feedback: f64,
    pub causal_impact: f64,
}

impl Default for MemoryFeatures {
    fn default() -> Self {
        Self { retrieval_frequency: 0, future_usefulness: 0.5, novelty: 0.5, recency_score: 0.5, user_feedback: 0.5, causal_impact: 0.5 }
    }
}

impl MemoryFeatures {
    #[must_use] pub fn new() -> Self { Self::default() }

    /// Vector representation of features (raw values).
    #[must_use]
    pub fn feature_vec(&self) -> Vec<f64> {
        vec![self.retrieval_frequency as f64, self.future_usefulness, self.novelty,
             self.recency_score, self.user_feedback, self.causal_impact]
    }

    /// When used as a weights container, `retrieval_frequency` is interpreted
    /// as `weight × WEIGHT_SCALE` (u32 storage of an f64 weight).
    #[must_use]
    pub fn weights_vec(&self) -> Vec<f64> {
        vec![self.retrieval_frequency as f64 / WEIGHT_SCALE, self.future_usefulness, self.novelty,
             self.recency_score, self.user_feedback, self.causal_impact]
    }
    fn set_weights(&mut self, w: &[f64]) {
        if w.len() >= 1 { self.retrieval_frequency = (w[0] * WEIGHT_SCALE).round().max(0.0) as u32; }
        if w.len() >= 2 { self.future_usefulness = w[1]; }
        if w.len() >= 3 { self.novelty = w[2]; }
        if w.len() >= 4 { self.recency_score = w[3]; }
        if w.len() >= 5 { self.user_feedback = w[4]; }
        if w.len() >= 6 { self.causal_impact = w[5]; }
    }
    fn default_weights() -> Self {
        Self {
            retrieval_frequency: (DEFAULT_WEIGHT * WEIGHT_SCALE).round() as u32,
            future_usefulness: DEFAULT_WEIGHT, novelty: DEFAULT_WEIGHT,
            recency_score: DEFAULT_WEIGHT, user_feedback: DEFAULT_WEIGHT, causal_impact: DEFAULT_WEIGHT,
        }
    }
}

/// Linear importance model: importance = w · features.
pub struct ImportanceModel {
    weights: Arc<RwLock<MemoryFeatures>>,
    samples: Arc<RwLock<Vec<(MemoryFeatures, f64)>>>,
    learning_rate: f64,
}

impl Default for ImportanceModel { fn default() -> Self { Self::new() } }

impl ImportanceModel {
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: Arc::new(RwLock::new(MemoryFeatures::default_weights())),
            samples: Arc::new(RwLock::new(Vec::new())),
            learning_rate: LEARNING_RATE,
        }
    }
    #[must_use] pub fn with_learning_rate(mut self, lr: f64) -> Self { self.learning_rate = lr; self }

    #[must_use]
    pub fn compute_importance(&self, features: &MemoryFeatures) -> f64 {
        let w = self.weights.read().weights_vec();
        let f = features.feature_vec();
        w.iter().zip(f.iter()).map(|(wi, fi)| wi * fi).sum()
    }

    /// Train on a single (features, actual_importance) sample. Updates weights
    /// via one step of gradient descent on MSE.
    pub fn train(&self, features: MemoryFeatures, actual_importance: f64) {
        self.samples.write().push((features.clone(), actual_importance));
        let predicted = self.compute_importance(&features);
        let error = predicted - actual_importance;
        let f = features.feature_vec();
        let mut w = self.weights.write();
        let mut wv = w.weights_vec();
        for (wi, fi) in wv.iter_mut().zip(f.iter()) {
            *wi -= self.learning_rate * 2.0 * error * fi;
        }
        w.set_weights(&wv);
    }
    #[must_use] pub fn get_weights(&self) -> MemoryFeatures { self.weights.read().clone() }
    #[must_use] pub fn sample_count(&self) -> usize { self.samples.read().len() }
    #[must_use] pub fn learning_rate(&self) -> f64 { self.learning_rate }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_weights_sum_near_one() {
        let m = ImportanceModel::new();
        let sum: f64 = m.get_weights().weights_vec().iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum was {sum}");
    }
    #[test]
    fn compute_importance_zero_for_zero_features() {
        let m = ImportanceModel::new();
        let f = MemoryFeatures { retrieval_frequency: 0, future_usefulness: 0.0, novelty: 0.0, recency_score: 0.0, user_feedback: 0.0, causal_impact: 0.0 };
        assert!((m.compute_importance(&f) - 0.0).abs() < 1e-9);
    }
    #[test]
    fn train_reduces_prediction_error() {
        let m = ImportanceModel::new();
        let f = MemoryFeatures { retrieval_frequency: 2, future_usefulness: 0.8, novelty: 0.3, recency_score: 0.6, user_feedback: 0.7, causal_impact: 0.4 };
        let target = 1.0;
        let initial_err = (m.compute_importance(&f) - target).abs();
        for _ in 0..10 { m.train(f.clone(), target); }
        let final_err = (m.compute_importance(&f) - target).abs();
        assert!(final_err <= initial_err + 1e-6, "final err {final_err} > initial err {initial_err}");
    }
    #[test]
    fn sample_count_increments() {
        let m = ImportanceModel::new();
        assert_eq!(m.sample_count(), 0);
        m.train(MemoryFeatures::default(), 0.5);
        assert_eq!(m.sample_count(), 1);
    }
    #[test]
    fn custom_learning_rate_is_applied() {
        let m = ImportanceModel::new().with_learning_rate(0.1);
        assert!((m.learning_rate() - 0.1).abs() < 1e-9);
    }
}
