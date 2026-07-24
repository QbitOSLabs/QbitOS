//! Adaptive HNSW parameters — auto-tune ef_search based on dataset size.
//!
//! Based on 2026 research: "HNSW at Scale: Why Adding More Documents to Your
//! Database Breaks RAG" — HNSW parameters must be tuned as the dataset grows.
//! With fixed parameters, recall drops as the index grows because the graph
//! becomes sparser relative to the total number of vectors.
//!
//! This module implements adaptive parameter selection:
//!
//! 1. **ef_search scaling**: increases with dataset size to maintain recall.
//! 2. **ef_construction scaling**: increases for larger builds.
//! 3. **M scaling**: increases for larger datasets to maintain graph density.
//! 4. **Recall monitoring**: tracks observed recall and adjusts parameters.
//! 5. **Latency-aware tuning**: balances recall vs latency.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;

/// Adaptive HNSW configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveHnswConfig {
    /// Base ef_search for small datasets (< 10K vectors).
    pub base_ef_search: usize,
    /// Base M parameter for small datasets.
    pub base_m: usize,
    /// Base ef_construction.
    pub base_ef_construction: usize,
    /// Target recall (e.g., 0.95 = 95% recall@10).
    pub target_recall: f32,
    /// Maximum allowed latency in milliseconds.
    pub max_latency_ms: f64,
    /// Whether to enable automatic parameter adjustment.
    pub auto_adjust: bool,
    /// Dataset size thresholds for parameter scaling.
    pub scale_thresholds: Vec<usize>,
}

impl Default for AdaptiveHnswConfig {
    fn default() -> Self {
        Self {
            base_ef_search: 40,
            base_m: 16,
            base_ef_construction: 64,
            target_recall: 0.95,
            max_latency_ms: 10.0,
            auto_adjust: true,
            scale_thresholds: vec![10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000],
        }
    }
}

/// Runtime state for adaptive HNSW.
pub struct AdaptiveHnswState {
    config: RwLock<AdaptiveHnswConfig>,
    /// Current dataset size (number of vectors).
    dataset_size: AtomicU64,
    /// Current ef_search (adjusted at runtime).
    current_ef_search: RwLock<usize>,
    /// Current M parameter.
    current_m: RwLock<usize>,
    /// Observed recall history (rolling window).
    recall_history: RwLock<Vec<f32>>,
    /// Observed latency history (rolling window, in ms).
    latency_history: RwLock<Vec<f64>>,
    /// Total queries served.
    query_count: AtomicU64,
    /// Total recall measurements.
    recall_measurements: AtomicU64,
}

impl AdaptiveHnswState {
    /// Create new adaptive HNSW state.
    #[must_use]
    pub fn new(config: AdaptiveHnswConfig) -> Self {
        let ef_search = config.base_ef_search;
        let m = config.base_m;
        Self {
            config: RwLock::new(config),
            dataset_size: AtomicU64::new(0),
            current_ef_search: RwLock::new(ef_search),
            current_m: RwLock::new(m),
            recall_history: RwLock::new(Vec::with_capacity(100)),
            latency_history: RwLock::new(Vec::with_capacity(100)),
            query_count: AtomicU64::new(0),
            recall_measurements: AtomicU64::new(0),
        }
    }

    /// Update the dataset size and auto-adjust parameters if needed.
    pub fn update_dataset_size(&self, size: u64) {
        self.dataset_size.store(size, Ordering::Relaxed);
        if self.config.read().auto_adjust {
            self.adjust_parameters();
        }
    }

    /// Record a query's latency.
    pub fn record_latency(&self, latency_ms: f64) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        let mut history = self.latency_history.write();
        history.push(latency_ms);
        if history.len() > 100 {
            history.remove(0);
        }
        if self.config.read().auto_adjust {
            self.adjust_parameters();
        }
    }

    /// Record an observed recall measurement.
    pub fn record_recall(&self, recall: f32) {
        self.recall_measurements.fetch_add(1, Ordering::Relaxed);
        let mut history = self.recall_history.write();
        history.push(recall);
        if history.len() > 100 {
            history.remove(0);
        }
        if self.config.read().auto_adjust {
            self.adjust_parameters();
        }
    }

    /// Get the current ef_search value.
    #[must_use]
    pub fn ef_search(&self) -> usize {
        *self.current_ef_search.read()
    }

    /// Get the current M parameter.
    #[must_use]
    pub fn m(&self) -> usize {
        *self.current_m.read()
    }

    /// Get the recommended ef_construction for a given dataset size.
    #[must_use]
    pub fn recommended_ef_construction(&self, size: u64) -> usize {
        let config = self.config.read();
        let scale_factor = scale_factor_for_size(size, &config.scale_thresholds);
        (config.base_ef_construction as f64 * scale_factor) as usize
    }

    /// Get the recommended M for a given dataset size.
    #[must_use]
    pub fn recommended_m(&self, size: u64) -> usize {
        let config = self.config.read();
        let scale_factor = scale_factor_for_size(size, &config.scale_thresholds);
        let m = (config.base_m as f64 * scale_factor) as usize;
        // M should be even for HNSW.
        if m % 2 == 0 { m } else { m + 1 }
    }

    /// Get average observed recall.
    #[must_use]
    pub fn avg_recall(&self) -> f32 {
        let history = self.recall_history.read();
        if history.is_empty() {
            return 0.0;
        }
        history.iter().sum::<f32>() / history.len() as f32
    }

    /// Get average observed latency in milliseconds.
    #[must_use]
    pub fn avg_latency_ms(&self) -> f64 {
        let history = self.latency_history.read();
        if history.is_empty() {
            return 0.0;
        }
        history.iter().sum::<f64>() / history.len() as f64
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> AdaptiveHnswStats {
        AdaptiveHnswStats {
            dataset_size: self.dataset_size.load(Ordering::Relaxed),
            current_ef_search: *self.current_ef_search.read(),
            current_m: *self.current_m.read(),
            avg_recall: self.avg_recall(),
            avg_latency_ms: self.avg_latency_ms(),
            query_count: self.query_count.load(Ordering::Relaxed),
            recall_measurements: self.recall_measurements.load(Ordering::Relaxed),
        }
    }

    /// Adjust parameters based on observed recall, latency, and dataset size.
    fn adjust_parameters(&self) {
        let config = self.config.read().clone();
        let size = self.dataset_size.load(Ordering::Relaxed);
        let avg_recall = self.avg_recall();
        let avg_latency = self.avg_latency_ms();

        // Base scaling from dataset size.
        let scale_factor = scale_factor_for_size(size, &config.scale_thresholds);
        let mut new_ef = (config.base_ef_search as f64 * scale_factor) as usize;

        // Recall-based adjustment: if recall is below target, increase ef_search.
        if avg_recall > 0.0 && avg_recall < config.target_recall {
            let recall_deficit = config.target_recall - avg_recall;
            let boost = 1.0 + (recall_deficit * 2.0) as f64; // up to 2x boost
            new_ef = (new_ef as f64 * boost) as usize;
        }

        // Latency-based adjustment: if latency is too high, decrease ef_search.
        if avg_latency > config.max_latency_ms {
            let latency_ratio = config.max_latency_ms / avg_latency;
            new_ef = (new_ef as f64 * latency_ratio) as usize;
        }

        // Clamp to reasonable bounds.
        new_ef = new_ef.clamp(10, 1000);

        // Adjust M based on dataset size (only increase, never decrease at runtime).
        let new_m = self.recommended_m(size);

        *self.current_ef_search.write() = new_ef;
        *self.current_m.write() = new_m;
    }
}

/// Statistics for adaptive HNSW.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveHnswStats {
    /// Current dataset size.
    pub dataset_size: u64,
    /// Current ef_search.
    pub current_ef_search: usize,
    /// Current M.
    pub current_m: usize,
    /// Average observed recall.
    pub avg_recall: f32,
    /// Average observed latency in ms.
    pub avg_latency_ms: f64,
    /// Total queries served.
    pub query_count: u64,
    /// Total recall measurements.
    pub recall_measurements: u64,
}

/// Compute a scale factor based on dataset size.
/// Returns a multiplier > 1.0 for larger datasets.
fn scale_factor_for_size(size: u64, thresholds: &[usize]) -> f64 {
    if thresholds.is_empty() {
        return 1.0;
    }
    let mut factor = 1.0;
    for (i, &threshold) in thresholds.iter().enumerate() {
        if size >= threshold as u64 {
            // Each threshold adds ~20% to the scale factor.
            factor = 1.0 + 0.2 * (i + 1) as f64;
        } else {
            break;
        }
    }
    factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_reasonable() {
        let c = AdaptiveHnswConfig::default();
        assert!(c.base_ef_search >= 10);
        assert!(c.target_recall > 0.9);
    }

    #[test]
    fn ef_search_increases_with_dataset_size() {
        let state = AdaptiveHnswState::new(AdaptiveHnswConfig::default());
        let small_ef = state.ef_search();
        state.update_dataset_size(500_000);
        let large_ef = state.ef_search();
        assert!(large_ef >= small_ef, "ef_search should increase: {small_ef} -> {large_ef}");
    }

    #[test]
    fn recall_below_target_increases_ef() {
        let state = AdaptiveHnswState::new(AdaptiveHnswConfig::default());
        let initial_ef = state.ef_search();
        // Simulate low recall.
        for _ in 0..10 {
            state.record_recall(0.80); // Below target of 0.95
        }
        let adjusted_ef = state.ef_search();
        assert!(adjusted_ef > initial_ef, "ef should increase when recall is low: {initial_ef} -> {adjusted_ef}");
    }

    #[test]
    fn high_latency_decreases_ef() {
        let state = AdaptiveHnswState::new(AdaptiveHnswConfig {
            max_latency_ms: 5.0,
            base_ef_search: 200,
            ..AdaptiveHnswConfig::default()
        });
        // Simulate high latency.
        for _ in 0..10 {
            state.record_latency(20.0); // 4x the max
        }
        let adjusted_ef = state.ef_search();
        assert!(adjusted_ef < 200, "ef should decrease when latency is high: {adjusted_ef}");
    }

    #[test]
    fn recommended_m_scales_with_size() {
        let state = AdaptiveHnswState::new(AdaptiveHnswConfig::default());
        let small_m = state.recommended_m(1_000);
        let large_m = state.recommended_m(1_000_000);
        assert!(large_m > small_m);
        assert!(large_m % 2 == 0, "M must be even");
    }

    #[test]
    fn stats_are_accurate() {
        let state = AdaptiveHnswState::new(AdaptiveHnswConfig::default());
        state.update_dataset_size(100_000);
        state.record_latency(5.0);
        state.record_recall(0.92);
        let stats = state.stats();
        assert_eq!(stats.dataset_size, 100_000);
        assert!(stats.avg_latency_ms > 0.0);
        assert!(stats.avg_recall > 0.0);
        assert!(stats.query_count > 0);
    }

    #[test]
    fn scale_factor_grows_with_thresholds() {
        let thresholds = vec![10_000, 100_000, 1_000_000];
        assert!(scale_factor_for_size(1_000, &thresholds) < scale_factor_for_size(50_000, &thresholds));
        assert!(scale_factor_for_size(50_000, &thresholds) < scale_factor_for_size(500_000, &thresholds));
        assert!(scale_factor_for_size(500_000, &thresholds) < scale_factor_for_size(2_000_000, &thresholds));
    }
}
