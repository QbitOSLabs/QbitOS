//! Regression guard.
//!
//! Before an improvement is accepted, the regression guard runs the full
//! benchmark suite and checks that no previously-passing benchmark now
//! fails. If any regression is detected, the improvement is rejected.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::evaluator::{BenchmarkResult, Evaluator};
use crate::improvement_engine::Improvement;

/// A detected regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    /// Benchmark id.
    pub benchmark_id: Uuid,
    /// Previous accuracy.
    pub previous: f64,
    /// Current accuracy.
    pub current: f64,
    /// Drop in accuracy.
    pub drop: f64,
    /// When the regression was detected.
    pub detected_at: DateTime<Utc>,
}

/// A regression report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Improvement id.
    pub improvement_id: Uuid,
    /// Detected regressions.
    pub regressions: Vec<Regression>,
    /// Whether the improvement is safe to apply.
    pub safe: bool,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// The regression guard.
pub struct RegressionGuard {
    evaluator: Arc<Evaluator>,
    baseline: Arc<RwLock<Vec<BenchmarkResult>>>,
}

impl RegressionGuard {
    /// Create a new guard.
    #[must_use]
    pub fn new(evaluator: Arc<Evaluator>) -> Self {
        Self { evaluator, baseline: Arc::new(RwLock::new(Vec::new())) }
    }

    /// Snapshot the current benchmark results as the baseline.
    pub fn snapshot_baseline(&self) {
        *self.baseline.write() = self.evaluator.results();
    }

    /// Check whether applying `improvement` would cause regressions.
    ///
    /// Compares the current benchmark results against the baseline. Any
    /// benchmark that previously passed but now fails is a regression.
    #[must_use]
    pub fn check(&self, improvement: &Improvement) -> RegressionReport {
        let start = std::time::Instant::now();
        let baseline = self.baseline.read();
        let current = self.evaluator.results();

        let mut regressions = Vec::new();
        for prev in baseline.iter() {
            if !prev.passed {
                continue;
            }
            // Find the current result for the same benchmark
            let curr = current.iter().find(|r| r.benchmark_id == prev.benchmark_id);
            if let Some(curr) = curr {
                let prev_acc = prev.metrics.iter().find(|m| m.name == "accuracy").map(|m| m.value).unwrap_or(0.0);
                let curr_acc = curr.metrics.iter().find(|m| m.name == "accuracy").map(|m| m.value).unwrap_or(0.0);
                if curr_acc < prev_acc {
                    regressions.push(Regression {
                        benchmark_id: prev.benchmark_id,
                        previous: prev_acc,
                        current: curr_acc,
                        drop: prev_acc - curr_acc,
                        detected_at: Utc::now(),
                    });
                }
            }
        }

        let safe = regressions.is_empty();
        RegressionReport {
            improvement_id: improvement.id,
            regressions,
            safe,
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        }
    }
}
