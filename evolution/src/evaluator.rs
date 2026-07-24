//! Evaluator.
//!
//! Runs benchmarks and collects metrics. A benchmark is a structured task
//! with a known ground-truth; metrics measure how well the system performed.

use ahash::AHashMap;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// The kind of metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    /// Accuracy (fraction of correct answers).
    Accuracy,
    /// Latency (in milliseconds).
    LatencyMs,
    /// Throughput (in tokens per second).
    ThroughputTps,
    /// Memory usage (in bytes).
    MemoryBytes,
    /// Token count.
    TokenCount,
    /// Custom metric.
    Custom,
}

/// A metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Metric name.
    pub name: String,
    /// Metric kind.
    pub kind: MetricKind,
    /// Value.
    pub value: f64,
    /// Unit (e.g. "ms", "tokens", "%").
    pub unit: String,
}

/// A benchmark: a structured task with a known ground-truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmark {
    /// Unique id.
    pub id: Uuid,
    /// Name.
    pub name: String,
    /// Description.
    pub description: String,
    /// The task prompt.
    pub prompt: String,
    /// The expected output (ground truth).
    pub expected: serde_json::Value,
    /// Evaluation function name (e.g. "exact_match", "f1", "bleu").
    pub evaluation: String,
}

/// A benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark id.
    pub benchmark_id: Uuid,
    /// Actual output.
    pub actual: serde_json::Value,
    /// Computed metrics.
    pub metrics: Vec<Metric>,
    /// Whether the benchmark passed.
    pub passed: bool,
    /// When the benchmark was run.
    pub run_at: DateTime<Utc>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// The evaluator: runs benchmarks and collects results.
pub struct Evaluator {
    benchmarks: Arc<RwLock<AHashMap<Uuid, Benchmark>>>,
    results: Arc<RwLock<Vec<BenchmarkResult>>>,
}

impl Evaluator {
    /// Create a new empty evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            benchmarks: Arc::new(RwLock::new(AHashMap::new())),
            results: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a benchmark.
    pub fn register(&self, benchmark: Benchmark) -> Uuid {
        let id = benchmark.id;
        self.benchmarks.write().insert(id, benchmark);
        id
    }

    /// Run a benchmark against actual output. Computes metrics using the
    /// benchmark's evaluation function.
    pub fn run(&self, benchmark_id: Uuid, actual: serde_json::Value) -> Option<BenchmarkResult> {
        let start = std::time::Instant::now();
        let benchmark = self.benchmarks.read().get(&benchmark_id).cloned()?;
        let metrics = Self::evaluate(&benchmark, &actual);
        let passed = metrics.iter().any(|m| m.name == "accuracy" && m.value >= 0.5);
        let result = BenchmarkResult {
            benchmark_id,
            actual,
            metrics,
            passed,
            run_at: Utc::now(),
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        };
        self.results.write().push(result.clone());
        Some(result)
    }

    fn evaluate(benchmark: &Benchmark, actual: &serde_json::Value) -> Vec<Metric> {
        let mut metrics = Vec::new();
        match benchmark.evaluation.as_str() {
            "exact_match" => {
                let match_score = if actual == &benchmark.expected { 1.0 } else { 0.0 };
                metrics.push(Metric { name: "accuracy".into(), kind: MetricKind::Accuracy, value: match_score, unit: "%".into() });
            }
            "f1" => {
                // Token-level F1
                let expected_tokens: std::collections::HashSet<String> = serde_json::to_string(&benchmark.expected)
                    .unwrap_or_default()
                    .split_whitespace()
                    .map(String::from)
                    .collect();
                let actual_tokens: std::collections::HashSet<String> = serde_json::to_string(actual)
                    .unwrap_or_default()
                    .split_whitespace()
                    .map(String::from)
                    .collect();
                let tp = expected_tokens.intersection(&actual_tokens).count();
                let precision = if actual_tokens.is_empty() { 0.0 } else { tp as f64 / actual_tokens.len() as f64 };
                let recall = if expected_tokens.is_empty() { 0.0 } else { tp as f64 / expected_tokens.len() as f64 };
                let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
                metrics.push(Metric { name: "f1".into(), kind: MetricKind::Accuracy, value: f1, unit: "%".into() });
            }
            "contains" => {
                let expected_str = serde_json::to_string(&benchmark.expected).unwrap_or_default();
                let actual_str = serde_json::to_string(actual).unwrap_or_default();
                let score = if actual_str.contains(&expected_str) { 1.0 } else { 0.0 };
                metrics.push(Metric { name: "accuracy".into(), kind: MetricKind::Accuracy, value: score, unit: "%".into() });
            }
            _ => {
                metrics.push(Metric { name: "custom".into(), kind: MetricKind::Custom, value: 0.0, unit: "".into() });
            }
        }
        metrics
    }

    /// Get all results.
    #[must_use]
    pub fn results(&self) -> Vec<BenchmarkResult> {
        self.results.read().clone()
    }

    /// Get the mean accuracy across all results.
    #[must_use]
    pub fn mean_accuracy(&self) -> f64 {
        let results = self.results.read();
        if results.is_empty() {
            return 0.0;
        }
        let total: f64 = results.iter()
            .filter_map(|r| r.metrics.iter().find(|m| m.name == "accuracy").map(|m| m.value))
            .sum();
        let count = results.iter().filter(|r| r.metrics.iter().any(|m| m.name == "accuracy")).count();
        if count == 0 { 0.0 } else { total / count as f64 }
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_benchmark() {
        let evaluator = Evaluator::new();
        let benchmark = Benchmark {
            id: Uuid::now_v7(),
            name: "test".into(),
            description: "test".into(),
            prompt: "What is 2+2?".into(),
            expected: serde_json::json!("4"),
            evaluation: "exact_match".into(),
        };
        let id = evaluator.register(benchmark);
        let result = evaluator.run(id, serde_json::json!("4")).unwrap();
        assert!(result.passed);
        let failed = evaluator.run(id, serde_json::json!("5")).unwrap();
        assert!(!failed.passed);
    }
}
