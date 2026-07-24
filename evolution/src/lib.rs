//! QbitOS Evolution Engine
//!
//! Controlled self-improvement. The engine detects weaknesses in the
//! system's behavior, proposes improvements, runs them in a sandbox,
//! benchmarks the results, and accepts or rejects the change.
//!
//! ```text
//! Detect Weakness → Generate Improvement → Run Benchmark
//!   → Compare Result → Accept or Reject
//! ```

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod evaluator;
pub mod neuroevolution;
pub mod improvement_engine;
pub mod regression_guard;

pub use evaluator::{Benchmark, BenchmarkResult, Evaluator, Metric, MetricKind};
pub use neuroevolution::{BehaviorDescriptor, Individual, NeuroEvoConfig, NeuroEvoError, QdLlmEngine, SharedQdLlmEngine};
pub use improvement_engine::{Improvement, ImprovementEngine, ImprovementKind};
pub use regression_guard::{Regression, RegressionGuard, RegressionReport};

/// Re-export of the composite evolution module.
pub use evolution_module::EvolutionModule;

mod evolution_module;
