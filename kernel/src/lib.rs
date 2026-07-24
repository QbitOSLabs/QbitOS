//! QbitOS Cognitive Kernel
//!
//! The kernel is the brain of the operating system. It runs the cognitive loop:
//!
//! ```text
//! Observe → Understand → Create Goal → Plan → Execute → Verify → Learn → Improve
//! ```
//!
//! Each stage emits events on the global [`event_system::EventBus`] and writes
//! structured state into the [`memory`] fabric. The [`scheduler::Scheduler`]
//! drives the loop with priority-aware, preemptive scheduling across multiple
//! cognitive agents.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate)]

pub mod config;
pub mod reasoning_kernel;
pub mod cognitive_kernel;
pub mod event_system;
pub mod reliability;
pub mod resource_manager;
pub mod runtime;
pub mod scheduler;

pub use cognitive_kernel::{CognitiveKernel, CognitiveStage, CognitiveState};
pub use event_system::{Event, EventBus, EventKind, EventId, SubscriptionId};
pub use resource_manager::{ResourceBudget, ResourceManager, ResourceKind};
pub use runtime::{KernelRuntime, RuntimeHandle};
pub use scheduler::{Priority, Scheduler, TaskId, TaskStatus};
pub use reliability::CircuitBreaker;

/// Semantic version of the kernel ABI.
pub const KERNEL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Bootstraps the global tracing subscriber with a JSON event filter suitable
/// for the kernel. Safe to call multiple times — subsequent calls are no-ops.
///
/// Honors the `QBITOS_LOG_FORMAT` environment variable:
/// - `pretty` (default) — human-readable colored output.
/// - `json` — machine-readable JSON (one object per line, ideal for log aggregation).
/// - `compact` — single-line compact format.
pub fn init_tracing() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,qbitos=debug"));
        let format = std::env::var("QBITOS_LOG_FORMAT")
            .unwrap_or_else(|_| "pretty".into())
            .to_ascii_lowercase();
        let use_ansi = std::env::var("NO_COLOR").is_err();
        let subscriber = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_ansi(use_ansi);
        match format.as_str() {
            "json" => subscriber.json().init(),
            "compact" => subscriber.compact().init(),
            _ => subscriber.pretty().init(),
        }
    });
}
