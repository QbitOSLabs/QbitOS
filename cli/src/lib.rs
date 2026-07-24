//! QbitOS CLI — the top-level orchestrator.
//!
//! This crate exposes a single `qbitos` binary that wires every QbitOS
//! subsystem into a coherent runtime:
//!
//! - Cognitive kernel with the observe → understand → plan → execute →
//!   verify → learn → improve loop
//! - L0–L5 memory fabric with HNSW vector index + consolidation
//! - Native neural inference engine (candle-based)
//! - Vision pipeline (image → encoder → memory → reasoning)
//! - Speech pipeline (audio → STT → reasoning → TTS → audio)
//! - World model (knowledge graph + causal reasoning + simulation)
//! - Planner (goal engine + task graph + decision engine)
//! - Skill genome (registry + generator + optimizer + executor)
//! - Tool VM (filesystem, terminal, http, WASM sandbox)
//! - Evolution engine (evaluator + improvement + regression guard)
//! - Dataset loader (40+ formats)
//!
//! Two modes:
//!
//! 1. **Interactive REPL** (`qbitos`) — boots the kernel and drops into a
//!    prompt where the user can chat, query memory, inspect skills, etc.
//! 2. **Subcommand** (`qbitos <cmd>`) — runs one-shot operations like
//!    `dataset load`, `memory stats`, `skill list`, `evolution run`.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::new_without_default,
    clippy::struct_excessive_bools,
    dead_code,
    missing_docs
)]

pub mod app;
pub mod eval_harness;
pub mod observability;
pub mod cognitive_context;
pub mod commands;
pub mod persistence;
pub mod pipeline;
pub mod production_skills;

pub use app::{App, AppHandle};
pub use eval_harness::{CaseScore, EvalBackend, EvalCase, EvalDatasetRegistry, EvalError, EvalJudge, EvalOutput, EvalReport, EvalRunner, RuleBasedJudge, StubEvalBackend, TagScore};
pub use observability::{ObservabilityError, SharedTracer, Span, SpanEvent, SpanGuard, SpanId, SpanKind, SpanStatus, Trace, TraceId, Tracer};
pub use persistence::{Persistence, StateSnapshot};
pub use pipeline::{DatasetToMemory, DatasetToSkills, IngestResult};

/// Semantic version of the CLI.
pub const CLI_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize global tracing.
pub fn init_tracing() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,qbitos=info"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_file(false)
            .with_line_number(false)
            .with_ansi(std::env::var("NO_COLOR").is_err())
            .init();
    });
}
