//! QbitOS Skills — the skill genome subsystem.
//!
//! A "skill" is a reusable, parameterized procedure that the cognitive kernel
//! can invoke to accomplish a goal. Skills are composed of ordered steps
//! (Read → Infer → Tool → Verify → Return) and tracked with confidence
//! scores that update after each execution.
//!
//! # Architecture
//!
//! - [`SkillModule`] — top-level handle, wires together the registry,
//!   generator, optimizer, and executor.
//! - [`SkillRegistry`] — stores skills, tracks execution stats, recomputes
//!   confidence with a Beta(2,2) prior.
//! - [`Skill`] — a named procedure with ordered [`SkillStep`]s.
//! - [`SkillExecutor`] — runs a skill's steps in sequence.
//! - [`SkillGenerator`] — synthesizes new skills from templates.
//! - [`SkillOptimizer`] — prunes low-performing skills, suggests improvements.
//! - [`ToolBackend`] / [`InferBackend`] — traits the executor calls into.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::new_without_default,
    missing_docs
)]

pub mod backends;
pub mod composition;
pub mod executor;
pub mod generator;
pub mod optimizer;
pub mod registry;
pub mod signatures;

pub use backends::{InferBackend, NoopInferBackend, NoopToolBackend, ToolBackend};
pub use composition::{PipelineResult, PipelineStep, SkillPipeline};
pub use executor::{ExecutionReport, SkillExecutor};
pub use generator::{GeneratedSkill, SkillGenerator};
pub use optimizer::{OptimizationResult, SkillOptimizer};
pub use registry::{Skill, SkillId, SkillRegistry, SkillStep, StepKind};

use parking_lot::RwLock;
use std::sync::Arc;

/// Top-level handle for the skills subsystem.
pub struct SkillModule {
    /// The skill registry (shared, thread-safe).
    pub registry: Arc<SkillRegistry>,
    /// The episodic memory (for recording execution outcomes).
    pub episodic: Arc<RwLock<qbitos_memory::EpisodicMemory>>,
}

impl SkillModule {
    /// Create a new skill module with the given memory and noop backends.
    #[must_use]
    pub fn new(episodic: Arc<RwLock<qbitos_memory::EpisodicMemory>>) -> Self {
        Self {
            registry: Arc::new(SkillRegistry::new()),
            episodic,
        }
    }

    /// Create a new skill module with noop backends (for testing).
    #[must_use]
    pub fn with_noop_backends() -> Self {
        use qbitos_memory::{EmbeddingModelStub, SharedEmbeddingModel};
        use std::sync::Arc as StdArc;
        let embedder: SharedEmbeddingModel = StdArc::new(EmbeddingModelStub::new(384));
        let memory = Arc::new(RwLock::new(qbitos_memory::EpisodicMemory::new(embedder)));
        Self::new(memory)
    }

    /// Get a reference to the registry.
    #[must_use]
    pub fn registry(&self) -> &SkillRegistry {
        &self.registry
    }
}
