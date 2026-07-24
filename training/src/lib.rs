//! QbitOS Training — Self-learning, self-improvement, and curriculum engines.
//!
//! Implements:
//!
//! 1. **Self-learning loop** — the agent generates its own training data
//!    from interactions, evaluates its performance, and updates its knowledge.
//! 2. **Curriculum learning** — progressive difficulty ordering of training
//!    tasks (easy → medium → hard).
//! 3. **Self-play** — the agent plays against itself to generate diverse
//!    training scenarios.
//! 4. **Self-improvement cycle** — detect weaknesses → generate fixes →
//!    test → accept/reject (with regression guard).
//! 5. **Experience replay** — sample from past experiences for training.
//! 6. **Meta-learning** — learn which learning strategies work best.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![allow(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod curriculum;
pub mod cusp;
pub mod experience_replay;
pub mod federated;
pub mod meta_learning;
pub mod self_improvement;
pub mod self_play;
pub mod training_loop;

pub use curriculum::{Curriculum, CurriculumStage, DifficultyLevel};
pub use cusp::{CuspConfig, CuspError, CuspGoal, CuspJudge, CuspOrchestrator, CuspRound, CuspSetter, CuspSolver, CuspVerifier, StubJudge, StubSetter, StubSolver, StubVerifier};
pub use experience_replay::{ExperienceReplay, Experience};
pub use federated::{FedConfig, FedCoordinator, FedError, FedRound, RoundState as FedRoundState, ParticipantInfo, ParticipantUpdate, SharedFedCoordinator, Weights};
pub use meta_learning::{MetaLearner, LearningStrategyEval};
pub use self_improvement::{SelfImprovementCycle, ImprovementProposal, WeaknessReport, PerformanceMetrics};
pub use self_play::{SelfPlayEngine, SelfPlayResult, GameOutcome};
pub use training_loop::{TrainingLoop, TrainingConfig, TrainingReport, TrainingStep};
