//! QbitOS Learning — Online learning and adaptation engines.
//!
//! Implements:
//!
//! 1. **Online learning** — incremental model updates from streaming feedback
//! 2. **Preference learning** — learn from pairwise comparisons (DPO-style)
//! 3. **Feedback loops** — collect, aggregate, and apply user feedback
//! 4. **Bandit algorithms** — multi-armed bandit for action selection
//! 5. **Reinforcement learning** — Q-learning for sequential decisions

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod bandit;
pub mod dpo;
pub mod feedback;
pub mod preference;
pub mod q_learning;

pub use bandit::{Bandit, Ucb1Bandit, EpsilonGreedyBandit, BanditArm};
pub use dpo::{DpoConfig, DpoError, DpoPair, DpoStats, DpoTrainer, SharedDpoTrainer};
pub use feedback::{FeedbackCollector, Feedback, FeedbackType};
pub use preference::{PreferenceLearner, PreferencePair, PreferenceModel};
pub use q_learning::{QLearner, QTable, QState, QAction};
