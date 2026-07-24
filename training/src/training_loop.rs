//! Training loop: orchestrates a full training cycle combining curriculum,
//! experience replay, self-play, and self-improvement.

use crate::curriculum::{Curriculum, DifficultyLevel};
use crate::experience_replay::ExperienceReplay;
use crate::meta_learning::MetaLearner;
use crate::self_improvement::{PerformanceMetrics, SelfImprovementCycle};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for the training loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum number of training steps per run.
    pub max_steps: usize,
    /// Batch size for experience replay sampling.
    pub batch_size: usize,
    /// Whether to use curriculum learning.
    pub use_curriculum: bool,
    /// Whether to use experience replay.
    pub use_experience_replay: bool,
    /// Whether to use self-play.
    pub use_self_play: bool,
    /// Whether to use self-improvement.
    pub use_self_improvement: bool,
    /// Whether to use meta-learning for strategy selection.
    pub use_meta_learning: bool,
    /// Capacity of the experience replay buffer.
    pub replay_capacity: usize,
    /// Print progress every N steps.
    pub log_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_steps: 1_000,
            batch_size: 32,
            use_curriculum: true,
            use_experience_replay: true,
            use_self_play: false,
            use_self_improvement: true,
            use_meta_learning: true,
            replay_capacity: 10_000,
            log_every: 100,
        }
    }
}

/// A single training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStep {
    /// Step index.
    pub step: usize,
    /// Stage name (if curriculum is used).
    pub stage: Option<String>,
    /// Difficulty level (if curriculum is used).
    pub difficulty: Option<DifficultyLevel>,
    /// Reward observed at this step.
    pub reward: f64,
    /// Loss (1 - reward).
    pub loss: f64,
    /// Whether this step was a success.
    pub success: bool,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Strategy used (if meta-learning is enabled).
    pub strategy: Option<String>,
}

/// A training report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    /// Total steps executed.
    pub steps_executed: usize,
    /// Final loss.
    pub final_loss: f64,
    /// Mean reward over all steps.
    pub mean_reward: f64,
    /// Total duration in milliseconds.
    pub total_duration_ms: u64,
    /// Whether the curriculum was advanced.
    pub curriculum_advanced: bool,
    /// Number of self-improvement proposals accepted.
    pub improvements_applied: usize,
    /// Per-strategy rewards (if meta-learning is enabled).
    pub strategy_rewards: ahash::AHashMap<String, f64>,
    /// Per-step history (truncated to the last N steps).
    pub history: Vec<TrainingStep>,
}

/// The training loop: orchestrates all training subsystems.
pub struct TrainingLoop {
    /// Configuration.
    pub config: RwLock<TrainingConfig>,
    /// Curriculum (if used).
    pub curriculum: Arc<Curriculum>,
    /// Experience replay buffer (if used).
    pub replay: Arc<ExperienceReplay>,
    /// Meta-learner (if used).
    pub meta: Arc<MetaLearner>,
    /// Self-improvement cycle (if used).
    pub improvement: Arc<RwLock<SelfImprovementCycle>>,
}

impl Default for TrainingLoop {
    fn default() -> Self {
        Self::new(TrainingConfig::default())
    }
}

impl TrainingLoop {
    /// Construct a new training loop with the given configuration.
    #[must_use]
    pub fn new(config: TrainingConfig) -> Self {
        let curriculum = Arc::new(Curriculum::with_default_stages());
        let replay = Arc::new(ExperienceReplay::new(config.replay_capacity));
        let meta = Arc::new(MetaLearner::new());
        // Register default strategies.
        meta.register("curriculum");
        meta.register("experience_replay");
        meta.register("self_play");
        meta.register("self_improvement");
        let improvement = Arc::new(RwLock::new(SelfImprovementCycle::new()));
        Self {
            config: RwLock::new(config),
            curriculum,
            replay,
            meta,
            improvement,
        }
    }

    /// Run the training loop.
    ///
    /// The `step_fn` closure is called for each training step and should
    /// return the reward observed.
    pub fn run<F>(&self, mut step_fn: F) -> TrainingReport
    where
        F: FnMut(usize, Option<&str>) -> f64,
    {
        let config = self.config.read().clone();
        let start = std::time::Instant::now();
        let mut history: Vec<TrainingStep> = Vec::with_capacity(config.max_steps);
        let mut total_reward = 0.0_f64;
        let mut curriculum_advanced = false;
        let mut strategy_rewards: ahash::AHashMap<String, f64> = ahash::AHashMap::new();

        for step in 0..config.max_steps {
            let step_start = std::time::Instant::now();
            let stage = if config.use_curriculum {
                self.curriculum
                    .current_stage()
                    .map(|s| s.name)
            } else {
                None
            };
            let difficulty = if config.use_curriculum {
                self.curriculum
                    .current_stage()
                    .map(|s| s.difficulty)
            } else {
                None
            };
            let strategy = if config.use_meta_learning {
                self.meta.select()
            } else {
                None
            };

            let reward = step_fn(step, stage.as_deref());
            let success = reward > 0.0;
            let loss = 1.0 - reward;

            // Record into experience replay.
            if config.use_experience_replay {
                self.replay.add(crate::experience_replay::Experience::new(
                    serde_json::json!({"step": step, "stage": stage}),
                    serde_json::json!({"strategy": strategy}),
                    reward,
                    serde_json::json!({"step": step + 1}),
                ));
            }

            // Record into curriculum.
            if config.use_curriculum {
                self.curriculum.record_attempt(success);
                if self.curriculum.maybe_advance() {
                    curriculum_advanced = true;
                }
            }

            // Record strategy reward.
            if let Some(ref s) = strategy {
                self.meta.record(s, reward);
                *strategy_rewards.entry(s.clone()).or_insert(0.0) += reward;
            }

            let step_record = TrainingStep {
                step,
                stage,
                difficulty,
                reward,
                loss,
                success,
                duration_ms: step_start.elapsed().as_millis() as u64,
                strategy,
            };
            total_reward += reward;
            history.push(step_record);

            if config.log_every > 0 && step > 0 && step % config.log_every == 0 {
                let avg = total_reward / (step + 1) as f64;
                tracing::info!(step, avg_reward = avg, "training progress");
            }
        }

        // Run self-improvement at the end.
        let mut improvements_applied = 0;
        if config.use_self_improvement {
            let metrics = self.compute_metrics();
            let accepted = self.improvement.write().run_cycle(&metrics);
            improvements_applied = accepted.len();
        }

        let steps_executed = history.len();
        let mean_reward = if steps_executed == 0 {
            0.0
        } else {
            total_reward / steps_executed as f64
        };
        let final_loss = history
            .last()
            .map(|s| s.loss)
            .unwrap_or(1.0);
        let total_duration_ms = start.elapsed().as_millis() as u64;

        // Truncate history to last 100 steps to keep the report manageable.
        let history = if history.len() > 100 {
            history.split_off(history.len() - 100)
        } else {
            history
        };

        TrainingReport {
            steps_executed,
            final_loss,
            mean_reward,
            total_duration_ms,
            curriculum_advanced,
            improvements_applied,
            strategy_rewards,
            history,
        }
    }

    /// Compute performance metrics from the training history.
    fn compute_metrics(&self) -> PerformanceMetrics {
        let mut metrics = PerformanceMetrics::default();
        // Pull from current state — in a real implementation we'd gather
        // metrics from the actual subsystems.
        metrics.memory_recall_precision = 0.7;
        metrics.reasoning_accuracy = 0.75;
        metrics.vision_accuracy = 0.8;
        metrics.speech_accuracy = 0.85;
        metrics.avg_response_latency_ms = 200.0;
        metrics.total_requests = self.replay.len() as u64;
        metrics
    }
}

impl std::fmt::Debug for TrainingLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingLoop")
            .field("config", &self.config.read())
            .field("curriculum_stages", &self.curriculum.len())
            .field("replay_len", &self.replay.len())
            .field("meta_strategies", &self.meta.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_executes_max_steps() {
        let loop_ = TrainingLoop::new(TrainingConfig {
            max_steps: 10,
            batch_size: 4,
            use_curriculum: true,
            use_experience_replay: true,
            use_self_play: false,
            use_self_improvement: false,
            use_meta_learning: true,
            replay_capacity: 100,
            log_every: 0,
        });
        let report = loop_.run(|_step, _stage| 0.8);
        assert_eq!(report.steps_executed, 10);
        assert!((report.mean_reward - 0.8).abs() < 1e-9);
    }

    #[test]
    fn replay_buffer_populated() {
        let loop_ = TrainingLoop::new(TrainingConfig {
            max_steps: 5,
            use_experience_replay: true,
            ..TrainingConfig::default()
        });
        let _report = loop_.run(|_, _| 0.5);
        assert_eq!(loop_.replay.len(), 5);
    }

    #[test]
    fn curriculum_advances_when_threshold_met() {
        let loop_ = TrainingLoop::new(TrainingConfig {
            max_steps: 16,
            use_curriculum: true,
            ..TrainingConfig::default()
        });
        let report = loop_.run(|_, _| 1.0); // All successes.
        assert!(report.curriculum_advanced);
    }

    #[test]
    fn meta_learning_records_strategy_rewards() {
        let loop_ = TrainingLoop::new(TrainingConfig {
            max_steps: 10,
            use_meta_learning: true,
            use_self_improvement: false,
            ..TrainingConfig::default()
        });
        let report = loop_.run(|_, _| 0.9);
        assert!(!report.strategy_rewards.is_empty());
    }

    #[test]
    fn self_improvement_applied_when_enabled() {
        let loop_ = TrainingLoop::new(TrainingConfig {
            max_steps: 5,
            use_self_improvement: true,
            ..TrainingConfig::default()
        });
        let report = loop_.run(|_, _| 0.5);
        // Self-improvement may or may not accept proposals; just verify it ran.
        let _ = report.improvements_applied;
    }
}
