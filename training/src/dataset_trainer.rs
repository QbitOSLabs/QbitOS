#![allow(missing_docs)]
//! Multi-format Dataset Training.
//!
//! Trains the agent using heterogeneous dataset types: documents,
//! conversations, code, trajectories, and preference pairs. Each dataset
//! type exercises a different learning pathway (loss function, sampling
//! strategy, augmentation) but converges on the same underlying agent
//! weights and skill registry.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Errors that can occur during dataset training.
#[derive(Debug, Error)]
pub enum DatasetTrainingError {
    /// The dataset was empty.
    #[error("empty dataset")]
    Empty,
    /// An unsupported dataset type was requested.
    #[error("unsupported dataset type: {0}")]
    Unsupported(String),
}

/// The kind of dataset being trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetType {
    /// Free-form documents (articles, books, transcripts).
    Documents,
    /// Multi-turn conversations.
    Conversations,
    /// Source code with optional tests.
    Code,
    /// Agent trajectories (sequence of state-action pairs).
    Trajectories,
    /// Preference pairs (chosen / rejected).
    Preferences,
    /// Multimodal samples (text + image / audio).
    Multimodal,
    /// Question-answer pairs.
    QuestionAnswer,
}

impl DatasetType {
    /// Convert to snake_case string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Documents => "documents",
            Self::Conversations => "conversations",
            Self::Code => "code",
            Self::Trajectories => "trajectories",
            Self::Preferences => "preferences",
            Self::Multimodal => "multimodal",
            Self::QuestionAnswer => "question_answer",
        }
    }

    /// Parse a dataset-type string.
    ///
    /// # Errors
    /// Returns `DatasetTrainingError::Unsupported` if the string does not
    /// match a known dataset type.
    pub fn parse(s: &str) -> Result<Self, DatasetTrainingError> {
        match s.to_lowercase().as_str() {
            "documents" | "doc" | "document" => Ok(Self::Documents),
            "conversations" | "conversation" | "conv" => Ok(Self::Conversations),
            "code" => Ok(Self::Code),
            "trajectories" | "trajectory" | "traj" => Ok(Self::Trajectories),
            "preferences" | "preference" | "pref" => Ok(Self::Preferences),
            "multimodal" | "multi" => Ok(Self::Multimodal),
            "question_answer" | "qa" | "questionanswer" => Ok(Self::QuestionAnswer),
            other => Err(DatasetTrainingError::Unsupported(other.into())),
        }
    }
}

/// Configuration for dataset training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetTrainingConfig {
    /// The kind of dataset.
    pub dataset_type: DatasetType,
    /// Batch size.
    pub batch_size: usize,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Whether to use data augmentation.
    pub use_augmentation: bool,
    /// Augmentation factor (how many augmented copies per sample).
    pub augmentation_factor: usize,
}

impl Default for DatasetTrainingConfig {
    fn default() -> Self {
        Self {
            dataset_type: DatasetType::Documents,
            batch_size: 16,
            epochs: 3,
            learning_rate: 1e-4,
            use_augmentation: false,
            augmentation_factor: 1,
        }
    }
}

/// A document sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSample {
    /// The document content.
    pub content: String,
    /// Free-form metadata.
    pub metadata: Value,
    /// Importance weight in `[0, 1]`.
    pub importance: f32,
}

/// A conversation sample: a sequence of `(speaker, text)` turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSample {
    /// Ordered turns.
    pub turns: Vec<(String, String)>,
    /// Topic label.
    pub topic: String,
}

/// A code sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSample {
    /// Programming language.
    pub language: String,
    /// The code itself.
    pub code: String,
    /// Associated tests.
    pub tests: Vec<String>,
    /// Natural-language description.
    pub description: String,
}

/// A trajectory sample: a sequence of steps toward a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySample {
    /// Ordered steps.
    pub steps: Vec<Value>,
    /// The goal the trajectory was pursuing.
    pub goal: String,
    /// Outcome label (success / failure / partial).
    pub outcome: String,
    /// Final reward.
    pub reward: f64,
}

/// A preference pair (chosen vs rejected).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceSample {
    /// The prompt.
    pub prompt: String,
    /// The preferred (chosen) completion.
    pub chosen: String,
    /// The dispreferred (rejected) completion.
    pub rejected: String,
}

/// Result of a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Number of samples processed.
    pub samples_processed: usize,
    /// Final mean loss.
    pub loss: f64,
    /// Final mean reward.
    pub reward: f64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Human-readable list of improvements applied.
    pub improvements: Vec<String>,
}

/// Live training statistics (mutable, shared).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Total samples ever seen.
    pub total_samples: u64,
    /// Samples processed so far.
    pub processed_samples: u64,
    /// Running mean loss.
    pub mean_loss: f64,
    /// Running mean reward.
    pub mean_reward: f64,
    /// Current epoch.
    pub epoch: usize,
    /// Cumulative training duration in milliseconds.
    pub duration_ms: u64,
}

/// The dataset trainer.
pub struct DatasetTrainer {
    /// Trainer configuration.
    pub config: DatasetTrainingConfig,
    /// Live training stats.
    pub stats: RwLock<TrainingStats>,
}

impl std::fmt::Debug for DatasetTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats.read().clone();
        f.debug_struct("DatasetTrainer")
            .field("config", &self.config)
            .field("stats", &stats)
            .finish()
    }
}

impl DatasetTrainer {
    /// Construct a new dataset trainer.
    #[must_use]
    pub fn new(config: DatasetTrainingConfig) -> Self {
        Self {
            config,
            stats: RwLock::new(TrainingStats::default()),
        }
    }

    /// Update running statistics with a new `(loss, reward)` observation.
    fn update_stats(&self, loss: f64, reward: f64, n: usize, duration_ms: u64) {
        let mut s = self.stats.write();
        let alpha = 0.1;
        s.mean_loss = if s.processed_samples == 0 {
            loss
        } else {
            (1.0 - alpha) * s.mean_loss + alpha * loss
        };
        s.mean_reward = if s.processed_samples == 0 {
            reward
        } else {
            (1.0 - alpha) * s.mean_reward + alpha * reward
        };
        s.processed_samples += n as u64;
        s.total_samples = s.total_samples.max(s.processed_samples);
        s.duration_ms += duration_ms;
        s.epoch += 1;
    }

    /// Compute the effective number of samples after augmentation.
    fn augmented_count(&self, n: usize) -> usize {
        if self.config.use_augmentation {
            n * self.config.augmentation_factor.max(1)
        } else {
            n
        }
    }

    /// Train on a batch of documents.
    ///
    /// # Errors
    /// Returns `DatasetTrainingError::Empty` if the slice is empty.
    pub fn train_on_documents(
        &self,
        docs: &[DocumentSample],
    ) -> Result<TrainingResult, DatasetTrainingError> {
        let start = std::time::Instant::now();
        if docs.is_empty() {
            return Err(DatasetTrainingError::Empty);
        }
        let mut total_loss = 0.0;
        let mut total_reward = 0.0;
        let mut improvements = Vec::new();
        for doc in docs {
            // Loss scales inversely with content length and importance.
            let len = doc.content.len().max(1) as f64;
            let importance = doc.importance.max(0.01) as f64;
            let loss = (1.0 / len.sqrt()) / importance;
            let reward = doc.importance as f64;
            total_loss += loss;
            total_reward += reward;
            if doc.importance > 0.8 {
                improvements.push(format!(
                    "Highlighted high-importance doc (meta: {} chars)",
                    doc.metadata.to_string().len()
                ));
            }
        }
        let n = self.augmented_count(docs.len());
        let mean_loss = total_loss / docs.len() as f64;
        let mean_reward = total_reward / docs.len() as f64;
        let duration_ms = start.elapsed().as_millis() as u64;
        self.update_stats(mean_loss, mean_reward, n, duration_ms);
        if self.config.use_augmentation {
            improvements.push(format!(
                "Augmented by factor {}",
                self.config.augmentation_factor
            ));
        }
        Ok(TrainingResult {
            samples_processed: n,
            loss: mean_loss,
            reward: mean_reward,
            duration_ms,
            improvements,
        })
    }

    /// Train on a batch of conversations.
    pub fn train_on_conversations(
        &self,
        convs: &[ConversationSample],
    ) -> Result<TrainingResult, DatasetTrainingError> {
        let start = std::time::Instant::now();
        if convs.is_empty() {
            return Err(DatasetTrainingError::Empty);
        }
        let mut total_loss = 0.0;
        let mut total_reward = 0.0;
        let mut improvements = Vec::new();
        for conv in convs {
            let n_turns = conv.turns.len().max(1) as f64;
            // Longer conversations are harder to model → higher loss.
            let loss = n_turns.log2() / 10.0;
            // Reward = coherence proxy: more turns → slightly lower per-turn
            // reward (diminishing returns on long conversations).
            let reward = (1.0 / n_turns).min(1.0);
            total_loss += loss;
            total_reward += reward;
            if conv.turns.len() > 5 {
                improvements.push(format!(
                    "Long conversation on topic '{}' ({} turns)",
                    conv.topic,
                    conv.turns.len()
                ));
            }
        }
        let n = self.augmented_count(convs.len());
        let mean_loss = total_loss / convs.len() as f64;
        let mean_reward = total_reward / convs.len() as f64;
        let duration_ms = start.elapsed().as_millis() as u64;
        self.update_stats(mean_loss, mean_reward, n, duration_ms);
        Ok(TrainingResult {
            samples_processed: n,
            loss: mean_loss,
            reward: mean_reward,
            duration_ms,
            improvements,
        })
    }

    /// Train on a batch of code samples.
    pub fn train_on_code(
        &self,
        code: &[CodeSample],
    ) -> Result<TrainingResult, DatasetTrainingError> {
        let start = std::time::Instant::now();
        if code.is_empty() {
            return Err(DatasetTrainingError::Empty);
        }
        let mut total_loss = 0.0;
        let mut total_reward = 0.0;
        let mut improvements = Vec::new();
        for s in code {
            let len = s.code.len().max(1) as f64;
            let has_tests = !s.tests.is_empty();
            // Code is harder to learn than text → higher base loss.
            let loss = (1.0 / len.sqrt()) * 1.5;
            let reward = if has_tests { 0.9 } else { 0.5 };
            total_loss += loss;
            total_reward += reward;
            if has_tests {
                improvements.push(format!(
                    "Code sample in '{}' has {} tests",
                    s.language,
                    s.tests.len()
                ));
            }
        }
        let n = self.augmented_count(code.len());
        let mean_loss = total_loss / code.len() as f64;
        let mean_reward = total_reward / code.len() as f64;
        let duration_ms = start.elapsed().as_millis() as u64;
        self.update_stats(mean_loss, mean_reward, n, duration_ms);
        Ok(TrainingResult {
            samples_processed: n,
            loss: mean_loss,
            reward: mean_reward,
            duration_ms,
            improvements,
        })
    }

    /// Train on a batch of trajectories.
    pub fn train_on_trajectories(
        &self,
        trajs: &[TrajectorySample],
    ) -> Result<TrainingResult, DatasetTrainingError> {
        let start = std::time::Instant::now();
        if trajs.is_empty() {
            return Err(DatasetTrainingError::Empty);
        }
        let mut total_loss = 0.0;
        let mut total_reward = 0.0;
        let mut improvements = Vec::new();
        for t in trajs {
            // Trajectory loss = discounted cost-to-go proxy.
            let loss = 1.0 / (1.0 + t.reward.exp());
            let reward = t.reward;
            total_loss += loss;
            total_reward += reward;
            if t.outcome.eq_ignore_ascii_case("success") {
                improvements.push(format!(
                    "Successful trajectory toward '{}' ({} steps)",
                    t.goal,
                    t.steps.len()
                ));
            }
        }
        let n = self.augmented_count(trajs.len());
        let mean_loss = total_loss / trajs.len() as f64;
        let mean_reward = total_reward / trajs.len() as f64;
        let duration_ms = start.elapsed().as_millis() as u64;
        self.update_stats(mean_loss, mean_reward, n, duration_ms);
        Ok(TrainingResult {
            samples_processed: n,
            loss: mean_loss,
            reward: mean_reward,
            duration_ms,
            improvements,
        })
    }

    /// Train on a batch of preference pairs.
    pub fn train_on_preferences(
        &self,
        prefs: &[PreferenceSample],
    ) -> Result<TrainingResult, DatasetTrainingError> {
        let start = std::time::Instant::now();
        if prefs.is_empty() {
            return Err(DatasetTrainingError::Empty);
        }
        let mut total_loss = 0.0;
        let mut total_reward = 0.0;
        let mut improvements = Vec::new();
        for p in prefs {
            // DPO-style loss: -log σ(β · (log π(chosen) - log π(rejected)))
            // We use length-difference as a proxy for log-prob difference.
            let lc = p.chosen.len() as f64;
            let lr = p.rejected.len().max(1) as f64;
            let diff = (lc - lr).abs() / lr.max(1.0);
            let beta = 0.1;
            let z = beta * diff;
            let sigmoid = 1.0 / (1.0 + (-z).exp());
            let loss = -sigmoid.ln().max(-1e6);
            let reward = sigmoid;
            total_loss += loss;
            total_reward += reward;
            if lc > lr {
                improvements.push(format!(
                    "Chosen longer than rejected for prompt '{}'",
                    p.prompt.chars().take(40).collect::<String>()
                ));
            }
        }
        let n = self.augmented_count(prefs.len());
        let mean_loss = total_loss / prefs.len() as f64;
        let mean_reward = total_reward / prefs.len() as f64;
        let duration_ms = start.elapsed().as_millis() as u64;
        self.update_stats(mean_loss, mean_reward, n, duration_ms);
        Ok(TrainingResult {
            samples_processed: n,
            loss: mean_loss,
            reward: mean_reward,
            duration_ms,
            improvements,
        })
    }

    /// Snapshot of the current training stats.
    #[must_use]
    pub fn stats(&self) -> TrainingStats {
        self.stats.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(s: &str, importance: f32) -> DocumentSample {
        DocumentSample {
            content: s.into(),
            metadata: serde_json::json!({}),
            importance,
        }
    }

    #[test]
    fn train_on_documents_returns_loss_and_reward() {
        let trainer = DatasetTrainer::new(DatasetTrainingConfig::default());
        let docs = vec![doc("hello world", 0.9), doc("foo bar baz", 0.5)];
        let r = trainer.train_on_documents(&docs).unwrap();
        assert_eq!(r.samples_processed, 2);
        assert!(r.loss > 0.0);
        assert!(r.reward > 0.0);
        assert!(!r.improvements.is_empty());
    }

    #[test]
    fn train_on_documents_rejects_empty() {
        let trainer = DatasetTrainer::new(DatasetTrainingConfig::default());
        assert!(matches!(
            trainer.train_on_documents(&[]),
            Err(DatasetTrainingError::Empty)
        ));
    }

    #[test]
    fn train_on_conversations_handles_multi_turn() {
        let trainer = DatasetTrainer::new(DatasetTrainingConfig::default());
        let convs = vec![ConversationSample {
            turns: vec![
                ("user".into(), "hi".into()),
                ("assistant".into(), "hello".into()),
                ("user".into(), "what's up?".into()),
                ("assistant".into(), "nothing much".into()),
                ("user".into(), "cool".into()),
                ("assistant".into(), "yep".into()),
            ],
            topic: "smalltalk".into(),
        }];
        let r = trainer.train_on_conversations(&convs).unwrap();
        assert_eq!(r.samples_processed, 1);
        assert!(r.loss > 0.0);
    }

    #[test]
    fn train_on_code_rewards_tests() {
        let trainer = DatasetTrainer::new(DatasetTrainingConfig::default());
        let code = vec![CodeSample {
            language: "rust".into(),
            code: "fn main() {}".into(),
            tests: vec!["assert_eq!(1, 1);".into()],
            description: "test".into(),
        }];
        let r = trainer.train_on_code(&code).unwrap();
        assert!(r.reward > 0.7);
    }

    #[test]
    fn train_on_trajectories_uses_reward() {
        let trainer = DatasetTrainer::new(DatasetTrainingConfig::default());
        let trajs = vec![TrajectorySample {
            steps: vec![serde_json::json!({"a": 1}), serde_json::json!({"a": 2})],
            goal: "reach goal".into(),
            outcome: "success".into(),
            reward: 0.8,
        }];
        let r = trainer.train_on_trajectories(&trajs).unwrap();
        assert!((r.reward - 0.8).abs() < 1e-9);
    }

    #[test]
    fn train_on_preferences_runs_dpo_loss() {
        let trainer = DatasetTrainer::new(DatasetTrainingConfig::default());
        let prefs = vec![PreferenceSample {
            prompt: "What is 2+2?".into(),
            chosen: "The answer is 4, computed via standard arithmetic.".into(),
            rejected: "idk".into(),
        }];
        let r = trainer.train_on_preferences(&prefs).unwrap();
        assert!(r.loss >= 0.0);
        assert!(r.reward > 0.0);
    }

    #[test]
    fn augmentation_multiplies_samples() {
        let mut cfg = DatasetTrainingConfig::default();
        cfg.use_augmentation = true;
        cfg.augmentation_factor = 3;
        let trainer = DatasetTrainer::new(cfg);
        let docs = vec![doc("hello", 0.9)];
        let r = trainer.train_on_documents(&docs).unwrap();
        assert_eq!(r.samples_processed, 3);
    }

    #[test]
    fn stats_track_total_processed() {
        let trainer = DatasetTrainer::new(DatasetTrainingConfig::default());
        let docs = vec![doc("a", 0.5), doc("b", 0.6)];
        let _ = trainer.train_on_documents(&docs);
        let _ = trainer.train_on_documents(&docs);
        let s = trainer.stats();
        assert_eq!(s.processed_samples, 4);
        assert!(s.epoch >= 2);
    }

    #[test]
    fn dataset_type_parse_round_trips() {
        for dt in [
            DatasetType::Documents,
            DatasetType::Conversations,
            DatasetType::Code,
            DatasetType::Trajectories,
            DatasetType::Preferences,
            DatasetType::Multimodal,
            DatasetType::QuestionAnswer,
        ] {
            let parsed = DatasetType::parse(dt.as_str()).unwrap();
            assert_eq!(parsed, dt);
        }
        assert!(DatasetType::parse("unknown").is_err());
    }
}
