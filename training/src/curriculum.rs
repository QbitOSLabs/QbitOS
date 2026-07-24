//! Curriculum learning: progressive difficulty ordering of training tasks.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Difficulty level of a curriculum stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifficultyLevel {
    /// Trivial tasks (warm-up).
    Trivial,
    /// Easy tasks.
    Easy,
    /// Medium tasks.
    Medium,
    /// Hard tasks.
    Hard,
    /// Expert tasks (frontier).
    Expert,
}

impl DifficultyLevel {
    /// Numeric rank (0=trivial, 4=expert).
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            Self::Trivial => 0,
            Self::Easy => 1,
            Self::Medium => 2,
            Self::Hard => 3,
            Self::Expert => 4,
        }
    }

    /// Promotion threshold: success rate required to advance from this level.
    #[must_use]
    pub fn promotion_threshold(self) -> f64 {
        match self {
            Self::Trivial => 0.90,
            Self::Easy => 0.85,
            Self::Medium => 0.75,
            Self::Hard => 0.65,
            Self::Expert => 0.50,
        }
    }

    /// Minimum number of samples before promotion is considered.
    #[must_use]
    pub fn min_samples_for_promotion(self) -> u32 {
        match self {
            Self::Trivial => 8,
            Self::Easy => 16,
            Self::Medium => 32,
            Self::Hard => 64,
            Self::Expert => 128,
        }
    }
}

impl std::fmt::Display for DifficultyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trivial => write!(f, "trivial"),
            Self::Easy => write!(f, "easy"),
            Self::Medium => write!(f, "medium"),
            Self::Hard => write!(f, "hard"),
            Self::Expert => write!(f, "expert"),
        }
    }
}

/// A stage in the curriculum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumStage {
    /// Stage id.
    pub id: String,
    /// Stage name.
    pub name: String,
    /// Difficulty level.
    pub difficulty: DifficultyLevel,
    /// Tasks included in this stage (free-form JSON).
    pub tasks: Vec<serde_json::Value>,
    /// Success rate on this stage so far.
    #[serde(default)]
    pub success_rate: f64,
    /// Number of attempts on this stage.
    #[serde(default)]
    pub attempts: u32,
    /// Number of successes on this stage.
    #[serde(default)]
    pub successes: u32,
}

impl CurriculumStage {
    /// Construct a new stage.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, difficulty: DifficultyLevel) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            difficulty,
            tasks: Vec::new(),
            success_rate: 0.0,
            attempts: 0,
            successes: 0,
        }
    }

    /// Add a task to the stage.
    pub fn add_task(&mut self, task: serde_json::Value) -> &mut Self {
        self.tasks.push(task);
        self
    }

    /// Record an attempt.
    pub fn record_attempt(&mut self, success: bool) {
        self.attempts += 1;
        if success {
            self.successes += 1;
        }
        self.success_rate = self.successes as f64 / self.attempts as f64;
    }

    /// Whether the learner is ready to be promoted from this stage.
    #[must_use]
    pub fn ready_for_promotion(&self) -> bool {
        self.attempts >= u32::from(self.difficulty.min_samples_for_promotion())
            && self.success_rate >= self.difficulty.promotion_threshold()
    }
}

/// The curriculum: an ordered sequence of stages.
#[derive(Default)]
pub struct Curriculum {
    stages: Arc<RwLock<Vec<CurriculumStage>>>,
    current_index: Arc<std::sync::atomic::AtomicUsize>,
    stage_index: Arc<RwLock<AHashMap<String, usize>>>,
}

impl Curriculum {
    /// Construct a new empty curriculum.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a curriculum with default stages (trivial → expert).
    #[must_use]
    pub fn with_default_stages() -> Self {
        let c = Self::new();
        c.add_stage(CurriculumStage::new(
            "stage_1",
            "Warm-up",
            DifficultyLevel::Trivial,
        ));
        c.add_stage(CurriculumStage::new(
            "stage_2",
            "Basic skills",
            DifficultyLevel::Easy,
        ));
        c.add_stage(CurriculumStage::new(
            "stage_3",
            "Intermediate",
            DifficultyLevel::Medium,
        ));
        c.add_stage(CurriculumStage::new(
            "stage_4",
            "Advanced",
            DifficultyLevel::Hard,
        ));
        c.add_stage(CurriculumStage::new(
            "stage_5",
            "Frontier",
            DifficultyLevel::Expert,
        ));
        c
    }

    /// Add a stage to the curriculum.
    pub fn add_stage(&self, stage: CurriculumStage) {
        let idx = self.stages.read().len();
        self.stage_index.write().insert(stage.id.clone(), idx);
        self.stages.write().push(stage);
    }

    /// Get the current stage.
    #[must_use]
    pub fn current_stage(&self) -> Option<CurriculumStage> {
        let idx = self.current_index.load(std::sync::atomic::Ordering::Relaxed);
        self.stages.read().get(idx).cloned()
    }

    /// Get the current stage index.
    #[must_use]
    pub fn current_index(&self) -> usize {
        self.current_index.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Advance to the next stage if the current one is ready for promotion.
    ///
    /// Returns `true` if the curriculum was advanced.
    pub fn maybe_advance(&self) -> bool {
        let stages = self.stages.read();
        let idx = self.current_index.load(std::sync::atomic::Ordering::Relaxed);
        if idx >= stages.len() {
            return false;
        }
        let current = &stages[idx];
        if !current.ready_for_promotion() {
            return false;
        }
        if idx + 1 >= stages.len() {
            return false; // Already at the last stage.
        }
        drop(stages);
        self.current_index
            .store(idx + 1, std::sync::atomic::Ordering::Relaxed);
        tracing::info!(from = idx, to = idx + 1, "curriculum advanced");
        true
    }

    /// Record an attempt on the current stage.
    pub fn record_attempt(&self, success: bool) {
        let idx = self.current_index.load(std::sync::atomic::Ordering::Relaxed);
        let mut stages = self.stages.write();
        if let Some(stage) = stages.get_mut(idx) {
            stage.record_attempt(success);
        }
    }

    /// Number of stages in the curriculum.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stages.read().len()
    }

    /// Whether the curriculum is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.read().is_empty()
    }

    /// List all stages (cloned).
    #[must_use]
    pub fn stages(&self) -> Vec<CurriculumStage> {
        self.stages.read().clone()
    }

    /// Get a stage by id.
    #[must_use]
    pub fn get_stage(&self, id: &str) -> Option<CurriculumStage> {
        let idx = *self.stage_index.read().get(id)?;
        self.stages.read().get(idx).cloned()
    }
}

impl std::fmt::Debug for Curriculum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Curriculum")
            .field("len", &self.len())
            .field("current_index", &self.current_index())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_stages_are_ordered() {
        let c = Curriculum::with_default_stages();
        assert_eq!(c.len(), 5);
        let stages = c.stages();
        assert_eq!(stages[0].difficulty, DifficultyLevel::Trivial);
        assert_eq!(stages[4].difficulty, DifficultyLevel::Expert);
    }

    #[test]
    fn promotion_requires_min_samples() {
        let mut stage = CurriculumStage::new("s", "s", DifficultyLevel::Trivial);
        for _ in 0..7 {
            stage.record_attempt(true);
        }
        // 7/7 success but below min_samples_for_promotion (8).
        assert!(!stage.ready_for_promotion());
        stage.record_attempt(true);
        assert!(stage.ready_for_promotion());
    }

    #[test]
    fn maybe_advance_progresses_when_ready() {
        let c = Curriculum::with_default_stages();
        assert_eq!(c.current_index(), 0);
        for _ in 0..8 {
            c.record_attempt(true);
        }
        assert!(c.maybe_advance());
        assert_eq!(c.current_index(), 1);
    }

    #[test]
    fn maybe_advance_blocked_when_not_ready() {
        let c = Curriculum::with_default_stages();
        for _ in 0..8 {
            c.record_attempt(false);
        }
        assert!(!c.maybe_advance());
        assert_eq!(c.current_index(), 0);
    }

    #[test]
    fn difficulty_levels_order() {
        assert!(DifficultyLevel::Trivial < DifficultyLevel::Easy);
        assert!(DifficultyLevel::Easy < DifficultyLevel::Medium);
        assert!(DifficultyLevel::Medium < DifficultyLevel::Hard);
        assert!(DifficultyLevel::Hard < DifficultyLevel::Expert);
    }
}
