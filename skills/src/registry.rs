//! Skill registry — stores skills and tracks execution statistics.

use ahash::AHashMap;
use chrono::Utc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A skill ID (sequential, starting from 1).
pub type SkillId = u64;

/// The kind of step in a skill's procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// Read input or recall from memory.
    Read,
    /// Run inference (LLM call).
    Infer,
    /// Call a tool.
    Tool,
    /// Verify the result.
    Verify,
    /// Return the final output.
    Return,
}

/// A single step in a skill's procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStep {
    /// Human-readable name for this step.
    pub name: String,
    /// The kind of step.
    pub kind: StepKind,
    /// The step's configuration as JSON (e.g. tool name, prompt template).
    pub config: serde_json::Value,
}

impl SkillStep {
    /// Create a new skill step.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: StepKind, config: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            kind,
            config,
        }
    }
}

/// A skill: a named, parameterized procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique ID assigned by the registry (None until registered).
    pub id: Option<SkillId>,
    /// Human-readable name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Ordered list of steps.
    pub steps: Vec<SkillStep>,
    /// Number of times this skill has been executed.
    pub execution_count: u64,
    /// Number of successful executions.
    pub success_count: u64,
    /// Total latency in milliseconds across all executions.
    pub total_latency_ms: u64,
    /// Last execution timestamp (Unix epoch ms).
    pub last_executed_at: Option<i64>,
}

impl Skill {
    /// Create a new skill with the given name and description.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: None,
            name: name.into(),
            description: description.into(),
            steps: Vec::new(),
            execution_count: 0,
            success_count: 0,
            total_latency_ms: 0,
            last_executed_at: None,
        }
    }

    /// Add a step to this skill.
    pub fn add_step(&mut self, step: SkillStep) -> &mut SkillStep {
        self.steps.push(step);
        self.steps.last_mut().expect("just pushed")
    }

    /// Compute the success rate (0.0 to 1.0).
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.execution_count == 0 {
            return 0.0;
        }
        self.success_count as f64 / self.execution_count as f64
    }

    /// Compute the mean latency in milliseconds.
    #[must_use]
    pub fn mean_latency_ms(&self) -> f64 {
        if self.execution_count == 0 {
            return 0.0;
        }
        self.total_latency_ms as f64 / self.execution_count as f64
    }

    /// Compute the confidence score using a Beta(2,2) prior with sample-size
    /// scaling. Returns a value in [0, 1].
    #[must_use]
    pub fn confidence(&self) -> f32 {
        // Beta(2,2) prior: alpha=2, beta=2. Posterior mean = (successes + 1) / (total + 2).
        // But we use a sample-size scaling factor so skills with few executions
        // don't get artificially high confidence.
        let alpha = 2.0_f64;
        let beta = 2.0_f64;
        let successes = self.success_count as f64 + alpha - 1.0;
        let total = self.execution_count as f64 + alpha + beta - 2.0;
        if total <= 0.0 {
            return 0.5;
        }
        let posterior_mean = successes / total;
        // Scale by a sigmoid of sample size so confidence grows with evidence.
        let sample_scale = 1.0 - (-0.1 * self.execution_count as f64).exp();
        (posterior_mean * sample_scale) as f32
    }
}

/// The skill registry — stores skills and tracks execution stats.
#[derive(Default)]
pub struct SkillRegistry {
    skills: RwLock<AHashMap<SkillId, Skill>>,
    by_name: RwLock<AHashMap<String, SkillId>>,
    next_id: AtomicU64,
}

impl SkillRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new skill. Returns the assigned ID.
    pub fn register(&self, mut skill: Skill) -> SkillId {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        skill.id = Some(id);
        let name = skill.name.clone();
        self.skills.write().insert(id, skill);
        self.by_name.write().insert(name, id);
        id
    }

    /// Get a skill by ID.
    #[must_use]
    pub fn get(&self, id: SkillId) -> Option<Skill> {
        self.skills.read().get(&id).cloned()
    }

    /// Get a skill by name.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<Skill> {
        let by_name = self.by_name.read();
        let id = *by_name.get(name)?;
        self.skills.read().get(&id).cloned()
    }

    /// List all skills.
    #[must_use]
    pub fn list(&self) -> Vec<Skill> {
        self.skills.read().values().cloned().collect()
    }

    /// Number of registered skills.
    #[must_use]
    pub fn len(&self) -> usize {
        self.skills.read().len()
    }

    /// Returns true if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skills.read().is_empty()
    }

    /// Update a skill in place.
    pub fn update(&self, skill: Skill) {
        if let Some(id) = skill.id {
            self.skills.write().insert(id, skill);
        }
    }

    /// Remove a skill by ID.
    pub fn remove(&self, id: SkillId) -> Option<Skill> {
        let mut skills = self.skills.write();
        let skill = skills.remove(&id)?;
        let mut by_name = self.by_name.write();
        by_name.remove(&skill.name);
        Some(skill)
    }

    /// Record a skill execution outcome.
    pub fn record_execution(&self, id: SkillId, success: bool) {
        self.record_execution_with_latency(id, success, 0);
    }

    /// Record a skill execution outcome with latency.
    pub fn record_execution_with_latency(&self, id: SkillId, success: bool, latency_ms: u64) {
        let mut skills = self.skills.write();
        if let Some(skill) = skills.get_mut(&id) {
            skill.execution_count += 1;
            if success {
                skill.success_count += 1;
            }
            skill.total_latency_ms += latency_ms;
            skill.last_executed_at = Some(Utc::now().timestamp_millis());
        }
    }

    /// Get the top N skills by confidence.
    #[must_use]
    pub fn top_by_confidence(&self, n: usize) -> Vec<Skill> {
        let mut skills: Vec<Skill> = self.skills.read().values().cloned().collect();
        skills.sort_by(|a, b| {
            b.confidence()
                .partial_cmp(&a.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        skills.into_iter().take(n).collect()
    }
}

/// Shared handle to a skill registry.
pub type SharedSkillRegistry = Arc<SkillRegistry>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_assigns_sequential_ids() {
        let reg = SkillRegistry::new();
        let id1 = reg.register(Skill::new("skill1", "first"));
        let id2 = reg.register(Skill::new("skill2", "second"));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn get_by_name_works() {
        let reg = SkillRegistry::new();
        let id = reg.register(Skill::new("my-skill", "test"));
        let skill = reg.get_by_name("my-skill").expect("skill not found");
        assert_eq!(skill.id, Some(id));
        assert_eq!(skill.name, "my-skill");
    }

    #[test]
    fn record_execution_updates_stats() {
        let reg = SkillRegistry::new();
        let id = reg.register(Skill::new("test", "test"));
        reg.record_execution_with_latency(id, true, 100);
        reg.record_execution_with_latency(id, false, 200);
        reg.record_execution_with_latency(id, true, 150);
        let skill = reg.get(id).unwrap();
        assert_eq!(skill.execution_count, 3);
        assert_eq!(skill.success_count, 2);
        assert_eq!(skill.total_latency_ms, 450);
        assert!((skill.success_rate() - 2.0 / 3.0).abs() < 1e-6);
        assert!((skill.mean_latency_ms() - 150.0).abs() < 1e-6);
    }

    #[test]
    fn confidence_increases_with_successful_executions() {
        let reg = SkillRegistry::new();
        let id = reg.register(Skill::new("test", "test"));
        let initial_conf = reg.get(id).unwrap().confidence();
        for _ in 0..10 {
            reg.record_execution(id, true);
        }
        let final_conf = reg.get(id).unwrap().confidence();
        assert!(final_conf > initial_conf);
        assert!(final_conf > 0.5);
    }

    #[test]
    fn top_by_confidence_returns_sorted() {
        let reg = SkillRegistry::new();
        let id1 = reg.register(Skill::new("low", "test"));
        let id2 = reg.register(Skill::new("high", "test"));
        for _ in 0..5 {
            reg.record_execution(id2, true);
        }
        for _ in 0..5 {
            reg.record_execution(id1, false);
        }
        let top = reg.top_by_confidence(2);
        assert_eq!(top[0].name, "high");
        assert_eq!(top[1].name, "low");
    }

    #[test]
    fn remove_skill_works() {
        let reg = SkillRegistry::new();
        let id = reg.register(Skill::new("test", "test"));
        assert!(reg.remove(id).is_some());
        assert!(reg.get(id).is_none());
        assert!(reg.get_by_name("test").is_none());
    }

    #[test]
    fn add_step_appends() {
        let mut skill = Skill::new("test", "test");
        skill.add_step(SkillStep::new("read", StepKind::Read, serde_json::json!({})));
        skill.add_step(SkillStep::new("infer", StepKind::Infer, serde_json::json!({})));
        assert_eq!(skill.steps.len(), 2);
        assert_eq!(skill.steps[0].kind, StepKind::Read);
        assert_eq!(skill.steps[1].kind, StepKind::Infer);
    }
}
