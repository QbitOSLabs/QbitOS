//! L4 Procedural Memory.
//!
//! Stores reusable skills and workflows. This layer is a thin wrapper around
//! the skill genome (see `qbitos-skills`); it persists skill metadata and
//! execution statistics so the planner can pick the right skill for a task.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::embedding::SharedEmbeddingModel;
use crate::object::{Importance, MemoryId, MemoryObject};

/// Statistics about a skill's execution history.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillStats {
    /// Number of times the skill was invoked.
    pub invocations: u64,
    /// Number of times the skill succeeded.
    pub successes: u64,
    /// Number of times the skill failed.
    pub failures: u64,
    /// Total wall-clock time spent executing the skill (microseconds).
    pub total_time_us: u64,
    /// Last time the skill was invoked.
    pub last_invoked_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl SkillStats {
    /// Success rate in `[0.0, 1.0]`.
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        let total = self.successes + self.failures;
        if total == 0 { 0.0 } else { self.successes as f32 / total as f32 }
    }

    /// Mean execution time in microseconds.
    #[must_use]
    pub fn mean_time_us(&self) -> f64 {
        if self.invocations == 0 { 0.0 } else { self.total_time_us as f64 / self.invocations as f64 }
    }

    /// Record a new execution.
    pub fn record(&mut self, success: bool, time_us: u64) {
        self.invocations += 1;
        if success {
            self.successes += 1;
        } else {
            self.failures += 1;
        }
        self.total_time_us += time_us;
        self.last_invoked_at = Some(chrono::Utc::now());
    }
}

/// A procedural memory entry: a skill with statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralEntry {
    /// Memory envelope.
    pub object: MemoryObject<String>,
    /// Skill statistics.
    pub stats: SkillStats,
}

impl ProceduralEntry {
    /// Confidence in the skill, derived from success rate and invocations.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        let rate = self.stats.success_rate();
        let sample_confidence = 1.0 - (-(self.stats.invocations as f32) / 10.0).exp();
        rate * sample_confidence
    }
}

/// L4 procedural memory.
pub struct ProceduralMemory {
    entries: Arc<RwLock<HashMap<String, ProceduralEntry>>>,
    embedder: SharedEmbeddingModel,
}

impl ProceduralMemory {
    /// Create a new procedural memory.
    #[must_use]
    pub fn new(embedder: SharedEmbeddingModel) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            embedder,
        }
    }

    /// Register a new skill.
    pub fn register(&self, name: impl Into<String>, description: impl Into<String>) -> MemoryId {
        let name = name.into();
        let description = description.into();
        let embedding = self.embedder.embed_text(&description);
        let mut object = MemoryObject::new(crate::object::MemoryType::Procedural, description, embedding);
        object.importance = Importance(0.6);
        let id = object.id;
        let entry = ProceduralEntry { object, stats: SkillStats::default() };
        self.entries.write().insert(name, entry);
        id
    }

    /// Record an execution of a skill.
    pub fn record_execution(&self, name: &str, success: bool, time_us: u64) -> bool {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(name) {
            entry.stats.record(success, time_us);
            // Boost importance after successful executions
            if success {
                entry.object.importance = entry.object.importance.boost(0.05);
            } else {
                entry.object.importance = entry.object.importance.decay(0.1);
            }
            true
        } else {
            false
        }
    }

    /// Look up a skill by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<ProceduralEntry> {
        self.entries.read().get(name).cloned()
    }

    /// List all registered skills, sorted by confidence (descending).
    #[must_use]
    pub fn list(&self) -> Vec<(String, ProceduralEntry)> {
        let mut entries: Vec<_> = self.entries.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort_by(|a, b| b.1.confidence().partial_cmp(&a.1.confidence()).unwrap_or(std::cmp::Ordering::Equal));
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingModelStub;

    #[test]
    fn register_and_invoke() {
        let emb: SharedEmbeddingModel = Arc::new(EmbeddingModelStub::new(32));
        let mem = ProceduralMemory::new(emb);
        mem.register("debug_build_error", "Read error, analyze cause, generate fix, validate.");
        mem.record_execution("debug_build_error", true, 1_500);
        mem.record_execution("debug_build_error", false, 2_000);
        let entry = mem.get("debug_build_error").unwrap();
        assert_eq!(entry.stats.invocations, 2);
        assert!((entry.stats.success_rate() - 0.5).abs() < 1e-6);
    }
}
