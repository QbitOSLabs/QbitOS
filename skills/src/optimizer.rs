//! Skill optimizer — prunes low-performing skills and suggests improvements.

use crate::registry::{SkillId, SkillRegistry};

/// Result of an optimization pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationResult {
    /// Skills that were pruned (removed).
    pub pruned: Vec<SkillId>,
    /// Skills that need improvement (low success rate).
    pub needs_improvement: Vec<SkillId>,
    /// Suggested improvements for each skill.
    pub suggestions: Vec<SkillSuggestion>,
}

/// A suggested improvement for a skill.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SkillSuggestion {
    /// The skill ID.
    pub skill_id: SkillId,
    /// The skill name.
    pub name: String,
    /// Current success rate.
    pub success_rate: f64,
    /// Current mean latency.
    pub mean_latency_ms: f64,
    /// The suggested action.
    pub suggestion: String,
}

/// The skill optimizer.
pub struct SkillOptimizer {
    /// Minimum number of executions before a skill is eligible for pruning.
    pub min_samples: u64,
    /// Success rate below which a skill is pruned.
    pub prune_threshold: f64,
    /// Success rate below which a skill is flagged for improvement.
    pub improve_threshold: f64,
    /// Latency threshold (ms) above which a skill is flagged as slow.
    pub slow_latency_threshold: f64,
}

impl SkillOptimizer {
    /// Create a new optimizer with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_samples: 16,
            prune_threshold: 0.2,
            improve_threshold: 0.5,
            slow_latency_threshold: 5000.0,
        }
    }

    /// Run an optimization pass on the registry.
    pub fn optimize(&self, registry: &SkillRegistry) -> OptimizationResult {
        let skills = registry.list();
        let mut pruned = Vec::new();
        let mut needs_improvement = Vec::new();
        let mut suggestions = Vec::new();

        for skill in &skills {
            if skill.execution_count < self.min_samples {
                continue;
            }

            let success_rate = skill.success_rate();
            let mean_latency = skill.mean_latency_ms();

            if success_rate < self.prune_threshold {
                suggestions.push(SkillSuggestion {
                    skill_id: skill.id.unwrap_or(0),
                    name: skill.name.clone(),
                    success_rate,
                    mean_latency_ms: mean_latency,
                    suggestion: format!(
                        "Prune: success rate {:.1}% is below {:.1}% threshold",
                        success_rate * 100.0,
                        self.prune_threshold * 100.0
                    ),
                });
                if let Some(id) = skill.id {
                    pruned.push(id);
                }
            } else if success_rate < self.improve_threshold {
                needs_improvement.push(skill.id.unwrap_or(0));
                suggestions.push(SkillSuggestion {
                    skill_id: skill.id.unwrap_or(0),
                    name: skill.name.clone(),
                    success_rate,
                    mean_latency_ms: mean_latency,
                    suggestion: format!(
                        "Improve: success rate {:.1}% is below {:.1}% threshold",
                        success_rate * 100.0,
                        self.improve_threshold * 100.0
                    ),
                });
            } else if mean_latency > self.slow_latency_threshold {
                suggestions.push(SkillSuggestion {
                    skill_id: skill.id.unwrap_or(0),
                    name: skill.name.clone(),
                    success_rate,
                    mean_latency_ms: mean_latency,
                    suggestion: format!(
                        "Optimize: mean latency {:.0}ms exceeds {:.0}ms threshold",
                        mean_latency, self.slow_latency_threshold
                    ),
                });
            }
        }

        // Actually prune the flagged skills.
        for id in &pruned {
            registry.remove(*id);
        }

        OptimizationResult {
            pruned,
            needs_improvement,
            suggestions,
        }
    }
}

impl Default for SkillOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimize_prunes_low_success_skills() {
        let reg = SkillRegistry::new();
        let id1 = reg.register(Skill::new("good", "test"));
        let id2 = reg.register(Skill::new("bad", "test"));
        for _ in 0..20 {
            reg.record_execution(id1, true);
        }
        for _ in 0..20 {
            reg.record_execution(id2, false);
        }
        let opt = SkillOptimizer::new();
        let result = opt.optimize(&reg);
        assert!(result.pruned.contains(&id2));
        assert!(!result.pruned.contains(&id1));
        assert!(reg.get(id2).is_none());
    }

    #[test]
    fn optimize_skills_with_few_samples() {
        let reg = SkillRegistry::new();
        let id = reg.register(Skill::new("test", "test"));
        reg.record_execution(id, false);  // Only 1 sample
        let opt = SkillOptimizer::new();
        let result = opt.optimize(&reg);
        assert!(result.pruned.is_empty());
        assert!(reg.get(id).is_some());
    }

    #[test]
    fn optimize_flags_slow_skills() {
        let reg = SkillRegistry::new();
        let id = reg.register(Skill::new("slow", "test"));
        for _ in 0..20 {
            reg.record_execution_with_latency(id, true, 6000);
        }
        let opt = SkillOptimizer::new();
        let result = opt.optimize(&reg);
        assert!(result.suggestions.iter().any(|s| s.skill_id == id && s.suggestion.starts_with("Optimize")));
    }
}
