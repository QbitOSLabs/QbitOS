//! Self-Evolving Memory Skills — RL-based skill selection + LLM-guided skill evolution.
//!
//! Based on 2026 research:
//! - MemSkill (arXiv 2602.02474): "Learning and Evolving Memory Skills for
//!   Self-Adaptive Agents" — closed-loop optimization combining RL for skill
//!   selection with LLM-guided skill evolution from hard cases.
//! - MUSE-Autoskill (arXiv 2605.27366): "Self-Evolving Agents via Skill
//!   Creation, Memory, and Evaluation" — agents create skills on demand,
//!   store and reuse them, organize them efficiently, and evaluate them.
//! - Hexmachina (OpenReview 2026): dual memory system (strategic + technical)
//!   for continual learning — instead of starting from scratch, the system
//!   accumulates strategic and technical knowledge across episodes.
//!
//! Key innovations implemented here:
//!
//! 1. **Skill Evolution Loop**: detect hard cases → analyze failure → generate
//!    improved skill variant → test → accept/reject.
//! 2. **RL-based Skill Selection**: UCB1 bandit selects which skill to use
//!    for a given task, balancing exploration and exploitation.
//! 3. **Dual Memory**: strategic memory (high-level plans, heuristics) +
//!    technical memory (specific execution traces, code snippets).
//! 4. **Skill Lineage**: track how skills evolve over time (parent → child).
//! 5. **Difficulty Estimation**: classify tasks as easy/medium/hard to
//!    trigger skill evolution only when needed.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};

/// A skill in the self-evolving system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolvingSkill {
    /// Unique skill ID.
    pub id: String,
    /// Skill name (human-readable).
    pub name: String,
    /// The skill's prompt or procedure.
    pub prompt: String,
    /// Parent skill ID (for lineage tracking). None if original.
    pub parent_id: Option<String>,
    /// Generation number (0 = original, 1 = first evolution, etc.).
    pub generation: u32,
    /// Difficulty level this skill is optimized for.
    pub difficulty: Difficulty,
    /// Skill category (strategic or technical).
    pub category: SkillCategory,
    /// RL statistics: number of times selected.
    pub selections: u64,
    /// RL statistics: total reward accumulated.
    pub total_reward: f64,
    /// RL statistics: mean reward.
    pub mean_reward: f64,
    /// Whether this skill is currently active.
    pub active: bool,
    /// When this skill was created.
    pub created_at: DateTime<Utc>,
    /// Last modification time.
    pub updated_at: DateTime<Utc>,
    /// Evolution metadata.
    pub evolution_metadata: Option<EvolutionMetadata>,
}

/// Difficulty level of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Difficulty {
    /// Easy tasks (success rate > 80%).
    Easy,
    /// Medium tasks (success rate 50-80%).
    Medium,
    /// Hard tasks (success rate < 50%).
    Hard,
}

/// Skill category (dual memory system).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillCategory {
    /// Strategic memory: high-level plans, heuristics, decision rules.
    Strategic,
    /// Technical memory: specific execution traces, code, tool sequences.
    Technical,
}

/// Metadata about how a skill was evolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetadata {
    /// Why the skill was evolved (e.g., "low success rate on hard cases").
    pub reason: String,
    /// The failure pattern that triggered evolution.
    pub failure_pattern: String,
    /// What was changed (e.g., "added verification step", "rephrased prompt").
    pub change_description: String,
    /// Performance before evolution.
    pub before_reward: f64,
    /// Performance after evolution (filled after testing).
    pub after_reward: Option<f64>,
    /// Whether the evolution was accepted.
    pub accepted: bool,
}

/// The self-evolving skill manager.
pub struct SelfEvolvingSkills {
    /// All skills, indexed by ID.
    skills: Arc<RwLock<HashMap<String, EvolvingSkill>>>,
    /// UCB1 exploration constant.
    exploration_constant: f64,
    /// Total selections across all skills (for UCB1).
    total_selections: Arc<std::sync::atomic::AtomicU64>,
    /// Maximum generations (to prevent infinite evolution).
    max_generations: u32,
    /// Minimum samples before evolving a skill.
    min_samples_to_evolve: u64,
    /// Reward threshold below which to trigger evolution.
    evolve_threshold: f64,
}

/// Result of selecting a skill.
#[derive(Debug, Clone)]
pub struct SkillSelection {
    /// The selected skill.
    pub skill: EvolvingSkill,
    /// The UCB1 score that led to selection.
    pub ucb1_score: f64,
}

/// Result of evaluating a skill execution.
#[derive(Debug, Clone)]
pub struct SkillEvaluation {
    /// The skill ID that was evaluated.
    pub skill_id: String,
    /// Reward in [0, 1].
    pub reward: f64,
    /// Whether the task succeeded.
    pub success: bool,
    /// Estimated difficulty of the task.
    pub difficulty: Difficulty,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Error message if the task failed.
    pub error: Option<String>,
}

impl SelfEvolvingSkills {
    /// Create a new self-evolving skill manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            skills: Arc::new(RwLock::new(HashMap::new())),
            exploration_constant: 1.414, // sqrt(2)
            total_selections: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            max_generations: 10,
            min_samples_to_evolve: 5,
            evolve_threshold: 0.5,
        }
    }

    /// Register a new skill.
    pub fn register(&self, skill: EvolvingSkill) {
        self.skills.write().insert(skill.id.clone(), skill);
    }

    /// Select the best skill for a task using UCB1.
    ///
    /// UCB1 formula: `mean_reward + c * sqrt(2 * ln(N) / n_i)`
    /// where N = total selections, n_i = selections of skill i.
    #[must_use]
    pub fn select(&self, difficulty: Difficulty, category: SkillCategory) -> Option<SkillSelection> {
        let skills = self.skills.read();
        let total = self.total_selections.load(std::sync::atomic::Ordering::Relaxed) as f64;
        if total == 0.0 {
            // No history — return the first active skill matching the criteria.
            return skills
                .values()
                .find(|s| s.active && s.difficulty == difficulty && s.category == category)
                .cloned()
                .map(|s| SkillSelection { skill: s, ucb1_score: f64::INFINITY });
        }

        let ln_total = total.ln();
        let mut best: Option<(EvolvingSkill, f64)> = None;
        for skill in skills.values() {
            if !skill.active || skill.difficulty != difficulty || skill.category != category {
                continue;
            }
            let n_i = skill.selections as f64;
            let ucb1 = if n_i == 0.0 {
                f64::INFINITY // Untried skills get infinite priority.
            } else {
                skill.mean_reward + self.exploration_constant * (2.0 * ln_total / n_i).sqrt()
            };
            if best.as_ref().is_none_or(|(_, s)| ucb1 > *s) {
                best = Some((skill.clone(), ucb1));
            }
        }
        best.map(|(skill, score)| SkillSelection { skill, ucb1_score: score })
    }

    /// Record the outcome of a skill execution.
    pub fn record(&self, evaluation: SkillEvaluation) {
        let mut skills = self.skills.write();
        self.total_selections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(skill) = skills.get_mut(&evaluation.skill_id) {
            skill.selections += 1;
            skill.total_reward += evaluation.reward;
            skill.mean_reward = skill.total_reward / skill.selections as f64;
            skill.updated_at = Utc::now();
        }
    }

    /// Check if a skill should be evolved.
    ///
    /// Returns `Some(EvolutionRequest)` if the skill has enough samples and
    /// its performance is below the evolution threshold.
    #[must_use]
    pub fn should_evolve(&self, skill_id: &str) -> Option<EvolutionRequest> {
        let skills = self.skills.read();
        let skill = skills.get(skill_id)?;
        if skill.selections < self.min_samples_to_evolve {
            return None;
        }
        if skill.generation >= self.max_generations {
            return None;
        }
        if skill.mean_reward >= self.evolve_threshold {
            return None;
        }
        Some(EvolutionRequest {
            skill_id: skill.id.clone(),
            current_reward: skill.mean_reward,
            selections: skill.selections,
            generation: skill.generation,
            difficulty: skill.difficulty,
            failure_pattern: format!(
                "Mean reward {:.3} after {} selections (target: {:.3})",
                skill.mean_reward, skill.selections, self.evolve_threshold
            ),
        })
    }

    /// Register an evolved skill (child of an existing skill).
    pub fn register_evolved(
        &self,
        parent_id: &str,
        new_prompt: String,
        change_description: String,
        before_reward: f64,
    ) -> Option<String> {
        let skills = self.skills.read();
        let parent = skills.get(parent_id)?;
        let child_id = format!("{}_gen{}", parent.name, parent.generation + 1);
        let child = EvolvingSkill {
            id: child_id.clone(),
            name: parent.name.clone(),
            prompt: new_prompt,
            parent_id: Some(parent_id.to_string()),
            generation: parent.generation + 1,
            difficulty: parent.difficulty,
            category: parent.category,
            selections: 0,
            total_reward: 0.0,
            mean_reward: 0.0,
            active: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            evolution_metadata: Some(EvolutionMetadata {
                reason: format!("Evolved from {} (reward was {:.3})", parent_id, before_reward),
                failure_pattern: format!("Parent reward: {:.3}", before_reward),
                change_description,
                before_reward,
                after_reward: None,
                accepted: false,
            }),
        };
        drop(skills);
        self.skills.write().insert(child_id.clone(), child);
        Some(child_id)
    }

    /// Accept or reject an evolved skill based on its performance.
    pub fn evaluate_evolved(&self, skill_id: &str, after_reward: f64, accepted: bool) {
        let mut skills = self.skills.write();
        if let Some(skill) = skills.get_mut(skill_id) {
            if let Some(meta) = &mut skill.evolution_metadata {
                meta.after_reward = Some(after_reward);
                meta.accepted = accepted;
            }
            if !accepted {
                skill.active = false; // Deactivate rejected skills.
            }
        }
        // Also deactivate the parent if the child is accepted.
        if accepted {
            // Extract parent_id first to avoid double mutable borrow.
            let parent_id = {
                let skills = self.skills.read();
                skills.get(skill_id).and_then(|s| s.parent_id.clone())
            };
            if let Some(pid) = parent_id {
                if let Some(parent) = skills.get_mut(&pid) {
                    parent.active = false;
                }
            }
        }
    }

    /// Get the skill lineage (parent chain).
    #[must_use]
    pub fn lineage(&self, skill_id: &str) -> Vec<EvolvingSkill> {
        let skills = self.skills.read();
        let mut chain = Vec::new();
        let mut current_id = skill_id.to_string();
        while let Some(skill) = skills.get(&current_id) {
            chain.push(skill.clone());
            match &skill.parent_id {
                Some(pid) => current_id = pid.clone(),
                None => break,
            }
        }
        chain
    }

    /// Get all active skills.
    #[must_use]
    pub fn active_skills(&self) -> Vec<EvolvingSkill> {
        self.skills
            .read()
            .values()
            .filter(|s| s.active)
            .cloned()
            .collect()
    }

    /// Get skills by category.
    #[must_use]
    pub fn by_category(&self, category: SkillCategory) -> Vec<EvolvingSkill> {
        self.skills
            .read()
            .values()
            .filter(|s| s.category == category)
            .cloned()
            .collect()
    }

    /// Get statistics about the skill system.
    #[must_use]
    pub fn stats(&self) -> SelfEvolvingStats {
        let skills = self.skills.read();
        let total = skills.len();
        let active = skills.values().filter(|s| s.active).count();
        let evolved = skills.values().filter(|s| s.generation > 0).count();
        let accepted = skills
            .values()
            .filter(|s| s.evolution_metadata.as_ref().is_some_and(|m| m.accepted))
            .count();
        let total_selections = self.total_selections.load(std::sync::atomic::Ordering::Relaxed);
        let avg_reward = {
            let sum: f64 = skills.values().map(|s| s.mean_reward * s.selections as f64).sum();
            let count: f64 = skills.values().map(|s| s.selections as f64).sum();
            if count > 0.0 { sum / count } else { 0.0 }
        };
        SelfEvolvingStats {
            total_skills: total,
            active_skills: active,
            evolved_skills: evolved,
            accepted_evolutions: accepted,
            total_selections,
            avg_reward,
        }
    }
}

impl Default for SelfEvolvingSkills {
    fn default() -> Self {
        Self::new()
    }
}

/// A request to evolve a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionRequest {
    /// The skill to evolve.
    pub skill_id: String,
    /// Current mean reward.
    pub current_reward: f64,
    /// Number of selections so far.
    pub selections: u64,
    /// Current generation.
    pub generation: u32,
    /// Difficulty level.
    pub difficulty: Difficulty,
    /// Description of the failure pattern.
    pub failure_pattern: String,
}

/// Statistics about the self-evolving skill system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolvingStats {
    /// Total skills (including inactive and evolved).
    pub total_skills: usize,
    /// Active skills.
    pub active_skills: usize,
    /// Evolved skills (generation > 0).
    pub evolved_skills: usize,
    /// Accepted evolutions.
    pub accepted_evolutions: usize,
    /// Total skill selections.
    pub total_selections: u64,
    /// Average reward across all selections.
    pub avg_reward: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skill(id: &str, name: &str, difficulty: Difficulty, category: SkillCategory) -> EvolvingSkill {
        EvolvingSkill {
            id: id.into(),
            name: name.into(),
            prompt: format!("Do {name}"),
            parent_id: None,
            generation: 0,
            difficulty,
            category,
            selections: 0,
            total_reward: 0.0,
            mean_reward: 0.0,
            active: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            evolution_metadata: None,
        }
    }

    #[test]
    fn select_returns_none_when_empty() {
        let mgr = SelfEvolvingSkills::new();
        let result = mgr.select(Difficulty::Easy, SkillCategory::Strategic);
        assert!(result.is_none());
    }

    #[test]
    fn select_returns_first_skill_with_no_history() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "summarize", Difficulty::Easy, SkillCategory::Technical));
        let result = mgr.select(Difficulty::Easy, SkillCategory::Technical);
        assert!(result.is_some());
        assert_eq!(result.unwrap().skill.id, "s1");
    }

    #[test]
    fn ucb1_prefers_untried_skills() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "a", Difficulty::Easy, SkillCategory::Technical));
        mgr.register(make_skill("s2", "b", Difficulty::Easy, SkillCategory::Technical));
        // Record s1 as tried with low reward.
        mgr.record(SkillEvaluation {
            skill_id: "s1".into(),
            reward: 0.1,
            success: false,
            difficulty: Difficulty::Easy,
            duration_ms: 100,
            error: Some("failed".into()),
        });
        // s2 should be preferred (untried → infinite UCB1).
        let result = mgr.select(Difficulty::Easy, SkillCategory::Technical);
        assert_eq!(result.unwrap().skill.id, "s2");
    }

    #[test]
    fn should_evolve_triggers_on_low_reward() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "a", Difficulty::Hard, SkillCategory::Technical));
        for _ in 0..10 {
            mgr.record(SkillEvaluation {
                skill_id: "s1".into(),
                reward: 0.2,
                success: false,
                difficulty: Difficulty::Hard,
                duration_ms: 50,
                error: Some("failed".into()),
            });
        }
        let req = mgr.should_evolve("s1");
        assert!(req.is_some());
        assert_eq!(req.unwrap().current_reward, 0.2);
    }

    #[test]
    fn should_not_evolve_with_high_reward() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "a", Difficulty::Easy, SkillCategory::Technical));
        for _ in 0..10 {
            mgr.record(SkillEvaluation {
                skill_id: "s1".into(),
                reward: 0.9,
                success: true,
                difficulty: Difficulty::Easy,
                duration_ms: 50,
                error: None,
            });
        }
        assert!(mgr.should_evolve("s1").is_none());
    }

    #[test]
    fn evolved_skill_has_lineage() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "a", Difficulty::Hard, SkillCategory::Technical));
        let child_id = mgr.register_evolved("s1", "improved prompt".into(), "rephrased".into(), 0.2);
        assert!(child_id.is_some());
        let lineage = mgr.lineage(&child_id.unwrap());
        assert_eq!(lineage.len(), 2);
        assert_eq!(lineage[0].generation, 1);
        assert_eq!(lineage[1].generation, 0);
    }

    #[test]
    fn accept_evolved_deactivates_parent() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "a", Difficulty::Hard, SkillCategory::Technical));
        let child_id = mgr.register_evolved("s1", "better".into(), "improved".into(), 0.2).unwrap();
        mgr.evaluate_evolved(&child_id, 0.8, true);
        let parent = mgr.active_skills();
        assert!(parent.iter().all(|s| s.id != "s1")); // Parent deactivated.
        assert!(parent.iter().any(|s| s.id == child_id)); // Child still active.
    }

    #[test]
    fn reject_evolved_deactivates_child() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "a", Difficulty::Hard, SkillCategory::Technical));
        let child_id = mgr.register_evolved("s1", "worse".into(), "tried".into(), 0.2).unwrap();
        mgr.evaluate_evolved(&child_id, 0.1, false);
        let active = mgr.active_skills();
        assert!(active.iter().any(|s| s.id == "s1")); // Parent still active.
        assert!(active.iter().all(|s| s.id != child_id)); // Child deactivated.
    }

    #[test]
    fn stats_are_correct() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "a", Difficulty::Easy, SkillCategory::Strategic));
        mgr.register(make_skill("s2", "b", Difficulty::Hard, SkillCategory::Technical));
        mgr.record(SkillEvaluation {
            skill_id: "s1".into(), reward: 0.8, success: true,
            difficulty: Difficulty::Easy, duration_ms: 50, error: None,
        });
        let stats = mgr.stats();
        assert_eq!(stats.total_skills, 2);
        assert_eq!(stats.active_skills, 2);
        assert_eq!(stats.total_selections, 1);
        assert!(stats.avg_reward > 0.0);
    }

    #[test]
    fn dual_memory_categories() {
        let mgr = SelfEvolvingSkills::new();
        mgr.register(make_skill("s1", "plan", Difficulty::Medium, SkillCategory::Strategic));
        mgr.register(make_skill("s2", "code", Difficulty::Medium, SkillCategory::Technical));
        let strategic = mgr.by_category(SkillCategory::Strategic);
        let technical = mgr.by_category(SkillCategory::Technical);
        assert_eq!(strategic.len(), 1);
        assert_eq!(technical.len(), 1);
    }
}
