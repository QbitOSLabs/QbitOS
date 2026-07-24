//! Improvement engine.
//!
//! Detects weaknesses and proposes improvements. An improvement is a
//! structured change to a skill, a prompt template, or a system parameter.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::evaluator::Evaluator;

/// The kind of improvement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImprovementKind {
    /// Update a skill's step sequence.
    UpdateSkill,
    /// Update a prompt template.
    UpdatePrompt,
    /// Update a system parameter.
    UpdateParameter,
    /// Add a new skill.
    AddSkill,
    /// Remove a skill.
    RemoveSkill,
}

/// An improvement proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    /// Unique id.
    pub id: Uuid,
    /// Kind of improvement.
    pub kind: ImprovementKind,
    /// Target (skill name, parameter name, etc.).
    pub target: String,
    /// The proposed change.
    pub change: serde_json::Value,
    /// Rationale.
    pub rationale: String,
    /// Expected benefit.
    pub expected_benefit: f32,
    /// Risk in `[0.0, 1.0]`.
    pub risk: f32,
    /// When the improvement was proposed.
    pub proposed_at: DateTime<Utc>,
    /// Whether the improvement was accepted.
    pub accepted: Option<bool>,
}

impl Improvement {
    /// Create a new improvement proposal.
    #[must_use]
    pub fn new(kind: ImprovementKind, target: impl Into<String>, change: serde_json::Value, rationale: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            kind,
            target: target.into(),
            change,
            rationale: rationale.into(),
            expected_benefit: 0.5,
            risk: 0.5,
            proposed_at: Utc::now(),
            accepted: None,
        }
    }
}

/// The improvement engine.
pub struct ImprovementEngine {
    evaluator: Arc<Evaluator>,
    proposals: Arc<RwLock<Vec<Improvement>>>,
}

impl ImprovementEngine {
    /// Create a new engine.
    #[must_use]
    pub fn new(evaluator: Arc<Evaluator>) -> Self {
        Self { evaluator, proposals: Arc::new(RwLock::new(Vec::new())) }
    }

    /// Detect weaknesses in recent benchmark results and propose improvements.
    #[must_use]
    pub fn detect_weaknesses(&self) -> Vec<Improvement> {
        let results = self.evaluator.results();
        let mut proposals = Vec::new();
        for result in &results {
            if !result.passed {
                // A failed benchmark suggests the corresponding skill needs work
                let improvement = Improvement::new(
                    ImprovementKind::UpdateSkill,
                    format!("skill_for_benchmark_{}", result.benchmark_id),
                    serde_json::json!({
                        "action": "review_steps",
                        "benchmark_id": result.benchmark_id,
                        "failed_metrics": result.metrics.iter()
                            .filter(|m| m.name == "accuracy" && m.value < 0.5)
                            .map(|m| m.name.clone())
                            .collect::<Vec<_>>(),
                    }),
                    format!("Benchmark {} failed with accuracy < 0.5", result.benchmark_id),
                );
                proposals.push(improvement);
            }
        }
        // Store proposals
        let mut store = self.proposals.write();
        for p in &proposals {
            store.push(p.clone());
        }
        proposals
    }

    /// Accept an improvement proposal.
    pub fn accept(&self, improvement_id: Uuid) -> bool {
        let mut proposals = self.proposals.write();
        for p in proposals.iter_mut() {
            if p.id == improvement_id {
                p.accepted = Some(true);
                return true;
            }
        }
        false
    }

    /// Reject an improvement proposal.
    pub fn reject(&self, improvement_id: Uuid) -> bool {
        let mut proposals = self.proposals.write();
        for p in proposals.iter_mut() {
            if p.id == improvement_id {
                p.accepted = Some(false);
                return true;
            }
        }
        false
    }

    /// List all proposals.
    #[must_use]
    pub fn proposals(&self) -> Vec<Improvement> {
        self.proposals.read().clone()
    }

    /// List pending (undecided) proposals.
    #[must_use]
    pub fn pending(&self) -> Vec<Improvement> {
        self.proposals.read().iter().filter(|p| p.accepted.is_none()).cloned().collect()
    }
}
