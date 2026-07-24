//! Goal engine.
//!
//! Parses an observation (a JSON value produced by the understand stage)
//! into a typed [`Goal`]. A goal carries a priority, success criteria, and
//! an estimated cost.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// The kind of goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalKind {
    /// Answer a question or provide information.
    Inform,
    /// Perform an action in the world (write a file, run a command).
    Act,
    /// Learn a new skill or fact.
    Learn,
    /// Plan ahead for a future event.
    Plan,
    /// Reflect on a past cycle.
    Reflect,
    /// Diagnose and fix a problem.
    Diagnose,
    /// Create something new (code, document, design).
    Create,
}

/// Goal status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalStatus {
    /// Just created, not yet planned.
    Pending,
    /// Plan generated, ready to execute.
    Planned,
    /// Currently executing.
    Executing,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed,
    /// Cancelled.
    Cancelled,
}

/// A goal: what the kernel is trying to achieve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Unique id.
    pub id: Uuid,
    /// Goal kind.
    pub kind: GoalKind,
    /// Human-readable description.
    pub description: String,
    /// Structured objective (e.g. `{"answer": "string"}` or `{"file": "path"}`).
    pub objective: serde_json::Value,
    /// Success criteria: a list of predicates that must all be true.
    pub success_criteria: Vec<String>,
    /// Priority in `[0.0, 1.0]`.
    pub priority: f32,
    /// Estimated cost in arbitrary units (time × resources).
    pub estimated_cost: f32,
    /// Status.
    pub status: GoalStatus,
    /// When the goal was created.
    pub created_at: DateTime<Utc>,
    /// Deadline, if any.
    pub deadline: Option<DateTime<Utc>>,
    /// Parent goal (for hierarchical decomposition).
    pub parent: Option<Uuid>,
}

impl Goal {
    /// Create a new goal.
    #[must_use]
    pub fn new(kind: GoalKind, description: impl Into<String>, objective: serde_json::Value) -> Self {
        Self {
            id: Uuid::now_v7(),
            kind,
            description: description.into(),
            objective,
            success_criteria: Vec::new(),
            priority: 0.5,
            estimated_cost: 1.0,
            status: GoalStatus::Pending,
            created_at: Utc::now(),
            deadline: None,
            parent: None,
        }
    }

    /// Attach a deadline.
    #[must_use]
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: f32) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }
}

/// The goal engine: derives a [`Goal`] from an observation.
pub struct GoalEngine {
    rules: Vec<GoalRule>,
}

/// A rule that maps an observation pattern to a goal kind.
#[derive(Clone)]
pub struct GoalRule {
    /// Human-readable name.
    pub name: String,
    /// Predicate: returns `true` if this rule applies to the observation.
    pub matches: Arc<dyn Fn(&serde_json::Value) -> bool + Send + Sync>,
    /// Goal kind to emit.
    pub kind: GoalKind,
    /// Default priority.
    pub priority: f32,
}

impl std::fmt::Debug for GoalRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoalRule")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("priority", &self.priority)
            .finish()
    }
}

impl GoalEngine {
    /// Create a new goal engine with default rules.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: vec![
                GoalRule {
                    name: "question".into(),
                    matches: Arc::new(|obs: &serde_json::Value| {
                        obs.get("type").and_then(|v| v.as_str()) == Some("question")
                            || obs.get("prompt").is_some()
                    }),
                    kind: GoalKind::Inform,
                    priority: 0.7,
                },
                GoalRule {
                    name: "command".into(),
                    matches: Arc::new(|obs: &serde_json::Value| {
                        obs.get("type").and_then(|v| v.as_str()) == Some("command")
                            || obs.get("action").is_some()
                    }),
                    kind: GoalKind::Act,
                    priority: 0.8,
                },
                GoalRule {
                    name: "error".into(),
                    matches: Arc::new(|obs: &serde_json::Value| {
                        obs.get("type").and_then(|v| v.as_str()) == Some("error")
                            || obs.get("error").is_some()
                    }),
                    kind: GoalKind::Diagnose,
                    priority: 0.9,
                },
                GoalRule {
                    name: "create".into(),
                    matches: Arc::new(|obs: &serde_json::Value| {
                        obs.get("type").and_then(|v| v.as_str()) == Some("create")
                            || obs.get("create").is_some()
                    }),
                    kind: GoalKind::Create,
                    priority: 0.6,
                },
                GoalRule {
                    name: "learn".into(),
                    matches: Arc::new(|obs: &serde_json::Value| {
                        obs.get("type").and_then(|v| v.as_str()) == Some("learn")
                            || obs.get("learn").is_some()
                    }),
                    kind: GoalKind::Learn,
                    priority: 0.4,
                },
            ],
        }
    }

    /// Add a custom rule.
    pub fn add_rule(&mut self, rule: GoalRule) {
        self.rules.push(rule);
    }

    /// Derive a goal from an observation. Returns `None` if no rule matches.
    #[must_use]
    pub fn derive(&self, observation: &serde_json::Value) -> Option<Goal> {
        for rule in &self.rules {
            if (rule.matches)(observation) {
                let description = observation
                    .get("prompt")
                    .or_else(|| observation.get("description"))
                    .or_else(|| observation.get("action"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unnamed goal")
                    .to_string();
                let mut goal = Goal::new(rule.kind, description, observation.clone());
                goal.priority = rule.priority;
                return Some(goal);
            }
        }
        None
    }
}

impl Default for GoalEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_question_goal() {
        let engine = GoalEngine::new();
        let obs = serde_json::json!({"type": "question", "prompt": "What is 2+2?"});
        let goal = engine.derive(&obs).unwrap();
        assert_eq!(goal.kind, GoalKind::Inform);
        assert_eq!(goal.description, "What is 2+2?");
    }

    #[test]
    fn derive_diagnostic_goal() {
        let engine = GoalEngine::new();
        let obs = serde_json::json!({"type": "error", "error": "build failed"});
        let goal = engine.derive(&obs).unwrap();
        assert_eq!(goal.kind, GoalKind::Diagnose);
    }

    #[test]
    fn no_match_returns_none() {
        let engine = GoalEngine::new();
        let obs = serde_json::json!({"foo": "bar"});
        assert!(engine.derive(&obs).is_none());
    }
}
