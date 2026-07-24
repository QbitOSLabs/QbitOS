//! The composite planner.
//!
//! Combines the goal engine, task graph, reasoning engine, and decision
//! engine into a single pipeline: observation → goal → plan (task graph) →
//! decisions at each step.

use crate::decision_engine::{Decision, DecisionEngine, DecisionStrategy};
use crate::goal_engine::{Goal, GoalEngine};
use crate::reasoning_engine::{FactBase, ReasoningEngine};
use crate::task_graph::{Task, TaskGraph, TaskKind};

/// The planner: turns observations into executable task graphs.
pub struct Planner {
    goal_engine: GoalEngine,
    decision_engine: DecisionEngine,
    reasoning_engine: ReasoningEngine,
}

impl Planner {
    /// Create a new planner with default sub-engines.
    #[must_use]
    pub fn new() -> Self {
        Self {
            goal_engine: GoalEngine::new(),
            decision_engine: DecisionEngine::new(DecisionStrategy::ExpectedUtility),
            reasoning_engine: ReasoningEngine::new(Vec::new()),
        }
    }

    /// Create a new planner with custom sub-engines.
    #[must_use]
    pub fn with_engines(
        goal_engine: GoalEngine,
        decision_engine: DecisionEngine,
        reasoning_engine: ReasoningEngine,
    ) -> Self {
        Self { goal_engine, decision_engine, reasoning_engine }
    }

    /// Derive a goal from an observation.
    #[must_use]
    pub fn derive_goal(&self, observation: &serde_json::Value) -> Option<Goal> {
        self.goal_engine.derive(observation)
    }

    /// Build a task graph to achieve a goal.
    ///
    /// This is a default decomposition: it creates a sequence of tasks
    /// (recall → infer → tool/skill → remember) that achieves most goals.
    /// Override [`Planner::custom_plan`] for goal-specific decomposition.
    #[must_use]
    pub fn plan(&self, goal: &Goal) -> TaskGraph {
        if let Some(custom) = self.custom_plan(goal) {
            return custom;
        }
        let mut graph = TaskGraph::new();
        let recall = graph.add_task(Task::new(
            TaskKind::Recall,
            format!("recall_context_for_{}", goal.kind.as_str_name()),
            serde_json::json!({"goal": goal.objective}),
        ));
        let infer = graph.add_task(Task::new(
            TaskKind::Infer,
            format!("infer_strategy_for_{}", goal.kind.as_str_name()),
            serde_json::json!({"goal": goal.objective}),
        ));
        let act = graph.add_task(Task::new(
            TaskKind::Tool,
            format!("execute_{}", goal.kind.as_str_name()),
            serde_json::json!({"goal": goal.objective}),
        ));
        let remember = graph.add_task(Task::new(
            TaskKind::Remember,
            "remember_outcome".to_string(),
            serde_json::json!({"goal": goal.objective}),
        ));
        graph.add_dependency(recall, infer);
        graph.add_dependency(infer, act);
        graph.add_dependency(act, remember);
        graph
    }

    /// Hook for goal-specific decomposition. Override to provide custom
    /// plans for specific goal kinds.
    #[must_use]
    fn custom_plan(&self, _goal: &Goal) -> Option<TaskGraph> {
        None
    }

    /// Pick the best decision from candidates.
    #[must_use]
    pub fn decide(&self, candidates: &[Decision]) -> Option<Decision> {
        self.decision_engine.decide(candidates)
    }

    /// Run forward chaining on a fact base.
    pub fn reason(&self, base: &mut FactBase) -> Vec<crate::reasoning_engine::ReasoningTrace> {
        self.reasoning_engine.forward_chain(base)
    }
}

impl Default for Planner {
    fn default() -> Self {
        Self::new()
    }
}

trait GoalKindExt {
    fn as_str_name(&self) -> &'static str;
}

impl GoalKindExt for crate::goal_engine::GoalKind {
    fn as_str_name(&self) -> &'static str {
        match self {
            Self::Inform => "inform",
            Self::Act => "act",
            Self::Learn => "learn",
            Self::Plan => "plan",
            Self::Reflect => "reflect",
            Self::Diagnose => "diagnose",
            Self::Create => "create",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal_engine::GoalKind;

    #[test]
    fn default_plan_for_question() {
        let planner = Planner::new();
        let goal = Goal::new(GoalKind::Inform, "answer question", serde_json::json!({"prompt": "hello"}));
        let graph = planner.plan(&goal);
        assert_eq!(graph.len(), 4);
        let order = graph.topological_order();
        assert_eq!(order.len(), 4);
    }
}
