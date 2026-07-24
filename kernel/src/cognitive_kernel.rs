//! The cognitive kernel.
//!
//! Implements the QbitOS cognitive loop:
//!
//! ```text
//! Observe → Understand → Create Goal → Plan → Execute → Verify → Learn → Improve
//! ```
//!
//! Each stage is a method on [`CognitiveKernel`] that produces a
//! [`CognitiveState`] snapshot and emits an [`Event`] on the global bus. The
//! kernel is driven by [`KernelRuntime`], which ticks the loop at a configured
//! rate and dispatches work to the [`Scheduler`].

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::event_system::{Event, EventBus, EventKind};
use crate::resource_manager::ResourceManager;
use crate::scheduler::{Scheduler, TaskId};

/// A single stage of the cognitive loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveStage {
    /// Sensory input has arrived.
    Observe,
    /// The input has been parsed into a semantic representation.
    Understand,
    /// A goal has been derived from the observation.
    CreateGoal,
    /// A plan has been generated to achieve the goal.
    Plan,
    /// The plan is being executed.
    Execute,
    /// The outcome has been verified against the goal.
    Verify,
    /// The outcome has been written to memory and a skill updated.
    Learn,
    /// A reflection cycle has completed and improvements proposed.
    Improve,
}

impl CognitiveStage {
    /// The next stage in the cognitive loop.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::Observe => Self::Understand,
            Self::Understand => Self::CreateGoal,
            Self::CreateGoal => Self::Plan,
            Self::Plan => Self::Execute,
            Self::Execute => Self::Verify,
            Self::Verify => Self::Learn,
            Self::Learn => Self::Improve,
            Self::Improve => Self::Observe,
        }
    }
}

/// A snapshot of the kernel's state at a given tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    /// Monotonically increasing tick counter.
    pub tick: u64,
    /// The current stage.
    pub stage: CognitiveStage,
    /// The id of the agent currently running, if any.
    pub agent_id: Uuid,
    /// The id of the current goal, if any.
    pub goal_id: Option<Uuid>,
    /// The id of the current plan, if any.
    pub plan_id: Option<Uuid>,
    /// The id of the task being executed, if any.
    pub task_id: Option<TaskId>,
    /// When this snapshot was taken.
    pub timestamp: DateTime<Utc>,
    /// Number of cognitive cycles completed since boot.
    pub cycles_completed: u64,
}

impl Default for CognitiveState {
    fn default() -> Self {
        Self {
            tick: 0,
            stage: CognitiveStage::Observe,
            agent_id: Uuid::now_v7(),
            goal_id: None,
            plan_id: None,
            task_id: None,
            timestamp: Utc::now(),
            cycles_completed: 0,
        }
    }
}

/// A cognitive cycle: a single pass through all 8 stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveCycle {
    /// Cycle id.
    pub id: Uuid,
    /// Started at.
    pub started_at: DateTime<Utc>,
    /// Finished at.
    pub finished_at: Option<DateTime<Utc>>,
    /// Initial observation.
    pub observation: serde_json::Value,
    /// Generated goal.
    pub goal: Option<serde_json::Value>,
    /// Generated plan.
    pub plan: Option<serde_json::Value>,
    /// Execution result.
    pub result: Option<serde_json::Value>,
    /// Verification outcome.
    pub verified: Option<bool>,
    /// Learning summary.
    pub learning: Option<serde_json::Value>,
    /// Improvement summary.
    pub improvement: Option<serde_json::Value>,
}

impl CognitiveCycle {
    /// Create a new cycle with the given observation.
    #[must_use]
    pub fn new(observation: serde_json::Value) -> Self {
        Self {
            id: Uuid::now_v7(),
            started_at: Utc::now(),
            finished_at: None,
            observation,
            goal: None,
            plan: None,
            result: None,
            verified: None,
            learning: None,
            improvement: None,
        }
    }
}

/// Trait that every cognitive stage handler must implement.
///
/// This decouples the kernel from any specific reasoning implementation, so
/// the kernel can drive a real neural-engine-backed planner or a fallback for
/// tests.
pub trait CognitiveStageHandler: Send + Sync {
    /// Observe the world: pull a sensory input from the perception bus.
    fn observe(&self, state: &mut CognitiveState) -> serde_json::Value;

    /// Understand the observation: parse it into a semantic representation.
    fn understand(&self, observation: &serde_json::Value, state: &mut CognitiveState) -> serde_json::Value;

    /// Create a goal from the understanding.
    fn create_goal(&self, understanding: &serde_json::Value, state: &mut CognitiveState) -> Option<serde_json::Value>;

    /// Plan how to achieve the goal.
    fn plan(&self, goal: &serde_json::Value, state: &mut CognitiveState) -> Option<serde_json::Value>;

    /// Execute the plan.
    fn execute(&self, plan: &serde_json::Value, state: &mut CognitiveState) -> serde_json::Value;

    /// Verify that the execution achieved the goal.
    fn verify(&self, goal: &serde_json::Value, result: &serde_json::Value, state: &mut CognitiveState) -> bool;

    /// Learn from the cycle: update memory, refine skills.
    fn learn(&self, cycle: &CognitiveCycle, state: &mut CognitiveState) -> serde_json::Value;

    /// Improve: propose changes to skills, strategies, or the kernel itself.
    fn improve(&self, cycle: &CognitiveCycle, state: &mut CognitiveState) -> serde_json::Value;
}

/// The cognitive kernel.
pub struct CognitiveKernel {
    bus: EventBus,
    scheduler: Scheduler,
    resources: ResourceManager,
    handler: Arc<dyn CognitiveStageHandler>,
    state: parking_lot::Mutex<CognitiveState>,
}

impl CognitiveKernel {
    /// Create a new cognitive kernel.
    #[must_use]
    pub fn new(
        bus: EventBus,
        scheduler: Scheduler,
        resources: ResourceManager,
        handler: Arc<dyn CognitiveStageHandler>,
    ) -> Self {
        Self {
            bus,
            scheduler,
            resources,
            handler,
            state: parking_lot::Mutex::new(CognitiveState::default()),
        }
    }

    /// Run one full cognitive cycle.
    ///
    /// Returns the completed cycle. Emits an [`EventKind::CognitiveStage`]
    /// event for each of the 8 stages, plus an [`EventKind::Memory`] event
    /// when learning is committed.
    pub async fn run_cycle(&self, observation: serde_json::Value) -> CognitiveCycle {
        let mut cycle = CognitiveCycle::new(observation);

        // Observe
        self.advance(CognitiveStage::Observe, &cycle.observation);
        let _observation = self.handler.observe(&mut self.state.lock());

        // Understand
        self.advance(CognitiveStage::Understand, &cycle.observation);
        let understanding = self.handler.understand(&cycle.observation, &mut self.state.lock());

        // Create Goal
        self.advance(CognitiveStage::CreateGoal, &understanding);
        let goal = self.handler.create_goal(&understanding, &mut self.state.lock());
        cycle.goal = goal.clone();

        // Plan
        let plan = if let Some(g) = &goal {
            self.advance(CognitiveStage::Plan, g);
            self.handler.plan(g, &mut self.state.lock())
        } else {
            None
        };
        cycle.plan = plan.clone();

        // Execute
        let result = if let Some(p) = &plan {
            self.advance(CognitiveStage::Execute, p);
            self.handler.execute(p, &mut self.state.lock())
        } else {
            serde_json::json!({"skipped": "no plan"})
        };
        cycle.result = Some(result.clone());

        // Verify
        let verified = if let Some(g) = &goal {
            self.advance(CognitiveStage::Verify, g);
            self.handler.verify(g, &result, &mut self.state.lock())
        } else {
            false
        };
        cycle.verified = Some(verified);

        // Learn
        self.advance(CognitiveStage::Learn, &serde_json::json!({"verified": verified}));
        let learning = self.handler.learn(&cycle, &mut self.state.lock());
        cycle.learning = Some(learning.clone());
        self.bus.publish(Event::new(
            EventKind::Memory,
            "memory.episodic.write",
            serde_json::json!({"cycle_id": cycle.id, "learning": learning}),
            "cognitive_kernel",
        )).ok();

        // Improve
        self.advance(CognitiveStage::Improve, &learning);
        let improvement = self.handler.improve(&cycle, &mut self.state.lock());
        cycle.improvement = Some(improvement.clone());
        self.bus.publish(Event::new(
            EventKind::Reflection,
            "reflection.cycle.completed",
            serde_json::json!({"cycle_id": cycle.id, "improvement": improvement}),
            "cognitive_kernel",
        )).ok();

        cycle.finished_at = Some(Utc::now());
        let mut state = self.state.lock();
        state.cycles_completed += 1;
        state.stage = CognitiveStage::Observe;
        cycle
    }

    fn advance(&self, stage: CognitiveStage, payload: &serde_json::Value) {
        let mut state = self.state.lock();
        state.tick += 1;
        state.stage = stage;
        state.timestamp = Utc::now();
        let snapshot = CognitiveState {
            tick: state.tick,
            stage,
            agent_id: state.agent_id,
            goal_id: state.goal_id,
            plan_id: state.plan_id,
            task_id: state.task_id,
            timestamp: state.timestamp,
            cycles_completed: state.cycles_completed,
        };
        drop(state);
        self.bus.publish(Event::new(
            EventKind::CognitiveStage,
            format!("cognitive.stage.{:?}", stage).to_lowercase(),
            payload.clone(),
            "cognitive_kernel",
        ).with_parent(snapshot_agent_event_id(&snapshot))).ok();
    }

    /// Get a snapshot of the current kernel state.
    #[must_use]
    pub fn state(&self) -> CognitiveState {
        self.state.lock().clone()
    }

    /// Get a reference to the scheduler.
    #[must_use]
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Get a reference to the resource manager.
    #[must_use]
    pub fn resources(&self) -> &ResourceManager {
        &self.resources
    }

    /// Get a reference to the event bus.
    #[must_use]
    pub fn bus(&self) -> &EventBus {
        &self.bus
    }
}

fn snapshot_agent_event_id(_state: &CognitiveState) -> crate::event_system::EventId {
    crate::event_system::EventId::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoHandler;

    impl CognitiveStageHandler for EchoHandler {
        fn observe(&self, _state: &mut CognitiveState) -> serde_json::Value {
            serde_json::json!({"status": "ok"})
        }
        fn understand(&self, observation: &serde_json::Value, _state: &mut CognitiveState) -> serde_json::Value {
            observation.clone()
        }
        fn create_goal(&self, understanding: &serde_json::Value, _state: &mut CognitiveState) -> Option<serde_json::Value> {
            Some(understanding.clone())
        }
        fn plan(&self, goal: &serde_json::Value, _state: &mut CognitiveState) -> Option<serde_json::Value> {
            Some(goal.clone())
        }
        fn execute(&self, plan: &serde_json::Value, _state: &mut CognitiveState) -> serde_json::Value {
            plan.clone()
        }
        fn verify(&self, _goal: &serde_json::Value, _result: &serde_json::Value, _state: &mut CognitiveState) -> bool {
            true
        }
        fn learn(&self, _cycle: &CognitiveCycle, _state: &mut CognitiveState) -> serde_json::Value {
            serde_json::json!({"learned": true})
        }
        fn improve(&self, _cycle: &CognitiveCycle, _state: &mut CognitiveState) -> serde_json::Value {
            serde_json::json!({"improved": false})
        }
    }

    #[tokio::test]
    async fn runs_full_cycle() {
        let bus = EventBus::new(64);
        let scheduler = Scheduler::new(4);
        let resources = ResourceManager::new();
        let kernel = CognitiveKernel::new(bus.clone(), scheduler, resources, Arc::new(EchoHandler));

        let cycle = kernel.run_cycle(serde_json::json!({"input": "hello"})).await;
        assert!(cycle.verified.unwrap_or(false));
        assert!(cycle.finished_at.is_some());
        assert_eq!(kernel.state().cycles_completed, 1);
        assert!(bus.count(EventKind::CognitiveStage) >= 8);
    }
}
