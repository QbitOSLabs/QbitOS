//! QbitOS Planner
//!
//! Turns goals into executable plans. Composed of four sub-engines:
//!
//! - [`goal_engine`]: parses observations into typed goals
//! - [`task_graph`]: builds a DAG of tasks with dependencies
//! - [`reasoning_engine`]: applies inference rules to refine plans
//! - [`decision_engine`]: picks the optimal action at each step

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod decision_engine;
pub mod parallel_planner;
pub mod goal_engine;
pub mod reasoning_engine;
pub mod task_graph;

pub use decision_engine::{Decision, DecisionEngine, DecisionStrategy};
pub use parallel_planner::{ParallelPlan, ParallelPlanner, ParallelPlannerError, PlannedTask, PlannerTaskId, PlanResultStore};
pub use goal_engine::{Goal, GoalEngine, GoalKind, GoalStatus};
pub use reasoning_engine::{InferenceRule, ReasoningEngine, ReasoningTrace};
pub use task_graph::{Task, TaskGraph, TaskKind, TaskStatus};

/// Re-export the composite planner.
pub use planner::Planner;

mod planner;
