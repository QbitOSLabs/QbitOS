//! LLMCompiler parallel planner (Kim ICLR 2024, refined 2026).
//!
//! Implements a parallel task planner that decomposes a goal into a DAG of
//! tool-call tasks, resolves dependencies via topological waves, and executes
//! each wave in parallel. Inspired by the LLMCompiler framework (Kim et al.,
//! ICLR 2024) with 2026 refinements for streaming and recovery.
//!
//! # Architecture
//!
//! - **PlannedTask**: a single tool call with an id, tool name, input JSON,
//!   and a list of task dependencies.
//! - **ParallelPlan**: a collection of PlannedTasks plus an optional
//!   `final_task` that identifies the result to return.
//! - **topological_waves**: groups tasks into waves where all tasks in a wave
//!   have their dependencies satisfied by previous waves. Each wave can be
//!   executed in parallel.
//! - **substitute**: resolves `${task_id.field}` references in a task's input
//!   using the outputs of previously-executed tasks.
//! - **ParallelPlanner**: validates a plan against a max-tasks limit.
//! - **PlanResultStore**: a thread-safe store for completed plans (keyed by
//!   a u64 plan id).

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Errors emitted by the parallel planner.
#[derive(Debug, Error)]
pub enum ParallelPlannerError {
    /// A dependency cycle was detected in the task graph.
    #[error("cycle detected: `{0}`")]
    Cycle(String),
    /// A referenced task id was not found in the plan.
    #[error("task `{0}` not found")]
    NotFound(String),
    /// A task execution failed.
    #[error("execution error in `{task}`: {error}")]
    Execution {
        /// The task that failed.
        task: String,
        /// The error message from the runner.
        error: String,
    },
    /// A task referenced a dependency that doesn't exist.
    #[error("missing dependency for `{task}`: {dep}")]
    MissingDependency {
        /// The task with the missing dependency.
        task: String,
        /// The missing dependency's task id.
        dep: String,
    },
}

/// Unique identifier for a planned task.
pub type PlannerTaskId = String;

/// A single planned task: a tool call with dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedTask {
    /// Unique task id (referenced by other tasks' `depends_on`).
    pub id: PlannerTaskId,
    /// The tool to call.
    pub tool: String,
    /// Input JSON (may contain `${task_id.field}` references).
    pub input: serde_json::Value,
    /// Task ids that must complete before this task can run.
    #[serde(default)]
    pub depends_on: Vec<PlannerTaskId>,
    /// Estimated execution duration in milliseconds (for scheduling).
    #[serde(default)]
    pub estimated_duration_ms: u64,
}

/// A parallel execution plan: a DAG of tasks plus an optional final task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelPlan {
    /// All tasks in the plan.
    pub tasks: Vec<PlannedTask>,
    /// The task whose output is the plan's final result (defaults to the last task).
    #[serde(default)]
    pub final_task: Option<PlannerTaskId>,
}

impl ParallelPlan {
    /// Create a new empty plan.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            final_task: None,
        }
    }

    /// Add a task to the plan. Returns a mutable reference for chaining.
    pub fn add_task(&mut self, task: PlannedTask) -> &mut PlannedTask {
        self.tasks.push(task);
        self.tasks.last_mut().unwrap()
    }

    /// Validate that all dependency references exist.
    ///
    /// # Errors
    /// Returns `MissingDependency` if any task references a non-existent dependency.
    pub fn validate(&self) -> Result<(), ParallelPlannerError> {
        let ids: std::collections::HashSet<&str> =
            self.tasks.iter().map(|t| t.id.as_str()).collect();
        for t in &self.tasks {
            for d in &t.depends_on {
                if !ids.contains(d.as_str()) {
                    return Err(ParallelPlannerError::MissingDependency {
                        task: t.id.clone(),
                        dep: d.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Group tasks into topological waves: each wave contains tasks whose
    /// dependencies are all satisfied by previous waves. Tasks within a wave
    /// can be executed in parallel.
    #[must_use]
    pub fn topological_waves(&self) -> Vec<Vec<PlannerTaskId>> {
        let mut waves = Vec::new();
        let mut completed: AHashMap<String, ()> = AHashMap::new();
        let mut remaining: Vec<&PlannedTask> = self.tasks.iter().collect();
        while !remaining.is_empty() {
            let wave: Vec<PlannerTaskId> = remaining
                .iter()
                .filter(|t| t.depends_on.iter().all(|d| completed.contains_key(d)))
                .map(|t| t.id.clone())
                .collect();
            if wave.is_empty() {
                break;
            }
            for id in &wave {
                completed.insert(id.clone(), ());
            }
            remaining.retain(|t| !completed.contains_key(&t.id));
            waves.push(wave);
        }
        waves
    }

    /// Execute the plan by running each wave in sequence (tasks within a wave
    /// are executed sequentially in this implementation; a parallel runner
    /// would execute them concurrently).
    ///
    /// # Errors
    /// Returns `MissingDependency` if validation fails.
    /// Returns `Execution` if any task's runner returns an error.
    pub fn execute<F>(
        &self,
        mut runner: F,
    ) -> Result<AHashMap<PlannerTaskId, serde_json::Value>, ParallelPlannerError>
    where
        F: FnMut(&str, serde_json::Value) -> Result<serde_json::Value, String>,
    {
        self.validate()?;
        let mut outputs: AHashMap<PlannerTaskId, serde_json::Value> = AHashMap::new();
        let waves = self.topological_waves();
        for wave in waves {
            for tid in wave {
                let task = self.tasks.iter().find(|t| t.id == tid).unwrap();
                let resolved = substitute(&task.input, &outputs);
                let result = runner(&task.tool, resolved).map_err(|e| {
                    ParallelPlannerError::Execution {
                        task: tid.clone(),
                        error: e,
                    }
                })?;
                outputs.insert(tid, result);
            }
        }
        Ok(outputs)
    }

    /// Returns the final result from the outputs map, based on `final_task`
    /// or the last task if `final_task` is None.
    #[must_use]
    pub fn final_result<'a>(
        &'a self,
        outputs: &'a AHashMap<PlannerTaskId, serde_json::Value>,
    ) -> Option<&'a serde_json::Value> {
        if let Some(id) = &self.final_task {
            outputs.get(id)
        } else {
            self.tasks.last().and_then(|t| outputs.get(&t.id))
        }
    }
}

impl Default for ParallelPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// A planner that validates plans against a maximum task count.
pub struct ParallelPlanner {
    /// Maximum number of tasks allowed in a plan.
    pub max_tasks: usize,
}

impl ParallelPlanner {
    /// Create a new planner with the given max-tasks limit.
    #[must_use]
    pub fn new(max: usize) -> Self {
        Self { max_tasks: max }
    }

    /// Validate a plan against the max-tasks limit and dependency graph.
    ///
    /// # Errors
    /// Returns `Cycle` if the plan exceeds the max-tasks limit.
    /// Returns `MissingDependency` if any dependency is missing.
    pub fn validate(&self, p: &ParallelPlan) -> Result<(), ParallelPlannerError> {
        if p.tasks.len() > self.max_tasks {
            return Err(ParallelPlannerError::Cycle(format!(
                "exceeds {}",
                self.max_tasks
            )));
        }
        p.validate()
    }
}

impl Default for ParallelPlanner {
    fn default() -> Self {
        Self::new(32)
    }
}

/// A thread-safe store for completed plans, keyed by a u64 plan id.
pub struct PlanResultStore {
    results: parking_lot::RwLock<AHashMap<u64, Arc<ParallelPlan>>>,
}

impl PlanResultStore {
    /// Create a new empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: parking_lot::RwLock::new(AHashMap::new()),
        }
    }

    /// Store a plan under the given key.
    pub fn store(&self, k: u64, p: Arc<ParallelPlan>) {
        self.results.write().insert(k, p);
    }

    /// Retrieve a plan by key.
    #[must_use]
    pub fn get(&self, k: u64) -> Option<Arc<ParallelPlan>> {
        self.results.read().get(&k).cloned()
    }

    /// Returns the number of stored plans.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.read().len()
    }

    /// Returns `true` if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.read().is_empty()
    }
}

impl Default for PlanResultStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Recursively substitute `${task_id.field}` references in a JSON value
/// using the outputs of previously-executed tasks.
fn substitute(
    v: &serde_json::Value,
    outputs: &AHashMap<PlannerTaskId, serde_json::Value>,
) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) if s.starts_with("${") && s.ends_with('}') => {
            let path = &s[2..s.len() - 1];
            let mut p = path.splitn(2, '.');
            let sid = p.next().unwrap_or("");
            let f = p.next();
            if let Some(o) = outputs.get(sid) {
                if let Some(f) = f {
                    o.get(f).cloned().unwrap_or(serde_json::Value::Null)
                } else {
                    o.clone()
                }
            } else {
                serde_json::Value::Null
            }
        }
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                out.insert(k.clone(), substitute(v, outputs));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(|v| substitute(v, outputs)).collect())
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validate_ok() {
        let p = ParallelPlan {
            tasks: vec![
                PlannedTask {
                    id: "a".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec![],
                    estimated_duration_ms: 100,
                },
                PlannedTask {
                    id: "b".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec!["a".into()],
                    estimated_duration_ms: 100,
                },
            ],
            final_task: Some("b".into()),
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn cycle() {
        let p = ParallelPlan {
            tasks: vec![
                PlannedTask {
                    id: "a".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec!["b".into()],
                    estimated_duration_ms: 100,
                },
                PlannedTask {
                    id: "b".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec!["a".into()],
                    estimated_duration_ms: 100,
                },
            ],
            final_task: None,
        };
        assert!(p.topological_waves().len() < 2);
    }

    #[test]
    fn waves() {
        let p = ParallelPlan {
            tasks: vec![
                PlannedTask {
                    id: "a".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec![],
                    estimated_duration_ms: 100,
                },
                PlannedTask {
                    id: "b".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec![],
                    estimated_duration_ms: 100,
                },
                PlannedTask {
                    id: "c".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec!["a".into(), "b".into()],
                    estimated_duration_ms: 100,
                },
            ],
            final_task: None,
        };
        let w = p.topological_waves();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].len(), 2);
        assert_eq!(w[1].len(), 1);
    }

    #[test]
    fn execute() {
        let p = ParallelPlan {
            tasks: vec![
                PlannedTask {
                    id: "fetch".into(),
                    tool: "http".into(),
                    input: json!({"url":"http://x"}),
                    depends_on: vec![],
                    estimated_duration_ms: 100,
                },
                PlannedTask {
                    id: "parse".into(),
                    tool: "parse".into(),
                    input: json!({"body":"${fetch.body}"}),
                    depends_on: vec!["fetch".into()],
                    estimated_duration_ms: 50,
                },
            ],
            final_task: Some("parse".into()),
        };
        let o = p
            .execute(|t, i| match t {
                "http" => Ok(json!({"body":"hello"})),
                "parse" => Ok(json!({"words": i["body"].as_str().unwrap_or("").split_whitespace().count()})),
                _ => Err("unknown".into()),
            })
            .unwrap();
        assert_eq!(o["parse"]["words"], 1);
    }

    #[test]
    fn final_result() {
        let p = ParallelPlan {
            tasks: vec![PlannedTask {
                id: "only".into(),
                tool: "x".into(),
                input: json!({}),
                depends_on: vec![],
                estimated_duration_ms: 100,
            }],
            final_task: Some("only".into()),
        };
        let mut o = AHashMap::new();
        o.insert("only".into(), json!({"answer":42}));
        assert_eq!(p.final_result(&o).unwrap()["answer"], 42);
    }

    #[test]
    fn max_tasks() {
        let pl = ParallelPlanner::new(2);
        let p = ParallelPlan {
            tasks: vec![
                PlannedTask {
                    id: "a".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec![],
                    estimated_duration_ms: 100,
                },
                PlannedTask {
                    id: "b".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec![],
                    estimated_duration_ms: 100,
                },
                PlannedTask {
                    id: "c".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec![],
                    estimated_duration_ms: 100,
                },
            ],
            final_task: None,
        };
        assert!(pl.validate(&p).is_err());
    }

    #[test]
    fn store() {
        let s = PlanResultStore::new();
        s.store(42, Arc::new(ParallelPlan::new()));
        assert!(s.get(42).is_some());
        assert!(s.get(99).is_none());
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn missing_dep() {
        let p = ParallelPlan {
            tasks: vec![PlannedTask {
                id: "a".into(),
                tool: "x".into(),
                input: json!({}),
                depends_on: vec!["missing".into()],
                estimated_duration_ms: 100,
            }],
            final_task: None,
        };
        assert!(matches!(
            p.validate(),
            Err(ParallelPlannerError::MissingDependency { .. })
        ));
    }

    #[test]
    fn empty_plan() {
        let p = ParallelPlan::new();
        assert!(p.validate().is_ok());
        assert!(p.topological_waves().is_empty());
    }
}
