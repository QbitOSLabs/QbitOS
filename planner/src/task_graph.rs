//! Task graph.
//!
//! A directed acyclic graph (DAG) of tasks. Each task has a kind, inputs,
//! outputs, and dependencies. The scheduler walks the DAG in topological
//! order, executing tasks in parallel when their dependencies are met.

use chrono::{DateTime, Utc};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// The kind of task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// Call the neural engine for inference.
    Infer,
    /// Retrieve a memory.
    Recall,
    /// Store a memory.
    Remember,
    /// Invoke a tool.
    Tool,
    /// Execute a skill.
    Skill,
    /// Run a vision model.
    Vision,
    /// Run a speech model.
    Speech,
    /// A pure-Rust computation.
    Compute,
    /// Wait for an external event.
    Wait,
}

/// Task status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Not yet ready (dependencies not met).
    Blocked,
    /// Ready to run.
    Ready,
    /// Currently running.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed,
    /// Skipped (e.g. parent failed).
    Skipped,
}

/// A task in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique id.
    pub id: Uuid,
    /// Task kind.
    pub kind: TaskKind,
    /// Human-readable name.
    pub name: String,
    /// Structured input.
    pub input: serde_json::Value,
    /// Structured output (filled in on completion).
    pub output: Option<serde_json::Value>,
    /// Status.
    pub status: TaskStatus,
    /// Estimated cost in arbitrary units.
    pub estimated_cost: f32,
    /// Actual cost (wall-clock seconds).
    pub actual_cost: Option<f32>,
    /// When the task was created.
    pub created_at: DateTime<Utc>,
    /// When the task started running.
    pub started_at: Option<DateTime<Utc>>,
    /// When the task finished.
    pub finished_at: Option<DateTime<Utc>>,
    /// Error message, if failed.
    pub error: Option<String>,
}

impl Task {
    /// Create a new task.
    #[must_use]
    pub fn new(kind: TaskKind, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            id: Uuid::now_v7(),
            kind,
            name: name.into(),
            input,
            output: None,
            status: TaskStatus::Blocked,
            estimated_cost: 1.0,
            actual_cost: None,
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
            error: None,
        }
    }
}

/// The task graph.
pub struct TaskGraph {
    graph: DiGraph<Task, ()>,
    task_index: HashMap<Uuid, NodeIndex>,
}

impl TaskGraph {
    /// Create a new empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            task_index: HashMap::new(),
        }
    }

    /// Add a task to the graph. Initially it has no dependencies, so its
    /// status is `Ready`.
    pub fn add_task(&mut self, task: Task) -> Uuid {
        let id = task.id;
        let mut task = task;
        task.status = TaskStatus::Ready;
        let node = self.graph.add_node(task);
        self.task_index.insert(id, node);
        id
    }

    /// Add a dependency: `from` must complete before `to` starts.
    ///
    /// Returns `false` if either task is missing or the edge would create a cycle.
    pub fn add_dependency(&mut self, from: Uuid, to: Uuid) -> bool {
        let Some(&from_node) = self.task_index.get(&from) else { return false };
        let Some(&to_node) = self.task_index.get(&to) else { return false };
        // Check for cycle: if there's already a path from `to` to `from`, adding
        // `from → to` would create a cycle.
        if self.has_path(to_node, from_node) {
            return false;
        }
        self.graph.add_edge(from_node, to_node, ());
        // The dependent task is now blocked
        if let Some(t) = self.graph.node_weight_mut(to_node) {
            if t.status == TaskStatus::Ready {
                t.status = TaskStatus::Blocked;
            }
        }
        true
    }

    fn has_path(&self, from: NodeIndex, to: NodeIndex) -> bool {
        if from == to {
            return true;
        }
        let mut stack = vec![from];
        let mut visited = std::collections::HashSet::new();
        while let Some(node) = stack.pop() {
            if !visited.insert(node) {
                continue;
            }
            for neighbor in self.graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                if neighbor == to {
                    return true;
                }
                stack.push(neighbor);
            }
        }
        false
    }

    /// Get the tasks that are currently ready to run (all dependencies completed).
    #[must_use]
    pub fn ready_tasks(&self) -> Vec<Uuid> {
        self.graph.node_indices()
            .filter_map(|node| {
                let task = self.graph.node_weight(node)?;
                if task.status != TaskStatus::Ready {
                    return None;
                }
                // Check that all predecessors are completed
                let all_completed = self.graph.neighbors_directed(node, petgraph::Direction::Incoming)
                    .all(|pred| {
                        self.graph.node_weight(pred).map_or(false, |t| t.status == TaskStatus::Completed)
                    });
                if all_completed { Some(task.id) } else { None }
            })
            .collect()
    }

    /// Mark a task as running.
    pub fn start(&mut self, id: Uuid) -> bool {
        let Some(node) = self.task_index.get(&id).copied() else { return false };
        if let Some(t) = self.graph.node_weight_mut(node) {
            t.status = TaskStatus::Running;
            t.started_at = Some(Utc::now());
            true
        } else {
            false
        }
    }

    /// Mark a task as completed.
    pub fn complete(&mut self, id: Uuid, output: serde_json::Value) -> bool {
        let Some(node) = self.task_index.get(&id).copied() else { return false };
        // First mark the task as completed
        let successors: Vec<NodeIndex> = self.graph.neighbors_directed(node, petgraph::Direction::Outgoing).collect();
        if let Some(t) = self.graph.node_weight_mut(node) {
            t.status = TaskStatus::Completed;
            t.output = Some(output);
            t.finished_at = Some(Utc::now());
            if let Some(start) = t.started_at {
                t.actual_cost = Some((Utc::now() - start).num_seconds() as f32);
            }
        } else {
            return false;
        }
        // Then transition any blocked successors whose predecessors are all completed
        for succ in successors {
            let all_preds_done = self.graph.neighbors_directed(succ, petgraph::Direction::Incoming)
                .all(|pred| {
                    self.graph.node_weight(pred).map_or(false, |t| t.status == TaskStatus::Completed)
                });
            if all_preds_done {
                if let Some(t) = self.graph.node_weight_mut(succ) {
                    if t.status == TaskStatus::Blocked {
                        t.status = TaskStatus::Ready;
                    }
                }
            }
        }
        true
    }

    /// Mark a task as failed and cascade-skip all transitive dependents.
    pub fn fail(&mut self, id: Uuid, error: impl Into<String>) -> bool {
        let Some(node) = self.task_index.get(&id).copied() else { return false };
        let error = error.into();
        if let Some(t) = self.graph.node_weight_mut(node) {
            t.status = TaskStatus::Failed;
            t.error = Some(error);
            t.finished_at = Some(Utc::now());
        }
        // Cascade skip — collect neighbors first to avoid simultaneous borrows
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            let neighbors: Vec<_> = self.graph.neighbors_directed(n, petgraph::Direction::Outgoing).collect();
            for neighbor in neighbors {
                let should_skip = self.graph.node_weight(neighbor).map_or(false, |t| {
                    t.status == TaskStatus::Blocked || t.status == TaskStatus::Ready
                });
                if should_skip {
                    if let Some(t) = self.graph.node_weight_mut(neighbor) {
                        t.status = TaskStatus::Skipped;
                        t.error = Some("dependency failed".into());
                    }
                    stack.push(neighbor);
                }
            }
        }
        true
    }

    /// Get a task by id.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<&Task> {
        self.task_index.get(&id).and_then(|n| self.graph.node_weight(*n))
    }

    /// Number of tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns `true` if the graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Returns `true` if all tasks are in a terminal state (completed, failed, or skipped).
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.graph.node_weights().all(|t| {
            t.status == TaskStatus::Completed || t.status == TaskStatus::Failed || t.status == TaskStatus::Skipped
        })
    }

    /// Returns a topological order of the tasks.
    #[must_use]
    pub fn topological_order(&self) -> Vec<Uuid> {
        use petgraph::visit::Topo;
        let mut topo = Topo::new(&self.graph);
        let mut order = Vec::new();
        while let Some(node) = topo.next(&self.graph) {
            if let Some(t) = self.graph.node_weight(node) {
                order.push(t.id);
            }
        }
        order
    }
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependency_cycle_rejected() {
        let mut g = TaskGraph::new();
        let a = g.add_task(Task::new(TaskKind::Compute, "a", serde_json::json!({})));
        let b = g.add_task(Task::new(TaskKind::Compute, "b", serde_json::json!({})));
        assert!(g.add_dependency(a, b));
        assert!(!g.add_dependency(b, a)); // would create cycle
    }

    #[test]
    fn ready_tasks_respect_dependencies() {
        let mut g = TaskGraph::new();
        let a = g.add_task(Task::new(TaskKind::Compute, "a", serde_json::json!({})));
        let b = g.add_task(Task::new(TaskKind::Compute, "b", serde_json::json!({})));
        let c = g.add_task(Task::new(TaskKind::Compute, "c", serde_json::json!({})));
        g.add_dependency(a, b);
        g.add_dependency(b, c);
        // Initially only `a` is ready
        let ready = g.ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], a);
        // Complete `a`, then `b` becomes ready
        g.start(a);
        g.complete(a, serde_json::json!({}));
        let ready = g.ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], b);
    }

    #[test]
    fn fail_cascades_to_skipped() {
        let mut g = TaskGraph::new();
        let a = g.add_task(Task::new(TaskKind::Compute, "a", serde_json::json!({})));
        let b = g.add_task(Task::new(TaskKind::Compute, "b", serde_json::json!({})));
        g.add_dependency(a, b);
        g.start(a);
        g.fail(a, "boom");
        assert_eq!(g.get(b).unwrap().status, TaskStatus::Skipped);
        assert!(g.is_done());
    }
}
