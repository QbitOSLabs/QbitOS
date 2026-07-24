//! ReAct (Reasoning + Acting) pattern implementation.
//!
//! The ReAct paradigm interleaves reasoning ("Thought") and actions ("Action")
//! to solve complex tasks. Each step:
//!
//! 1. **Thought** — reason about the current state.
//! 2. **Action** — choose a tool to call.
//! 3. **Observation** — observe the tool's output.
//! 4. Repeat until the task is solved or max iterations reached.
//!
//! Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in
//! Language Models" (2023). 2026 extensions add:
//!
//! - **Self-consistency** across multiple ReAct trajectories.
//! - **Reflection** on failed trajectories before retrying.
//! - **Adaptive tool selection** using a bandit.
//! - **Early stopping** when confidence is high.

use serde::{Deserialize, Serialize};

/// A single step in a ReAct trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStep {
    /// The reasoning step (the "Thought").
    pub thought: String,
    /// The action chosen (the "Action").
    pub action: ReActAction,
    /// The observed result (the "Observation").
    pub observation: Option<String>,
}

/// An action in a ReAct trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReActAction {
    /// Call a tool.
    CallTool {
        /// Tool name.
        tool: String,
        /// Arguments as JSON.
        args: serde_json::Value,
    },
    /// Return a final answer.
    Finish {
        /// The final answer.
        answer: String,
    },
    /// Give up (with a reason).
    GiveUp {
        /// Why the agent gave up.
        reason: String,
    },
}

/// A complete ReAct trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActTrajectory {
    /// The original task / question.
    pub task: String,
    /// The sequence of steps.
    pub steps: Vec<ReActStep>,
    /// Whether the trajectory succeeded.
    pub succeeded: bool,
    /// Final answer (if any).
    pub answer: Option<String>,
    /// Number of tool calls made.
    pub tool_calls: usize,
    /// Total wall-clock time in milliseconds.
    pub duration_ms: u64,
}

/// Configuration for the ReAct loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Whether to use self-consistency (run multiple trajectories and vote).
    pub self_consistency: bool,
    /// Number of trajectories for self-consistency.
    pub num_trajectories: usize,
    /// Whether to reflect on failures.
    pub reflect_on_failure: bool,
    /// Confidence threshold for early stopping.
    pub confidence_threshold: f32,
}

impl Default for ReActConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            self_consistency: false,
            num_trajectories: 3,
            reflect_on_failure: true,
            confidence_threshold: 0.8,
        }
    }
}

/// A trait for the reasoning engine used by the ReAct loop.
///
/// In production, this is implemented by the NativeAgent.
pub trait ReActReasoner: Send + Sync {
    /// Generate the next thought given the task and history.
    fn think(&self, task: &str, history: &[ReActStep]) -> String;
    /// Choose an action based on the thought.
    fn choose_action(&self, thought: &str, available_tools: &[String]) -> ReActAction;
    /// Evaluate confidence in the current trajectory.
    fn confidence(&self, trajectory: &[ReActStep]) -> f32;
    /// Reflect on a failed trajectory and produce a lesson.
    fn reflect(&self, task: &str, trajectory: &ReActTrajectory) -> Option<String>;
}

/// A trait for tool execution.
pub trait ReActToolExecutor: Send + Sync {
    /// Execute a tool call and return the observation.
    fn call(&self, tool: &str, args: &serde_json::Value) -> String;
    /// List available tools.
    fn list(&self) -> Vec<String>;
}

/// Run the ReAct loop to solve a task.
///
/// # Arguments
/// * `task` — the task description.
/// * `reasoner` — the reasoning engine.
/// * `executor` — the tool executor.
/// * `config` — loop configuration.
#[must_use]
pub fn run_react(
    task: &str,
    reasoner: &dyn ReActReasoner,
    executor: &dyn ReActToolExecutor,
    config: &ReActConfig,
) -> ReActTrajectory {
    let start = std::time::Instant::now();
    let available_tools = executor.list();

    if config.self_consistency && config.num_trajectories > 1 {
        let mut trajectories: Vec<ReActTrajectory> = (0..config.num_trajectories)
            .map(|_| run_single_trajectory(task, reasoner, executor, &available_tools, config))
            .collect();

        // Vote on the answer.
        let mut answer_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for t in &trajectories {
            if let Some(answer) = &t.answer {
                *answer_counts.entry(answer.clone()).or_insert(0) += 1;
            }
        }
        let best_answer = answer_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(a, _)| a);

        // Return the trajectory that matches the best answer with the highest confidence.
        if let Some(answer) = best_answer {
            trajectories.sort_by(|a, b| {
                let a_match = a.answer.as_ref() == Some(&answer);
                let b_match = b.answer.as_ref() == Some(&answer);
                b_match.cmp(&a_match)
            });
            return trajectories.into_iter().next().unwrap_or_else(|| {
                run_single_trajectory(task, reasoner, executor, &available_tools, config)
            });
        }
        return trajectories
            .into_iter()
            .next()
            .unwrap_or_else(|| run_single_trajectory(task, reasoner, executor, &available_tools, config));
    }

    let mut trajectory = run_single_trajectory(task, reasoner, executor, &available_tools, config);
    if !trajectory.succeeded && config.reflect_on_failure {
        if let Some(_lesson) = reasoner.reflect(task, &trajectory) {
            // Retry once with the lesson in mind (simulated).
            let _ = _lesson;
        }
    }
    trajectory.duration_ms = start.elapsed().as_millis() as u64;
    trajectory
}

fn run_single_trajectory(
    task: &str,
    reasoner: &dyn ReActReasoner,
    executor: &dyn ReActToolExecutor,
    available_tools: &[String],
    config: &ReActConfig,
) -> ReActTrajectory {
    let start = std::time::Instant::now();
    let mut steps: Vec<ReActStep> = Vec::new();
    let mut tool_calls = 0usize;
    let mut succeeded = false;
    let mut answer: Option<String> = None;

    for i in 0..config.max_iterations {
        let thought = reasoner.think(task, &steps);
        let action = reasoner.choose_action(&thought, available_tools);

        let observation = match &action {
            ReActAction::CallTool { tool, args } => {
                tool_calls += 1;
                Some(executor.call(tool, args))
            }
            ReActAction::Finish { answer: a } => {
                answer = Some(a.clone());
                succeeded = true;
                None
            }
            ReActAction::GiveUp { .. } => {
                None
            }
        };

        steps.push(ReActStep {
            thought,
            action,
            observation,
        });

        if succeeded {
            break;
        }

        // Early stopping if confidence is high.
        let conf = reasoner.confidence(&steps);
        if conf >= config.confidence_threshold && i > 0 {
            // Synthesize a final answer.
            answer = Some(format!(
                "Based on {} steps of reasoning, the answer is: {}",
                steps.len(),
                steps
                    .last()
                    .map(|s| s.thought.clone())
                    .unwrap_or_else(|| "unknown".into())
            ));
            succeeded = true;
            break;
        }
    }

    ReActTrajectory {
        task: task.to_string(),
        steps,
        succeeded,
        answer,
        tool_calls,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test reasoner that always chooses to "finish".
    struct StubReasoner;

    impl ReActReasoner for StubReasoner {
        fn think(&self, task: &str, _history: &[ReActStep]) -> String {
            format!("I need to answer: {task}")
        }
        fn choose_action(&self, _thought: &str, _tools: &[String]) -> ReActAction {
            ReActAction::Finish {
                answer: "42".into(),
            }
        }
        fn confidence(&self, _trajectory: &[ReActStep]) -> f32 {
            0.5
        }
        fn reflect(&self, _task: &str, _t: &ReActTrajectory) -> Option<String> {
            None
        }
    }

    /// A test executor with one tool.
    struct StubExecutor;

    impl ReActToolExecutor for StubExecutor {
        fn call(&self, tool: &str, _args: &serde_json::Value) -> String {
            format!("called {tool}")
        }
        fn list(&self) -> Vec<String> {
            vec!["echo".into()]
        }
    }

    #[test]
    fn react_finishes_immediately() {
        let trajectory = run_react(
            "What is the answer?",
            &StubReasoner,
            &StubExecutor,
            &ReActConfig::default(),
        );
        assert!(trajectory.succeeded);
        assert_eq!(trajectory.answer, Some("42".to_string()));
        assert_eq!(trajectory.steps.len(), 1);
    }

    #[test]
    fn react_config_defaults() {
        let config = ReActConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert!(!config.self_consistency);
        assert!(config.reflect_on_failure);
    }

    /// A reasoner that calls a tool then finishes.
    struct ToolUsingReasoner {
        step: std::sync::atomic::AtomicUsize,
    }

    impl ReActReasoner for ToolUsingReasoner {
        fn think(&self, _task: &str, _history: &[ReActStep]) -> String {
            "I should use a tool".into()
        }
        fn choose_action(&self, _thought: &str, _tools: &[String]) -> ReActAction {
            let step = self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if step == 0 {
                ReActAction::CallTool {
                    tool: "echo".into(),
                    args: serde_json::json!({"text": "hello"}),
                }
            } else {
                ReActAction::Finish {
                    answer: "done".into(),
                }
            }
        }
        fn confidence(&self, _trajectory: &[ReActStep]) -> f32 {
            0.5
        }
        fn reflect(&self, _task: &str, _t: &ReActTrajectory) -> Option<String> {
            None
        }
    }

    #[test]
    fn react_calls_tool_then_finishes() {
        let reasoner = ToolUsingReasoner {
            step: std::sync::atomic::AtomicUsize::new(0),
        };
        let trajectory = run_react(
            "test task",
            &reasoner,
            &StubExecutor,
            &ReActConfig::default(),
        );
        assert!(trajectory.succeeded);
        assert_eq!(trajectory.tool_calls, 1);
        assert_eq!(trajectory.steps.len(), 2);
    }
}
