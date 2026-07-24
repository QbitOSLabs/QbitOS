//! Skill executor — runs a skill's steps in sequence.

use crate::backends::{InferBackend, ToolBackend};
use crate::registry::{SkillId, SkillRegistry, StepKind};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// The status of a skill execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    /// The skill completed successfully.
    Success,
    /// The skill failed at some step.
    Failure,
    /// The skill was skipped (e.g. not found).
    Skipped,
}

impl ExecutionStatus {
    /// Returns true if the execution succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }
}

/// A record of a single step's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// The step name.
    pub name: String,
    /// The step kind.
    pub kind: StepKind,
    /// Whether this step succeeded.
    pub success: bool,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// The output produced by this step.
    pub output: serde_json::Value,
}

/// A full execution report for a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    /// The skill ID that was executed.
    pub skill_id: SkillId,
    /// The final status.
    pub status: ExecutionStatus,
    /// Per-step records.
    pub steps: Vec<StepRecord>,
    /// Total duration in milliseconds.
    pub duration_ms: u64,
    /// The accumulated context (output of the last step).
    pub context: serde_json::Value,
    /// Error message if the execution failed.
    pub error: Option<String>,
}

/// The skill executor.
pub struct SkillExecutor;

impl SkillExecutor {
    /// Create a new executor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Execute a skill by ID.
    ///
    /// Runs each step in sequence, passing the accumulated context to each step.
    /// Records the execution outcome in the registry.
    pub async fn execute(
        &self,
        skill_id: SkillId,
        input: serde_json::Value,
        registry: &SkillRegistry,
        tool: &dyn ToolBackend,
        infer: &dyn InferBackend,
    ) -> ExecutionReport {
        let start = Instant::now();
        let Some(skill) = registry.get(skill_id) else {
            return ExecutionReport {
                skill_id,
                status: ExecutionStatus::Skipped,
                steps: Vec::new(),
                duration_ms: start.elapsed().as_millis() as u64,
                context: input,
                error: Some(format!("skill {skill_id} not found")),
            };
        };

        let mut context = input;
        let mut steps = Vec::new();

        for step in &skill.steps {
            let step_start = Instant::now();
            let result = match step.kind {
                StepKind::Read => {
                    // Read steps just merge the config into the context.
                    let mut merged = context.clone();
                    if let serde_json::Value::Object(ref mut obj) = merged {
                        if let serde_json::Value::Object(cfg) = &step.config {
                            for (k, v) in cfg {
                                obj.insert(k.clone(), v.clone());
                            }
                        }
                    }
                    Ok(merged)
                }
                StepKind::Infer => {
                    let prompt = step.config.get("prompt")
                        .and_then(|p| p.as_str())
                        .unwrap_or("");
                    infer.infer(prompt, &context).await.map(|text| {
                        let mut ctx = context.clone();
                        if let serde_json::Value::Object(ref mut obj) = ctx {
                            obj.insert("inference".into(), serde_json::Value::String(text));
                        }
                        ctx
                    })
                }
                StepKind::Tool => {
                    let tool_name = step.config.get("tool")
                        .and_then(|t| t.as_str())
                        .unwrap_or("noop");
                    let args = step.config.get("args").cloned().unwrap_or(serde_json::Value::Null);
                    tool.call(tool_name, &args).await.map(|result| {
                        let mut ctx = context.clone();
                        if let serde_json::Value::Object(ref mut obj) = ctx {
                            obj.insert("tool_result".into(), result);
                        }
                        ctx
                    })
                }
                StepKind::Verify => {
                    // Verify steps always succeed in the base executor.
                    Ok(context.clone())
                }
                StepKind::Return => {
                    // Return steps extract the final output.
                    Ok(context.clone())
                }
            };

            let step_duration = step_start.elapsed().as_millis() as u64;
            let (success, output) = match result {
                Ok(ctx) => (true, ctx.clone()),
                Err(e) => (false, serde_json::json!({"error": e.to_string()})),
            };

            steps.push(StepRecord {
                name: step.name.clone(),
                kind: step.kind,
                success,
                duration_ms: step_duration,
                output: output.clone(),
            });

            if !success {
                let report = ExecutionReport {
                    skill_id,
                    status: ExecutionStatus::Failure,
                    steps,
                    duration_ms: start.elapsed().as_millis() as u64,
                    context,
                    error: Some("step failed".into()),
                };
                registry.record_execution_with_latency(
                    skill_id,
                    false,
                    report.duration_ms,
                );
                return report;
            }

            context = output;
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        registry.record_execution_with_latency(skill_id, true, duration_ms);
        ExecutionReport {
            skill_id,
            status: ExecutionStatus::Success,
            steps,
            duration_ms,
            context,
            error: None,
        }
    }
}

impl Default for SkillExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Skill, SkillStep};

    #[tokio::test]
    async fn execute_unknown_skill_returns_skipped() {
        let reg = SkillRegistry::new();
        let exec = SkillExecutor::new();
        let tool = NoopToolBackend::new();
        let infer = NoopInferBackend::new();
        let report = exec.execute(999, serde_json::json!({}), &reg, &tool, &infer).await;
        assert_eq!(report.status, ExecutionStatus::Skipped);
    }

    #[tokio::test]
    async fn execute_simple_skill_succeeds() {
        let reg = SkillRegistry::new();
        let mut skill = Skill::new("test", "test skill");
        skill.add_step(SkillStep::new("read", StepKind::Read, serde_json::json!({"input": "hello"})));
        skill.add_step(SkillStep::new("return", StepKind::Return, serde_json::json!({})));
        let id = reg.register(skill);

        let exec = SkillExecutor::new();
        let tool = NoopToolBackend::new();
        let infer = NoopInferBackend::new();
        let report = exec.execute(id, serde_json::json!({}), &reg, &tool, &infer).await;
        assert_eq!(report.status, ExecutionStatus::Success);
        assert_eq!(report.steps.len(), 2);
        assert!(report.steps.iter().all(|s| s.success));

        // Verify stats were recorded.
        let updated = reg.get(id).unwrap();
        assert_eq!(updated.execution_count, 1);
        assert_eq!(updated.success_count, 1);
    }

    #[tokio::test]
    async fn execute_infer_step_calls_backend() {
        let reg = SkillRegistry::new();
        let mut skill = Skill::new("infer-test", "test");
        skill.add_step(SkillStep::new("infer", StepKind::Infer, serde_json::json!({"prompt": "hello"})));
        let id = reg.register(skill);

        let exec = SkillExecutor::new();
        let tool = NoopToolBackend::new();
        let infer = NoopInferBackend::new();
        let report = exec.execute(id, serde_json::json!({}), &reg, &tool, &infer).await;
        assert_eq!(report.status, ExecutionStatus::Success);
        assert_eq!(infer.call_count(), 1);
    }
}
