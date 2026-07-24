//! Skill composition — pipelines that chain multiple skills together.

#![allow(missing_docs)]
use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use crate::backends::{InferBackend, ToolBackend};
use crate::executor::SkillExecutor;
use crate::registry::{SkillId, SkillRegistry};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStep {
    pub skill_id: SkillId,
    #[serde(default)]
    pub input_mapping: HashMap<String, String>,
    #[serde(default)]
    pub output_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub outputs: HashMap<String, serde_json::Value>,
    pub final_output: serde_json::Value,
    pub duration_ms: u64,
    pub success: bool,
}

pub struct SkillPipeline {
    steps: Vec<PipelineStep>,
}

impl Default for SkillPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillPipeline {
    pub fn new() -> Self { Self { steps: Vec::new() } }

    pub fn add_step(&mut self, step: PipelineStep) -> &mut PipelineStep {
        self.steps.push(step);
        self.steps.last_mut().unwrap()
    }

    pub fn len(&self) -> usize { self.steps.len() }

    pub fn is_empty(&self) -> bool { self.steps.is_empty() }

    pub async fn execute(
        &self,
        registry: &SkillRegistry,
        tool: &dyn ToolBackend,
        infer: &dyn InferBackend,
    ) -> PipelineResult {
        let start = Instant::now();
        let executor = SkillExecutor::new();
        let mut outputs: HashMap<String, serde_json::Value> = HashMap::new();
        let mut last_context = serde_json::json!({});

        for step in &self.steps {
            let input = if step.input_mapping.is_empty() {
                last_context.clone()
            } else {
                let mut mapped = serde_json::Map::new();
                for (key, source) in &step.input_mapping {
                    if let Some(val) = outputs.get(source) {
                        mapped.insert(key.clone(), val.clone());
                    }
                }
                serde_json::Value::Object(mapped)
            };

            let report = executor.execute(step.skill_id, input, registry, tool, infer).await;
            let output_key = if step.output_key.is_empty() {
                format!("step_{}", step.skill_id)
            } else {
                step.output_key.clone()
            };
            outputs.insert(output_key.clone(), report.context.clone());
            last_context = report.context.clone();

            if !report.status.is_success() {
                return PipelineResult {
                    outputs,
                    final_output: serde_json::Value::Null,
                    duration_ms: start.elapsed().as_millis() as u64,
                    success: false,
                };
            }
        }

        PipelineResult {
            outputs,
            final_output: last_context,
            duration_ms: start.elapsed().as_millis() as u64,
            success: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{NoopInferBackend, NoopToolBackend};
    use crate::registry::{Skill, SkillStep, StepKind};

    #[tokio::test]
    async fn empty_pipeline() {
        let registry = SkillRegistry::new();
        let tool = NoopToolBackend::new();
        let infer = NoopInferBackend::new();
        let pipeline = SkillPipeline::new();
        let result = pipeline.execute(&registry, &tool, &infer).await;
        assert!(result.success);
        assert_eq!(result.outputs.len(), 0);
    }

    #[tokio::test]
    async fn single_step_pipeline() {
        let registry = SkillRegistry::new();
        let mut skill = Skill::new("test", "test skill");
        skill.add_step(SkillStep::new("read", StepKind::Read, serde_json::json!({"input": "${input.text}"})));
        let id = registry.register(skill);
        let tool = NoopToolBackend::new();
        let infer = NoopInferBackend::new();
        let mut pipeline = SkillPipeline::new();
        pipeline.add_step(PipelineStep {
            skill_id: id,
            input_mapping: HashMap::new(),
            output_key: "result".into(),
        });
        let result = pipeline.execute(&registry, &tool, &infer).await;
        assert!(result.success);
        assert!(result.outputs.contains_key("result"));
    }

    #[test]
    fn add_steps() {
        let mut pipeline = SkillPipeline::new();
        pipeline.add_step(PipelineStep { skill_id: 1, input_mapping: HashMap::new(), output_key: "a".into() });
        pipeline.add_step(PipelineStep { skill_id: 2, input_mapping: HashMap::new(), output_key: "b".into() });
        assert_eq!(pipeline.len(), 2);
    }
}
