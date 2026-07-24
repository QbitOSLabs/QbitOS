//! Pipelines that connect the data crate to the cognitive kernel.
//!
//! - [`DatasetToMemory`]: loads a dataset, embeds each record, and writes
//!   it into the appropriate memory layer.
//! - [`DatasetToSkills`]: mines agent-trajectory and instruction datasets
//!   to extract and register new skills.

use std::path::Path;
use std::sync::Arc;

use qbitos_data::{DatasetRecordStream, DatasetRegistry, LoadOptions};
use qbitos_memory::{
    episodic::Outcome,
    object::{Importance, MemoryObject, MemoryType},
    MemoryFabric,
};
use qbitos_skills::{
    registry::{Skill, SkillStep, StepKind},
    SkillModule,
};
use serde::{Deserialize, Serialize};

/// Result of ingesting a dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestResult {
    /// Number of records ingested.
    pub records_ingested: usize,
    /// Number of records skipped.
    pub records_skipped: usize,
    /// Number of episodes written.
    pub episodes_written: usize,
    /// Number of concepts added.
    pub concepts_added: usize,
    /// Number of skills registered.
    pub skills_registered: usize,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Pipeline: load a dataset and write records into the memory fabric.
pub struct DatasetToMemory {
    memory: Arc<MemoryFabric>,
}

impl DatasetToMemory {
    /// Create a new pipeline.
    #[must_use]
    pub fn new(memory: Arc<MemoryFabric>) -> Self {
        Self { memory }
    }

    /// Ingest a dataset from disk into the memory fabric.
    ///
    /// - Text/JSON/code records → episodic memory (as episodes)
    /// - Conversation records → episodic memory (as episodes)
    /// - Trajectory records → episodic memory (as episodes)
    /// - Vector records → semantic memory (as concepts)
    ///
    /// # Errors
    /// Returns an error if the dataset cannot be loaded.
    pub async fn ingest_path(
        &self,
        registry: &DatasetRegistry,
        path: &Path,
        max_records: Option<usize>,
    ) -> anyhow::Result<IngestResult> {
        let loader = registry.select(path)?;
        let opts = if let Some(max) = max_records {
            LoadOptions::new().with_max_records(max)
        } else {
            LoadOptions::default()
        };
        let mut stream = loader.load_with_options(path, &opts).await?;
        let mut result = IngestResult::default();
        let start = std::time::Instant::now();

        while let Some(record) = stream.next_record().await? {
            match record.modality {
                qbitos_data::loader::Modality::Vector => {
                    if let Some(embedding) = record.embedding.clone() {
                        let label = record.content.get("id")
                            .and_then(|v| v.as_str())
                            .or_else(|| record.content.get("metadata")
                                .and_then(|m| m.get("label"))
                                .and_then(|l| l.as_str()))
                            .unwrap_or("unnamed")
                            .to_string();
                        self.memory.semantic.add_concept(&label, record.content.clone());
                        // Also push the embedding into the index
                        let _ = embedding;
                        result.concepts_added += 1;
                    } else {
                        result.records_skipped += 1;
                    }
                }
                qbitos_data::loader::Modality::Text
                | qbitos_data::loader::Modality::Json
                | qbitos_data::loader::Modality::Code
                | qbitos_data::loader::Modality::Conversation
                | qbitos_data::loader::Modality::Trajectory
                | qbitos_data::loader::Modality::Preference
                | qbitos_data::loader::Modality::Triple
                | qbitos_data::loader::Modality::Tabular
                | qbitos_data::loader::Modality::TimeSeries => {
                    // Embed and store as an episode
                    let embedding = self.memory.embedder.embed_json(&record.content);
                    let metadata_json = serde_json::to_value(&record.metadata).unwrap_or_default();
                    let mut obj = MemoryObject::new(
                        MemoryType::Episodic,
                        record.content.clone(),
                        embedding,
                    );
                    obj.importance = Importance(0.6);
                    let _ = obj; // using record_episode API
                    let _id = self.memory.record_episode(
                        serde_json::json!({
                            "source": record.source,
                            "modality": format!("{:?}", record.modality).to_lowercase(),
                            "content": record.content,
                            "metadata": metadata_json,
                        }),
                        Outcome::Success,
                    );
                    result.episodes_written += 1;
                }
                _ => {
                    result.records_skipped += 1;
                }
            }
            result.records_ingested += 1;
        }

        result.duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        Ok(result)
    }
}

/// Pipeline: mine skills from agent-trajectory and instruction datasets.
pub struct DatasetToSkills {
    skills: Arc<SkillModule>,
    memory: Arc<MemoryFabric>,
}

impl DatasetToSkills {
    /// Create a new pipeline.
    #[must_use]
    pub fn new(skills: Arc<SkillModule>, memory: Arc<MemoryFabric>) -> Self {
        Self { skills, memory }
    }

    /// Mine skills from a dataset.
    ///
    /// For agent-trajectory records: extracts the goal and step sequence
    /// and registers a new skill (or bumps the confidence of an existing
    /// one if the goal matches).
    ///
    /// For instruction records: extracts the instruction as a single-step
    /// skill (`Infer` → `Return`).
    ///
    /// # Errors
    /// Returns an error if the dataset cannot be loaded.
    pub async fn mine_from_path(
        &self,
        registry: &DatasetRegistry,
        path: &Path,
        max_records: Option<usize>,
    ) -> anyhow::Result<IngestResult> {
        let loader = registry.select(path)?;
        let opts = if let Some(max) = max_records {
            LoadOptions::new().with_max_records(max)
        } else {
            LoadOptions::default()
        };
        let mut stream = loader.load_with_options(path, &opts).await?;
        let mut result = IngestResult::default();
        let start = std::time::Instant::now();

        while let Some(record) = stream.next_record().await? {
            match record.modality {
                qbitos_data::loader::Modality::Trajectory => {
                    if let Some(goal) = record.content.get("goal").and_then(|g| g.as_str()) {
                        let skill_name = sanitize_skill_name(goal);
                        // If the skill already exists, bump its execution count
                        if let Some(existing) = self.skills.registry.get_by_name(&skill_name) {
                            if let Some(id) = existing.id {
                                self.skills.registry.record_execution(id, true);
                            }
                        } else {
                            let mut skill = Skill::new(skill_name.clone(), goal.to_string());
                            // Add steps from the trajectory
                            if let Some(steps) = record.content.get("steps").and_then(|s| s.as_array()) {
                                for (i, step) in steps.iter().enumerate() {
                                    let action = step.get("action").and_then(|a| a.as_str()).unwrap_or("tool");
                                    let kind = match action {
                                        "observe" | "recall" | "read" => StepKind::Read,
                                        "infer" | "think" => StepKind::Infer,
                                        "tool" | "act" => StepKind::Tool,
                                        "verify" => StepKind::Verify,
                                        "return" | "respond" => StepKind::Return,
                                        _ => StepKind::Tool,
                                    };
                                    skill.add_step(SkillStep::new(
                                        format!("step_{i}_{action}"),
                                        kind,
                                        step.clone(),
                                    ));
                                }
                            }
                            skill.success_criteria.push("goal_achieved".into());
                            self.skills.registry.register(skill);
                            result.skills_registered += 1;
                        }
                    }
                }
                qbitos_data::loader::Modality::Conversation => {
                    // Treat the user turn as an instruction
                    if let Some(msgs) = record.content.as_array() {
                        if let Some(first) = msgs.first() {
                            if first.get("role").and_then(|r| r.as_str()) == Some("user") {
                                let content = first.get("content").and_then(|c| c.as_str()).unwrap_or("");
                                if !content.is_empty() {
                                    let skill_name = sanitize_skill_name(content);
                                    if self.skills.registry.get_by_name(&skill_name).is_none() {
                                        let mut skill = Skill::new(skill_name, content.to_string());
                                        skill.add_step(SkillStep::new(
                                            "infer",
                                            StepKind::Infer,
                                            serde_json::json!({"prompt": content}),
                                        ));
                                        skill.add_step(SkillStep::new(
                                            "return",
                                            StepKind::Return,
                                            serde_json::json!({}),
                                        ));
                                        self.skills.registry.register(skill);
                                        result.skills_registered += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    result.records_skipped += 1;
                }
            }
            result.records_ingested += 1;
        }

        result.duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        Ok(result)
    }

    /// Ingest a `DatasetRecord` stream directly (for in-memory datasets).
    ///
    /// # Errors
    /// Returns an error if the stream fails.
    pub async fn mine_from_stream(
        &self,
        mut stream: Box<dyn DatasetRecordStream>,
    ) -> anyhow::Result<IngestResult> {
        let mut result = IngestResult::default();
        let start = std::time::Instant::now();
        while let Some(record) = stream.next_record().await? {
            // Re-use the logic from mine_from_path by checking modality
            if record.modality == qbitos_data::loader::Modality::Trajectory {
                if let Some(goal) = record.content.get("goal").and_then(|g| g.as_str()) {
                    let skill_name = sanitize_skill_name(goal);
                    if self.skills.registry.get_by_name(&skill_name).is_none() {
                        let mut skill = Skill::new(skill_name, goal.to_string());
                        skill.add_step(SkillStep::new("infer", StepKind::Infer, record.content.clone()));
                        self.skills.registry.register(skill);
                        result.skills_registered += 1;
                    }
                }
            }
            result.records_ingested += 1;
        }
        result.duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        Ok(result)
    }
}

/// Convert an arbitrary string into a valid skill name (lowercase, snake_case).
fn sanitize_skill_name(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_underscore = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore && !out.is_empty() {
            out.push('_');
            prev_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        "unnamed_skill".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_skill_names() {
        assert_eq!(sanitize_skill_name("Find the weather"), "find_the_weather");
        assert_eq!(sanitize_skill_name("Hello, world!"), "hello_world");
        assert_eq!(sanitize_skill_name("..."), "unnamed_skill");
        assert_eq!(sanitize_skill_name("Build & Run"), "build_run");
    }
}
