//! Agent-trajectory dataset loader.
//!
//! Loads datasets of agent trajectories: goal → plan → act → observe → reflect.
//!
//! Format:
//! ```json
//! {
//!   "goal": "Find the weather in Paris",
//!   "steps": [
//!     {"thought": "I need to call the weather API", "action": "tool_call", "tool": "get_weather", "args": {"city": "Paris"}},
//!     {"observation": "Sunny, 22°C"},
//!     {"thought": "I have the answer", "action": "respond", "response": "It's sunny in Paris, 22°C"}
//!   ],
//!   "outcome": "success"
//! }
//! ```

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// Agent-trajectory loader.
pub struct AgentTrajectoryLoader;

impl Default for AgentTrajectoryLoader {
    fn default() -> Self {
        Self
    }
}

impl AgentTrajectoryLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for AgentTrajectoryLoader {
    fn id(&self) -> &str { "agent-trajectory" }
    fn name(&self) -> &str { "Agent trajectories (goal → plan → act → observe)" }
    fn extensions(&self) -> &[&str] { &["jsonl", "ndjson", "json"] }

    fn matches(&self, path: &Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("trajectory")
            || name.contains("agent")
            || name.contains("react")
            || name.contains("reflexion")
    }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("json");
        match ext {
            "jsonl" | "ndjson" => {
                let file = tokio::fs::File::open(path).await.map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        DatasetError::NotFound(path.display().to_string())
                    } else {
                        DatasetError::Io(e)
                    }
                })?;
                Ok(Box::new(TrajectoryJsonlStream {
                    reader: BufReader::new(file),
                    source: path.to_string_lossy().to_string(),
                    line_buf: String::new(),
                    offset: options.offset.unwrap_or(0),
                    max_records: options.max_records,
                    records_seen: 0,
                    finished: false,
                }))
            }
            _ => {
                let content = crate::loader::read_file_to_string(path).await?;
                let value: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| DatasetError::ParseError { format: "json".into(), error: e.to_string() })?;
                let source = path.to_string_lossy().to_string();
                let records = match value {
                    serde_json::Value::Array(arr) => {
                        arr.into_iter()
                            .skip(options.offset.unwrap_or(0))
                            .take(options.max_records.unwrap_or(usize::MAX))
                            .filter_map(|v| normalize(source.clone(), v))
                            .collect()
                    }
                    other => normalize(source, other).into_iter().collect(),
                };
                Ok(Box::new(crate::streaming::MemoryRecordStream::new(records)))
            }
        }
    }
}

/// Streaming trajectory JSONL reader.
pub struct TrajectoryJsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for TrajectoryJsonlStream {
    async fn next_record(&mut self) -> Result<Option<DatasetRecord>, DatasetError> {
        if self.finished {
            return Ok(None);
        }
        loop {
            self.line_buf.clear();
            let n = self.reader.read_line(&mut self.line_buf).await?;
            if n == 0 {
                self.finished = true;
                return Ok(None);
            }
            let line = self.line_buf.trim();
            if line.is_empty() {
                continue;
            }
            self.records_seen += 1;
            if self.records_seen <= self.offset {
                continue;
            }
            if let Some(max) = self.max_records {
                if self.records_seen > self.offset + max {
                    self.finished = true;
                    return Ok(None);
                }
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| DatasetError::ParseError { format: "jsonl".into(), error: e.to_string() })?;
            if let Some(record) = normalize(self.source.clone(), value) {
                return Ok(Some(record));
            }
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen
    }
}

/// Normalize a trajectory. Must have a `goal` (or `task`) and `steps`
/// (or `trajectory`, `actions`, `trace`).
fn normalize(source: String, value: serde_json::Value) -> Option<DatasetRecord> {
    let obj = value.as_object()?;
    let goal = obj.get("goal").or_else(|| obj.get("task")).or_else(|| obj.get("objective"))?.clone();
    let steps = obj.get("steps")
        .or_else(|| obj.get("trajectory"))
        .or_else(|| obj.get("actions"))
        .or_else(|| obj.get("trace"))?
        .clone();
    let outcome = obj.get("outcome").or_else(|| obj.get("result")).cloned()
        .unwrap_or(serde_json::Value::String("unknown".into()));

    let mut record = DatasetRecord::json(source, serde_json::json!({
        "goal": goal,
        "steps": steps,
        "outcome": outcome,
    }));
    record.modality = crate::loader::Modality::Trajectory;
    record.metadata.insert("format".into(), serde_json::Value::String("agent_trajectory".into()));
    if let Some(step_count) = record.content["steps"].as_array().map(|a| a.len()) {
        record.metadata.insert(
            "step_count".into(),
            serde_json::Value::Number(serde_json::Number::from(step_count)),
        );
    }
    Some(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_trajectory_jsonl() {
        let tmp = tempfile::Builder::new().suffix(".jsonl").tempfile().unwrap();
        let path = tmp.path().with_file_name("agent-trajectory-test.jsonl");
        std::fs::rename(tmp.path(), &path).unwrap();
        std::fs::write(&path, r#"{"goal":"hi","steps":[{"action":"observe"}],"outcome":"success"}"#).unwrap();
        let loader = AgentTrajectoryLoader::new();
        let mut stream = loader.load(&path).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.modality, crate::loader::Modality::Trajectory);
        assert_eq!(r.content["goal"], "hi");
    }
}
