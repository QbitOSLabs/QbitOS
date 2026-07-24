//! Function-calling dataset loader.
//!
//! Loads datasets that teach the model to invoke tools via structured JSON.
//!
//! Format:
//! ```json
//! {
//!   "tools": [{"name": "get_weather", "schema": {...}}],
//!   "messages": [
//!     {"role": "user", "content": "What's the weather in Paris?"},
//!     {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
//!     {"role": "tool", "tool_call_id": "1", "content": "Sunny, 22°C"}
//!   ]
//! }
//! ```

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// Function-calling loader.
pub struct FunctionCallLoader;

impl Default for FunctionCallLoader {
    fn default() -> Self {
        Self
    }
}

impl FunctionCallLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for FunctionCallLoader {
    fn id(&self) -> &str { "function-call" }
    fn name(&self) -> &str { "Function calling (OpenAI tool_calls)" }
    fn extensions(&self) -> &[&str] { &["jsonl", "ndjson", "json"] }

    fn matches(&self, path: &Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("function")
            || name.contains("tool")
            || name.contains("glm")
            || name.contains("tool_calls")
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
                Ok(Box::new(FunctionCallJsonlStream {
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

/// Streaming function-call JSONL reader.
pub struct FunctionCallJsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for FunctionCallJsonlStream {
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

/// Normalize a function-call example.
///
/// A valid example must have either a `tools` field (list of tool schemas)
/// or a `messages` array containing at least one `tool_calls` or `tool`
/// role message.
fn normalize(source: String, value: serde_json::Value) -> Option<DatasetRecord> {
    let obj = value.as_object()?;
    let tools = obj.get("tools").cloned().unwrap_or(serde_json::Value::Array(vec![]));
    let messages = obj.get("messages")?.as_array()?.clone();
    if messages.is_empty() {
        return None;
    }
    let has_tool_use = messages.iter().any(|m| {
        m.get("tool_calls").is_some() || m.get("role").and_then(|r| r.as_str()) == Some("tool")
    });
    if !has_tool_use && tools.as_array().map_or(true, |t| t.is_empty()) {
        return None;
    }

    let mut record = DatasetRecord::json(source, serde_json::json!({
        "tools": tools,
        "messages": messages,
    }));
    record.modality = crate::loader::Modality::Conversation;
    record.metadata.insert("format".into(), serde_json::Value::String("function_call".into()));
    record.metadata.insert(
        "tool_count".into(),
        serde_json::Value::Number(
            serde_json::Number::from(record.content["tools"].as_array().map(|a| a.len()).unwrap_or(0)),
        ),
    );
    Some(record)
}
