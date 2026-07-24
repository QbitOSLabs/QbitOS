//! Instruction dataset loader.
//!
//! Supports:
//!
//! - **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
//! - **OpenAI JSONL**: `{"prompt": "...", "completion": "..."}`
//! - **Llama format**: `{"instruction": "...", "output": "..."}`
//! - **Dolly**: `{"instruction": "...", "context": "...", "response": "..."}`

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// Instruction loader.
pub struct InstructionLoader;

impl Default for InstructionLoader {
    fn default() -> Self {
        Self
    }
}

impl InstructionLoader {
    /// Create a new instruction loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for InstructionLoader {
    fn id(&self) -> &str { "instruction" }
    fn name(&self) -> &str { "Instruction (Alpaca, OpenAI JSONL, Llama, Dolly)" }
    fn extensions(&self) -> &[&str] { &["jsonl", "ndjson", "json"] }

    fn matches(&self, path: &Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("instruction")
            || name.contains("alpaca")
            || name.contains("dolly")
            || name.contains("oasst")
            || name.contains("flan")
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
                Ok(Box::new(InstructionJsonlStream {
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
                            .filter_map(|v| normalize_instruction(source.clone(), v))
                            .collect()
                    }
                    other => normalize_instruction(source, other).into_iter().collect(),
                };
                Ok(Box::new(crate::streaming::MemoryRecordStream::new(records)))
            }
        }
    }
}

/// Streaming instruction JSONL reader.
pub struct InstructionJsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for InstructionJsonlStream {
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
            if let Some(record) = normalize_instruction(self.source.clone(), value) {
                return Ok(Some(record));
            }
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen
    }
}

/// Normalize an instruction JSON value to a uniform `{"instruction", "input", "output"}` shape.
fn normalize_instruction(source: String, value: serde_json::Value) -> Option<DatasetRecord> {
    let obj = value.as_object()?;
    let instruction = obj.get("instruction").and_then(|v| v.as_str())
        .or_else(|| obj.get("prompt").and_then(|v| v.as_str()))
        .or_else(|| obj.get("question").and_then(|v| v.as_str()))?;
    let output = obj.get("output").and_then(|v| v.as_str())
        .or_else(|| obj.get("completion").and_then(|v| v.as_str()))
        .or_else(|| obj.get("response").and_then(|v| v.as_str()))
        .or_else(|| obj.get("answer").and_then(|v| v.as_str()))?;
    let input = obj.get("input").and_then(|v| v.as_str())
        .or_else(|| obj.get("context").and_then(|v| v.as_str()))
        .unwrap_or("");

    let mut record = DatasetRecord::json(source, serde_json::json!({
        "instruction": instruction,
        "input": input,
        "output": output,
    }));
    record.modality = crate::loader::Modality::Conversation;
    record.metadata.insert("format".into(), serde_json::Value::String("instruction".into()));
    Some(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_alpaca_jsonl() {
        let tmp = tempfile::Builder::new().suffix(".jsonl").tempfile().unwrap();
        let path = tmp.path().with_file_name("alpaca-test.jsonl");
        std::fs::rename(tmp.path(), &path).unwrap();
        std::fs::write(&path, r#"{"instruction":"Say hi","input":"","output":"Hi!"}"#).unwrap();
        let loader = InstructionLoader::new();
        let mut stream = loader.load(&path).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.content["instruction"], "Say hi");
        assert_eq!(r.content["output"], "Hi!");
    }
}
