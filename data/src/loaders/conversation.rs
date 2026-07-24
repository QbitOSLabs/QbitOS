//! Conversation dataset loader.
//!
//! Supports the most common conversation dataset formats:
//!
//! - **ShareGPT JSON**: `{"conversations": [{"from": "human", "value": "..."}, ...]}`
//! - **ChatML**: `{"messages": [{"role": "user", "content": "..."}, ...]}`
//! - **OpenAI messages**: same as ChatML
//! - **Llama chat**: `[[ "user", "..."], ["assistant", "..."]]`
//!
//! The loader auto-detects the format from the JSON structure and emits
//! one record per conversation (with the conversation normalized to the
//! OpenAI messages format).

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// Conversation loader.
pub struct ConversationLoader;

impl Default for ConversationLoader {
    fn default() -> Self {
        Self
    }
}

impl ConversationLoader {
    /// Create a new conversation loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for ConversationLoader {
    fn id(&self) -> &str { "conversation" }
    fn name(&self) -> &str { "Conversation (ShareGPT, ChatML, OpenAI, Llama)" }
    fn extensions(&self) -> &[&str] { &["jsonl", "ndjson", "json"] }

    fn matches(&self, path: &Path) -> bool {
        // Match .jsonl/.json files whose name suggests conversation data
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("conversation")
            || name.contains("chat")
            || name.contains("sharegpt")
            || name.contains("dialog")
            || name.contains("messages")
    }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        // Detect format by reading the first non-empty line / whole file
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
                Ok(Box::new(ConversationJsonlStream {
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
                // Single JSON file: parse, normalize, emit one record per conversation
                let content = crate::loader::read_file_to_string(path).await?;
                let value: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| DatasetError::ParseError { format: "json".into(), error: e.to_string() })?;
                let source = path.to_string_lossy().to_string();
                let records = match value {
                    serde_json::Value::Array(arr) => {
                        arr.into_iter()
                            .skip(options.offset.unwrap_or(0))
                            .take(options.max_records.unwrap_or(usize::MAX))
                            .filter_map(|v| normalize_conversation(source.clone(), v))
                            .collect()
                    }
                    other => normalize_conversation(source, other).into_iter().collect(),
                };
                Ok(Box::new(crate::streaming::MemoryRecordStream::new(records)))
            }
        }
    }
}

/// Streaming conversation JSONL reader.
pub struct ConversationJsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for ConversationJsonlStream {
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
            if let Some(record) = normalize_conversation(self.source.clone(), value) {
                return Ok(Some(record));
            }
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen
    }
}

/// Normalize a conversation JSON value to the OpenAI messages format.
///
/// Returns `None` if the value is not a recognizable conversation.
fn normalize_conversation(source: String, value: serde_json::Value) -> Option<DatasetRecord> {
    let messages = if let Some(convs) = value.get("conversations") {
        // ShareGPT format
        if let Some(arr) = convs.as_array() {
            arr.iter().map(|c| {
                let from = c.get("from").and_then(|v| v.as_str()).unwrap_or("user");
                let value = c.get("value").and_then(|v| v.as_str()).unwrap_or("");
                let role = match from {
                    "human" | "user" => "user",
                    "gpt" | "assistant" | "bot" => "assistant",
                    "system" => "system",
                    "tool" | "function" => "tool",
                    _ => "user",
                };
                serde_json::json!({"role": role, "content": value})
            }).collect::<Vec<_>>()
        } else {
            return None;
        }
    } else if let Some(msgs) = value.get("messages") {
        // ChatML / OpenAI format
        if let Some(arr) = msgs.as_array() {
            arr.clone()
        } else {
            return None;
        }
    } else if let Some(arr) = value.as_array() {
        // Llama format: array of (role, content) tuples
        arr.iter().filter_map(|item| {
            if let Some(pair) = item.as_array() {
                if pair.len() == 2 {
                    let role = pair[0].as_str().unwrap_or("user");
                    let content = pair[1].as_str().unwrap_or("");
                    Some(serde_json::json!({"role": role, "content": content}))
                } else {
                    None
                }
            } else if let Some(_obj) = item.as_object() {
                // Already in {role, content} format
                Some(item.clone())
            } else {
                None
            }
        }).collect::<Vec<_>>()
    } else {
        return None;
    };

    if messages.is_empty() {
        return None;
    }

    let mut record = DatasetRecord::json(source, serde_json::Value::Array(messages));
    record.modality = crate::loader::Modality::Conversation;
    record.metadata.insert("format".into(), serde_json::Value::String("openai_messages".into()));
    record.metadata.insert(
        "turn_count".into(),
        serde_json::Value::Number(serde_json::Number::from(record.content.as_array().map(|a| a.len()).unwrap_or(0))),
    );
    Some(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_sharegpt_jsonl() {
        let tmp = tempfile::Builder::new().suffix(".jsonl").tempfile().unwrap();
        let name = tmp.path().file_name().unwrap().to_str().unwrap();
        // Rename to include "conversation" in the name so the loader matches
        let conv_path = tmp.path().with_file_name(format!("conversation-{name}"));
        std::fs::rename(tmp.path(), &conv_path).unwrap();
        std::fs::write(&conv_path, r#"{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"hello"}]}"#).unwrap();
        let loader = ConversationLoader::new();
        let mut stream = loader.load(&conv_path).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.modality, crate::loader::Modality::Conversation);
        let msgs = r.content.as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
    }
}
