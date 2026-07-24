//! Preference (DPO / RLHF) dataset loader.
//!
//! Loads chosen/rejected preference pairs used for Direct Preference
//! Optimization and RLHF.
//!
//! Format:
//! ```json
//! {
//!   "prompt": "...",
//!   "chosen": "...",
//!   "rejected": "..."
//! }
//! ```
//! or
//! ```json
//! {
//!   "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
//!   "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
//! }
//! ```

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// Preference (DPO/RLHF) loader.
pub struct PreferenceLoader;

impl Default for PreferenceLoader {
    fn default() -> Self {
        Self
    }
}

impl PreferenceLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for PreferenceLoader {
    fn id(&self) -> &str { "preference" }
    fn name(&self) -> &str { "Preference pairs (DPO, RLHF)" }
    fn extensions(&self) -> &[&str] { &["jsonl", "ndjson", "json"] }

    fn matches(&self, path: &Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("preference")
            || name.contains("dpo")
            || name.contains("rlhf")
            || name.contains("reward")
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
                Ok(Box::new(PreferenceJsonlStream {
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

/// Streaming preference JSONL reader.
pub struct PreferenceJsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for PreferenceJsonlStream {
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

/// Normalize a preference pair. Must have `chosen` and `rejected` fields.
fn normalize(source: String, value: serde_json::Value) -> Option<DatasetRecord> {
    let obj = value.as_object()?;
    let chosen = obj.get("chosen")?.clone();
    let rejected = obj.get("rejected")?.clone();
    let prompt = obj.get("prompt").cloned()
        .or_else(|| obj.get("question").cloned())
        .unwrap_or(serde_json::Value::Null);

    let mut record = DatasetRecord::json(source, serde_json::json!({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }));
    record.modality = crate::loader::Modality::Preference;
    record.metadata.insert("format".into(), serde_json::Value::String("preference".into()));
    Some(record)
}
