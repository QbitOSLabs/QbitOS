//! Vector dataset loader.
//!
//! Loads datasets of pre-computed embeddings with optional metadata.
//!
//! Format (JSONL):
//! ```json
//! {"id": "vec-1", "embedding": [0.1, 0.2, 0.3], "metadata": {"label": "cat"}}
//! ```
//! Format (JSON):
//! ```json
//! {"vectors": [{"id": "vec-1", "embedding": [...], "metadata": {...}}]}
//! ```

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// Vector dataset loader.
pub struct VectorDatasetLoader;

impl Default for VectorDatasetLoader {
    fn default() -> Self {
        Self
    }
}

impl VectorDatasetLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for VectorDatasetLoader {
    fn id(&self) -> &str { "vector" }
    fn name(&self) -> &str { "Vector dataset (embeddings + metadata)" }
    fn extensions(&self) -> &[&str] { &["jsonl", "ndjson", "json"] }

    fn matches(&self, path: &Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("vector") || name.contains("embedding") || name.contains("embed")
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
                Ok(Box::new(VectorJsonlStream {
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

/// Streaming vector JSONL reader.
pub struct VectorJsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for VectorJsonlStream {
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

/// Normalize a vector record.
fn normalize(source: String, value: serde_json::Value) -> Option<DatasetRecord> {
    let obj = value.as_object()?;
    let embedding = obj.get("embedding")?.as_array()?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect::<Vec<f32>>();
    if embedding.is_empty() {
        return None;
    }
    let id = obj.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let metadata = obj.get("metadata").cloned().unwrap_or(serde_json::Value::Object(Default::default()));

    let mut record = DatasetRecord::json(source, serde_json::json!({
        "id": id,
        "metadata": metadata,
    }));
    record.modality = crate::loader::Modality::Vector;
    record.embedding = Some(embedding);
    Some(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_vector_jsonl() {
        let tmp = tempfile::Builder::new().suffix(".jsonl").tempfile().unwrap();
        let path = tmp.path().with_file_name("vector-test.jsonl");
        std::fs::rename(tmp.path(), &path).unwrap();
        std::fs::write(&path, r#"{"id":"v1","embedding":[0.1,0.2,0.3],"metadata":{"label":"cat"}}"#).unwrap();
        let loader = VectorDatasetLoader::new();
        let mut stream = loader.load(&path).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.modality, crate::loader::Modality::Vector);
        assert_eq!(r.embedding.unwrap(), vec![0.1, 0.2, 0.3]);
    }
}
