//! JSON Lines loader: `.jsonl` / `.ndjson`.
//!
//! Reads one JSON object per line. This is the most common format for
//! instruction tuning, conversation, and agent-trajectory datasets.

use async_trait::async_trait;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::schema::DatasetSchema;

/// JSONL loader for `.jsonl` / `.ndjson` files.
pub struct JsonlLoader;

impl Default for JsonlLoader {
    fn default() -> Self {
        Self
    }
}

impl JsonlLoader {
    /// Create a new JSONL loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for JsonlLoader {
    fn id(&self) -> &str { "jsonl" }
    fn name(&self) -> &str { "JSON Lines (.jsonl, .ndjson)" }
    fn extensions(&self) -> &[&str] { &["jsonl", "ndjson"] }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let file = tokio::fs::File::open(path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                DatasetError::NotFound(path.display().to_string())
            } else {
                DatasetError::Io(e)
            }
        })?;
        let reader = BufReader::new(file);
        Ok(Box::new(JsonlStream {
            reader,
            source: path.to_string_lossy().to_string(),
            line_buf: String::new(),
            records_seen: AtomicUsize::new(0),
            offset: options.offset.unwrap_or(0),
            max_records: options.max_records,
            schema: None,
            finished: false,
        }))
    }
}

/// Streaming JSONL reader.
pub struct JsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    records_seen: AtomicUsize,
    offset: usize,
    max_records: Option<usize>,
    schema: Option<DatasetSchema>,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for JsonlStream {
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
            let seen = self.records_seen.fetch_add(1, Ordering::SeqCst);
            // Skip records before the offset
            if seen < self.offset {
                continue;
            }
            // Stop if we've reached the max
            if let Some(max) = self.max_records {
                if seen >= self.offset + max {
                    self.finished = true;
                    return Ok(None);
                }
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| DatasetError::ParseError { format: "jsonl".into(), error: e.to_string() })?;
            // Capture schema from the first record
            if self.schema.is_none() {
                self.schema = Some(DatasetSchema::infer_from(&value));
            }
            return Ok(Some(DatasetRecord::json(self.source.clone(), value)));
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen.load(Ordering::SeqCst)
    }

    fn schema(&self) -> Option<&DatasetSchema> {
        self.schema.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_jsonl() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n").unwrap();
        let loader = JsonlLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let mut count = 0;
        while stream.next_record().await.unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn load_jsonl_with_offset_and_max() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n{\"a\":4}\n").unwrap();
        let loader = JsonlLoader::new();
        let opts = LoadOptions { offset: Some(1), max_records: Some(2), ..Default::default() };
        let mut stream = loader.load_with_options(tmp.path(), &opts).await.unwrap();
        let r1 = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r1.content["a"], 2);
        let r2 = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r2.content["a"], 3);
        assert!(stream.next_record().await.unwrap().is_none());
    }
}
