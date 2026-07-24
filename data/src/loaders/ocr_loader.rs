//! OCR dataset loader (requires `ocr-loader` feature).
//!
//! Loads datasets of images paired with text annotations:
//!
//! ```json
//! {"image": "path/to/img.jpg", "text": "the quick brown fox"}
//! ```

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// OCR loader.
pub struct OcrLoader;

impl Default for OcrLoader {
    fn default() -> Self {
        Self
    }
}

impl OcrLoader {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for OcrLoader {
    fn id(&self) -> &str { "ocr" }
    fn name(&self) -> &str { "OCR (image + text annotation pairs)" }
    fn extensions(&self) -> &[&str] { &["jsonl", "ndjson"] }

    fn matches(&self, path: &Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("ocr") || name.contains("annotation")
    }

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
        Ok(Box::new(OcrJsonlStream {
            reader: BufReader::new(file),
            source: path.to_string_lossy().to_string(),
            line_buf: String::new(),
            offset: options.offset.unwrap_or(0),
            max_records: options.max_records,
            records_seen: 0,
            finished: false,
        }))
    }
}

/// Streaming OCR JSONL reader.
pub struct OcrJsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for OcrJsonlStream {
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
            let image_path = value.get("image").and_then(|v| v.as_str()).unwrap_or("");
            let text = value.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let mut record = DatasetRecord::json(self.source.clone(), value);
            record.modality = crate::loader::Modality::Image;
            record.metadata.insert("image_path".into(), serde_json::Value::String(image_path.into()));
            record.metadata.insert("caption".into(), serde_json::Value::String(text.into()));
            // Optionally load the image bytes
            if !image_path.is_empty() {
                if let Ok(bytes) = tokio::fs::read(image_path).await {
                    record.bytes = Some(bytes);
                }
            }
            return Ok(Some(record));
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen
    }
}
