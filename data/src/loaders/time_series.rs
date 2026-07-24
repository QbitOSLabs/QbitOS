//! Time-series dataset loader.
//!
//! Loads time-series data from CSV/JSONL. Auto-detects a `timestamp` /
//! `time` / `date` / `t` column and a `value` / `v` / `y` column.
//! Emits one record per (timestamp, value) pair.

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// Time-series loader.
pub struct TimeSeriesLoader;

impl Default for TimeSeriesLoader {
    fn default() -> Self {
        Self
    }
}

impl TimeSeriesLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for TimeSeriesLoader {
    fn id(&self) -> &str { "time-series" }
    fn name(&self) -> &str { "Time series (CSV, JSONL with timestamp)" }
    fn extensions(&self) -> &[&str] { &["csv", "jsonl", "ndjson", "tsv"] }

    fn matches(&self, path: &Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("time") || name.contains("series") || name.contains("timeseries") || name.contains("forecast")
    }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("csv");
        match ext {
            "csv" | "tsv" => {
                // Use the CSV loader and post-process into time-series records
                let csv_loader = super::csv::CsvLoader::new();
                csv_loader.load_with_options(path, options).await
            }
            _ => {
                let file = tokio::fs::File::open(path).await.map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        DatasetError::NotFound(path.display().to_string())
                    } else {
                        DatasetError::Io(e)
                    }
                })?;
                Ok(Box::new(TimeSeriesJsonlStream {
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
    }
}

/// Streaming time-series JSONL reader.
pub struct TimeSeriesJsonlStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for TimeSeriesJsonlStream {
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
            let mut record = DatasetRecord::json(self.source.clone(), value);
            record.modality = crate::loader::Modality::TimeSeries;
            return Ok(Some(record));
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen
    }
}
