//! Core loader trait and types.
//!
//! Every loader implements [`DatasetLoader`]. Loaders return a
//! [`DatasetRecordStream`] that yields [`DatasetRecord`]s one at a time,
//! so even multi-gigabyte datasets can be processed without loading them
//! fully into memory.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

/// The modality of a record: which sense / channel it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Plain text.
    Text,
    /// Structured JSON.
    Json,
    /// Source code.
    Code,
    /// Image.
    Image,
    /// Audio.
    Audio,
    /// Video.
    Video,
    /// Tabular row.
    Tabular,
    /// Time-series point.
    TimeSeries,
    /// Knowledge-graph triple.
    Triple,
    /// Embedding vector.
    Vector,
    /// Conversation turn.
    Conversation,
    /// Agent trajectory step.
    Trajectory,
    /// Preference pair (chosen / rejected).
    Preference,
    /// Binary blob (file, archive, etc.).
    Binary,
}

/// A single record produced by a loader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRecord {
    /// Unique record id (assigned at load time).
    pub id: Uuid,
    /// Modality.
    pub modality: Modality,
    /// MIME type (best-effort).
    pub mime: String,
    /// Source path or URL.
    pub source: String,
    /// The record's content as JSON (always populated — loaders convert
    /// their native format to JSON for uniform downstream processing).
    pub content: serde_json::Value,
    /// Optional raw bytes (for images, audio, video, binary blobs).
    pub bytes: Option<Vec<u8>>,
    /// Optional embedding (for vector datasets).
    pub embedding: Option<Vec<f32>>,
    /// Free-form metadata.
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// When the record was loaded.
    pub loaded_at: DateTime<Utc>,
}

impl DatasetRecord {
    /// Create a new text record.
    #[must_use]
    pub fn text(source: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            modality: Modality::Text,
            mime: "text/plain".into(),
            source: source.into(),
            content: serde_json::Value::String(content.into()),
            bytes: None,
            embedding: None,
            metadata: BTreeMap::new(),
            loaded_at: Utc::now(),
        }
    }

    /// Create a new JSON record.
    #[must_use]
    pub fn json(source: impl Into<String>, content: serde_json::Value) -> Self {
        Self {
            id: Uuid::now_v7(),
            modality: Modality::Json,
            mime: "application/json".into(),
            source: source.into(),
            content,
            bytes: None,
            embedding: None,
            metadata: BTreeMap::new(),
            loaded_at: Utc::now(),
        }
    }

    /// Create a new binary record.
    #[must_use]
    pub fn binary(source: impl Into<String>, mime: impl Into<String>, bytes: Vec<u8>) -> Self {
        let mime_str = mime.into();
        Self {
            id: Uuid::now_v7(),
            modality: Modality::Binary,
            mime: mime_str,
            source: source.into(),
            content: serde_json::Value::Null,
            bytes: Some(bytes),
            embedding: None,
            metadata: BTreeMap::new(),
            loaded_at: Utc::now(),
        }
    }

    /// Attach an embedding.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Attach a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Options passed to a loader.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadOptions {
    /// Maximum number of records to load (None = no limit).
    pub max_records: Option<usize>,
    /// Offset to start from (for pagination).
    pub offset: Option<usize>,
    /// File encoding override (default: UTF-8).
    pub encoding: Option<String>,
    /// CSV delimiter (default: ',').
    pub csv_delimiter: Option<char>,
    /// Whether to include raw bytes in records (default: false for text formats).
    pub include_bytes: bool,
    /// Filter by modality (None = all).
    pub modality_filter: Option<Modality>,
    /// Additional format-specific options.
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl LoadOptions {
    /// Create a new default options set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of records.
    #[must_use]
    pub fn with_max_records(mut self, max: usize) -> Self {
        self.max_records = Some(max);
        self
    }

    /// Set the offset.
    #[must_use]
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// Statistics about a load operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadStats {
    /// Number of records loaded.
    pub records_loaded: usize,
    /// Number of records skipped (filtered, malformed, etc.).
    pub records_skipped: usize,
    /// Total bytes read.
    pub bytes_read: u64,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Format that was used.
    pub format: String,
}

/// A stream of records. Implementations may yield records from disk, network,
/// or memory. Use `next_record` to pull the next record; the stream ends
/// when `Ok(None)` is returned.
#[async_trait]
pub trait DatasetRecordStream: Send {
    /// Pull the next record. Returns `Ok(None)` when the stream is exhausted.
    ///
    /// # Errors
    /// Returns [`DatasetError`] if reading fails.
    async fn next_record(&mut self) -> Result<Option<DatasetRecord>, DatasetError>;

    /// Returns the total number of records seen so far.
    fn records_seen(&self) -> usize {
        0
    }

    /// Returns the schema of the dataset, if known.
    fn schema(&self) -> Option<&DatasetSchema> {
        None
    }
}

// Re-export for the trait method above.
use crate::schema::DatasetSchema;

/// Errors emitted by loaders.
#[derive(Debug, Error)]
pub enum DatasetError {
    /// The file or resource was not found.
    #[error("not found: {0}")]
    NotFound(String),
    /// The format is not supported.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    /// The file could not be parsed.
    #[error("parse error in {format} file: {error}")]
    ParseError {
        /// Format name.
        format: String,
        /// Error message.
        error: String,
    },
    /// I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// CSV error.
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    /// YAML error.
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// A feature flag is disabled.
    #[error("feature `{feature}` is not enabled; rebuild with `--features {feature}`")]
    FeatureDisabled {
        /// Feature name.
        feature: String,
    },
    /// A custom error from a loader.
    #[error("{0}")]
    Custom(String),
}

impl DatasetError {
    /// Create a custom error.
    #[must_use]
    pub fn custom(msg: impl Into<String>) -> Self {
        Self::Custom(msg.into())
    }
}

/// The trait every loader implements.
#[async_trait]
pub trait DatasetLoader: Send + Sync {
    /// The loader's unique id (e.g. `"jsonl"`, `"parquet"`).
    fn id(&self) -> &str;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// File extensions this loader handles (e.g. `["jsonl", "ndjson"]`).
    fn extensions(&self) -> &[&str];

    /// Whether this loader can handle the given path.
    fn matches(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| self.extensions().iter().any(|x| *x == e))
            .unwrap_or(false)
    }

    /// Detect the schema of a dataset without loading all records.
    ///
    /// Default implementation loads the first record and infers from it.
    async fn detect_schema(&self, path: &Path) -> Result<DatasetSchema, DatasetError> {
        let mut stream = self.load_with_options(path, &LoadOptions::new().with_max_records(1)).await?;
        if let Some(record) = stream.next_record().await? {
            Ok(DatasetSchema::infer_from(&record.content))
        } else {
            Ok(DatasetSchema::empty())
        }
    }

    /// Load a dataset, returning a stream of records.
    ///
    /// # Errors
    /// Returns [`DatasetError`] if the dataset cannot be loaded.
    async fn load(&self, path: &Path) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        self.load_with_options(path, &LoadOptions::default()).await
    }

    /// Load with options.
    ///
    /// # Errors
    /// Returns [`DatasetError`] if the dataset cannot be loaded.
    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError>;
}

/// A shared loader handle.
pub type SharedLoader = Arc<dyn DatasetLoader>;

/// Helper: read a file fully into a string.
///
/// # Errors
/// Propagates I/O errors.
pub async fn read_file_to_string(path: &Path) -> Result<String, DatasetError> {
    tokio::fs::read_to_string(path).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            DatasetError::NotFound(path.display().to_string())
        } else {
            DatasetError::Io(e)
        }
    })
}

/// Helper: read a file fully into bytes.
///
/// # Errors
/// Propagates I/O errors.
pub async fn read_file_to_bytes(path: &Path) -> Result<Vec<u8>, DatasetError> {
    tokio::fs::read(path).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            DatasetError::NotFound(path.display().to_string())
        } else {
            DatasetError::Io(e)
        }
    })
}

/// Helper: convert a `Path` to a lossy string.
#[must_use]
pub fn path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

/// Helper: collect all paths matching a glob pattern.
///
/// # Errors
/// Returns an error if the pattern is invalid.
pub fn glob_paths(pattern: &str) -> Result<Vec<PathBuf>, DatasetError> {
    glob::glob(pattern)
        .map_err(|e| DatasetError::Custom(format!("invalid glob pattern: {e}")))?
        .filter_map(|r| r.ok())
        .collect::<Vec<_>>()
        .into_iter()
        .map(Ok)
        .collect()
}
