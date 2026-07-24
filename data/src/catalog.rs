//! Dataset catalog.
//!
//! A catalog keeps track of all datasets known to the system: their paths,
//! formats, schemas, and statistics. Used by the cognitive kernel to
//! discover what data is available for training or evaluation.

use ahash::AHashMap;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use crate::schema::DatasetSchema;

/// A catalog entry: metadata about a single dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Unique id.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Path or URL to the dataset.
    pub path: PathBuf,
    /// Format id (e.g. `"jsonl"`, `"parquet"`).
    pub format: String,
    /// Number of records (None if unknown).
    pub record_count: Option<usize>,
    /// Size in bytes (None if unknown).
    pub size_bytes: Option<u64>,
    /// Schema (None if not yet detected).
    pub schema: Option<DatasetSchema>,
    /// Tags (e.g. `["training", "qa"]`).
    pub tags: Vec<String>,
    /// When the entry was added to the catalog.
    pub added_at: DateTime<Utc>,
    /// Last-modified time of the source file (if known).
    pub source_modified_at: Option<DateTime<Utc>>,
}

/// The dataset catalog.
#[derive(Clone)]
pub struct DatasetCatalog {
    entries: Arc<RwLock<AHashMap<Uuid, CatalogEntry>>>,
    by_name: Arc<RwLock<AHashMap<String, Uuid>>>,
}

impl DatasetCatalog {
    /// Create a new empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(AHashMap::new())),
            by_name: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    /// Add an entry to the catalog. Returns the entry's id.
    pub fn add(&self, entry: CatalogEntry) -> Uuid {
        let id = entry.id;
        self.by_name.write().insert(entry.name.clone(), id);
        self.entries.write().insert(id, entry);
        id
    }

    /// Register a new dataset by path + format.
    pub fn register(
        &self,
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        format: impl Into<String>,
        tags: Vec<String>,
    ) -> Uuid {
        let name = name.into();
        let path = path.into();
        let format = format.into();
        let size_bytes = std::fs::metadata(&path).ok().map(|m| m.len());
        let source_modified_at = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| DateTime::<Utc>::from_timestamp(d.as_secs() as i64, 0).unwrap_or_else(Utc::now));
        let entry = CatalogEntry {
            id: Uuid::now_v7(),
            name: name.clone(),
            path,
            format,
            record_count: None,
            size_bytes,
            schema: None,
            tags,
            added_at: Utc::now(),
            source_modified_at,
        };
        self.add(entry)
    }

    /// Look up an entry by id.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<CatalogEntry> {
        self.entries.read().get(&id).cloned()
    }

    /// Look up an entry by name.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<CatalogEntry> {
        let id = *self.by_name.read().get(name)?;
        self.get(id)
    }

    /// Update an entry's schema.
    pub fn set_schema(&self, id: Uuid, schema: DatasetSchema) -> bool {
        let mut entries = self.entries.write();
        if let Some(e) = entries.get_mut(&id) {
            e.schema = Some(schema);
            true
        } else {
            false
        }
    }

    /// Update an entry's record count.
    pub fn set_record_count(&self, id: Uuid, count: usize) -> bool {
        let mut entries = self.entries.write();
        if let Some(e) = entries.get_mut(&id) {
            e.record_count = Some(count);
            true
        } else {
            false
        }
    }

    /// List all entries, sorted by name.
    #[must_use]
    pub fn list(&self) -> Vec<CatalogEntry> {
        let mut entries: Vec<_> = self.entries.read().values().cloned().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    /// Find entries by tag.
    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<CatalogEntry> {
        self.entries.read().values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .cloned()
            .collect()
    }

    /// Remove an entry by id.
    pub fn remove(&self, id: Uuid) -> bool {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.remove(&id) {
            self.by_name.write().remove(&entry.name);
            true
        } else {
            false
        }
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Returns `true` if the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

impl Default for DatasetCatalog {
    fn default() -> Self {
        Self::new()
    }
}
