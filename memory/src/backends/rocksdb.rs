//! RocksDB backend (requires `rocksdb` feature).
//!
//! Stores memory objects as serialized values in a RocksDB instance with
//! multiple column families:
//!
//! - `objects` — `id → serialized MemoryObject`
//! - `embeddings` — `id → serialized Embedding` (for fast vector fetch)
//! - `index_by_layer` — `layer:id → ()` (for layer-scoped iteration)
//! - `index_by_importance` — `importance_bytes:id → ()` (for importance ordering)
//!
//! RocksDB is a durable embedded KV store — perfect for long-term memory
//! that needs to survive process restarts without an external service.

use async_trait::async_trait;
use parking_lot::RwLock;
use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, DBWithThreadMode, IteratorMode, Options, SingleThreaded, WriteBatch};
use std::path::PathBuf;
use std::sync::Arc;

use crate::backends::{BackendError, MemoryBackend, SimilarityQuery, SimilarityResult};
use crate::embedding::Embedding;
use crate::object::{Importance, MemoryId, MemoryObject, MemoryType};

/// RocksDB backend configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RocksdbConfig {
    /// Path to the RocksDB data directory.
    pub path: PathBuf,
    /// Whether to create the database if it doesn't exist.
    pub create_if_missing: bool,
    /// Whether to enable compression.
    pub enable_compression: bool,
    /// Block cache size in bytes (default 128 MB).
    #[serde(default = "default_block_cache")]
    pub block_cache_size: usize,
    /// Write buffer size in bytes (default 64 MB per CF).
    #[serde(default = "default_write_buffer")]
    pub write_buffer_size: usize,
    /// Max write buffer number (default 3).
    #[serde(default = "default_max_write_buffers")]
    pub max_write_buffer_number: i32,
    /// Level 0 file count trigger (default 4).
    #[serde(default = "default_level0_files")]
    pub level0_file_num_compaction_trigger: i32,
    /// Max background jobs (default 8).
    #[serde(default = "default_max_jobs")]
    pub max_background_jobs: i32,
    /// Bloom filter bits per key (default 10).
    #[serde(default = "default_bloom_bits")]
    pub bloom_bits: f64,
    /// Whether to enable prefix extractor for faster range scans.
    #[serde(default = "default_true")]
    pub enable_prefix_extractor: bool,
}

fn default_block_cache() -> usize { 128 * 1024 * 1024 }
fn default_write_buffer() -> usize { 64 * 1024 * 1024 }
fn default_max_write_buffers() -> i32 { 3 }
fn default_level0_files() -> i32 { 4 }
fn default_max_jobs() -> i32 { 8 }
fn default_bloom_bits() -> f64 { 10.0 }
fn default_true() -> bool { true }

impl Default for RocksdbConfig {
    fn default() -> Self {
        let path = std::env::var("QBITOS_ROCKSDB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/var/lib/qbitos/rocksdb"));
        Self {
            path,
            create_if_missing: true,
            enable_compression: true,
            block_cache_size: 128 * 1024 * 1024,
            write_buffer_size: 64 * 1024 * 1024,
            max_write_buffer_number: 3,
            level0_file_num_compaction_trigger: 4,
            max_background_jobs: 8,
            bloom_bits: 10.0,
            enable_prefix_extractor: true,
        }
    }
}

const CF_OBJECTS: &str = "objects";
const CF_EMBEDDINGS: &str = "embeddings";
const CF_LAYER_INDEX: &str = "index_by_layer";
const CF_IMPORTANCE_INDEX: &str = "index_by_importance";

/// RocksDB backend.
pub struct RocksdbBackend {
    db: Arc<DBWithThreadMode<SingleThreaded>>,
    config: RocksdbConfig,
    count: RwLock<usize>,
}

impl RocksdbBackend {
    /// Create a new backend, opening (or creating) the RocksDB database.
    ///
    /// # Errors
    /// Returns [`BackendError`] if the database cannot be opened.
    pub fn new(config: RocksdbConfig) -> Result<Self, BackendError> {
        let mut opts = Options::default();
        opts.create_if_missing(config.create_if_missing);
        opts.create_missing_column_families(config.create_if_missing);
        if config.enable_compression {
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        }

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_OBJECTS, opts.clone()),
            ColumnFamilyDescriptor::new(CF_EMBEDDINGS, opts.clone()),
            ColumnFamilyDescriptor::new(CF_LAYER_INDEX, opts.clone()),
            ColumnFamilyDescriptor::new(CF_IMPORTANCE_INDEX, opts.clone()),
        ];

        let db = DBWithThreadMode::open_cf_descriptors(&opts, &config.path, cf_descriptors)
            .map_err(|e| BackendError::Connection(format!("rocksdb open failed: {e}")))?;

        // Count existing objects
        let count = {
            let cf = db.cf_handle(CF_OBJECTS)
                .ok_or_else(|| BackendError::Connection("objects CF not found".into()))?;
            let mut n = 0;
            for item in db.iterator_cf(cf, IteratorMode::Start) {
                let _ = item;
                n += 1;
            }
            n
        };

        Ok(Self {
            db: Arc::new(db),
            config,
            count: RwLock::new(count),
        })
    }

    fn cf(&self, name: &str) -> Result<&ColumnFamily, BackendError> {
        self.db.cf_handle(name)
            .ok_or_else(|| BackendError::Connection(format!("CF `{name}` not found")))
    }

    fn id_key(id: MemoryId) -> Vec<u8> {
        id.0.as_bytes().to_vec()
    }

    fn layer_index_key(layer: MemoryType, id: MemoryId) -> Vec<u8> {
        let layer_str = format!("{:?}", layer).to_lowercase();
        let mut key = layer_str.into_bytes();
        key.push(b':');
        key.extend_from_slice(id.0.as_bytes());
        key
    }

    fn importance_index_key(importance: f32, id: MemoryId) -> Vec<u8> {
        // Importance is stored big-endian so that higher importance sorts first
        // when iterating in reverse.
        let importance_bytes = (importance.to_bits()).to_be_bytes();
        let mut key = importance_bytes.to_vec();
        key.extend_from_slice(id.0.as_bytes());
        key
    }
}

#[async_trait]
impl MemoryBackend for RocksdbBackend {
    fn name(&self) -> &str { "rocksdb" }
    fn is_durable(&self) -> bool { true }

    async fn store(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let id_key = Self::id_key(object.id);
        let value = serde_json::to_vec(object)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;

        let cf_objects = self.cf(CF_OBJECTS)?;
        let cf_embeddings = self.cf(CF_EMBEDDINGS)?;
        let cf_layer = self.cf(CF_LAYER_INDEX)?;
        let cf_importance = self.cf(CF_IMPORTANCE_INDEX)?;

        let mut batch = WriteBatch::default();
        batch.put_cf(cf_objects, &id_key, &value);

        // Store embedding separately for fast vector fetch
        if !object.embedding.data.is_empty() {
            let emb_bytes: Vec<u8> = object.embedding.data.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            batch.put_cf(cf_embeddings, &id_key, &emb_bytes);
        }

        // Layer index
        let layer_key = Self::layer_index_key(object.layer, object.id);
        batch.put_cf(cf_layer, &layer_key, b"");

        // Importance index
        let imp_key = Self::importance_index_key(object.importance.0, object.id);
        batch.put_cf(cf_importance, &imp_key, b"");

        self.db.write(batch)
            .map_err(|e| BackendError::Connection(format!("rocksdb write failed: {e}")))?;

        // Bump count (approximate — we don't check if the key already existed)
        *self.count.write() += 1;
        Ok(())
    }

    async fn get(&self, id: MemoryId) -> Result<MemoryObject, BackendError> {
        let id_key = Self::id_key(id);
        let cf_objects = self.cf(CF_OBJECTS)?;
        let value = self.db.get_cf(cf_objects, &id_key)
            .map_err(|e| BackendError::Connection(format!("rocksdb get failed: {e}")))?
            .ok_or(BackendError::NotFound(id))?;
        let mut object: MemoryObject = serde_json::from_slice(&value)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;

        // Try to fetch the embedding from the embeddings CF
        let cf_embeddings = self.cf(CF_EMBEDDINGS)?;
        if let Some(emb_bytes) = self.db.get_cf(cf_embeddings, &id_key)
            .map_err(|e| BackendError::Connection(format!("rocksdb get embedding failed: {e}")))?
        {
            let data: Vec<f32> = emb_bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            object.embedding = Embedding { data };
        }

        Ok(object)
    }

    async fn delete(&self, id: MemoryId) -> Result<(), BackendError> {
        let id_key = Self::id_key(id);

        // First fetch the object so we can remove its index entries
        let object = self.get(id).await.ok();

        let cf_objects = self.cf(CF_OBJECTS)?;
        let cf_embeddings = self.cf(CF_EMBEDDINGS)?;
        let cf_layer = self.cf(CF_LAYER_INDEX)?;
        let cf_importance = self.cf(CF_IMPORTANCE_INDEX)?;

        let mut batch = WriteBatch::default();
        batch.delete_cf(cf_objects, &id_key);
        batch.delete_cf(cf_embeddings, &id_key);

        if let Some(obj) = object {
            let layer_key = Self::layer_index_key(obj.layer, obj.id);
            batch.delete_cf(cf_layer, &layer_key);
            let imp_key = Self::importance_index_key(obj.importance.0, obj.id);
            batch.delete_cf(cf_importance, &imp_key);
        }

        self.db.write(batch)
            .map_err(|e| BackendError::Connection(format!("rocksdb delete failed: {e}")))?;

        *self.count.write() = self.count.read().saturating_sub(1);
        Ok(())
    }

    async fn count(&self) -> Result<usize, BackendError> {
        Ok(*self.count.read())
    }

    async fn flush(&self) -> Result<(), BackendError> {
        self.db.flush()
            .map_err(|e| BackendError::Connection(format!("rocksdb flush failed: {e}")))?;
        Ok(())
    }
}
