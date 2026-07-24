//! Sled backend (requires `sled` feature).
//!
//! Stores memory objects in an embedded Sled tree. Sled is a lightweight
//! embedded KV store — pure Rust, no C++ dependency.

use async_trait::async_trait;
use parking_lot::RwLock;
use sled::{Db, Tree};
use std::path::PathBuf;
use std::sync::Arc;

use crate::backends::{BackendError, MemoryBackend, SimilarityQuery, SimilarityResult};
use crate::embedding::Embedding;
use crate::object::{MemoryId, MemoryObject};

/// Sled backend configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SledConfig {
    /// Path to the Sled data directory.
    pub path: PathBuf,
    /// Cache size in bytes.
    pub cache_capacity: u64,
}

impl Default for SledConfig {
    fn default() -> Self {
        let path = std::env::var("QBITOS_SLED_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/var/lib/qbitos/sled"));
        Self {
            path,
            cache_capacity: 1024 * 1024 * 1024,
        }
    }
}

/// Sled backend.
pub struct SledBackend {
    db: Arc<Db>,
    objects: Tree,
    embeddings: Tree,
    config: SledConfig,
    count: RwLock<usize>,
}

impl SledBackend {
    /// Create a new backend, opening (or creating) the Sled database.
    pub fn new(config: SledConfig) -> Result<Self, BackendError> {
        let db = sled::Config::default()
            .path(&config.path)
            .cache_capacity(config.cache_capacity)
            .flush_every_ms(Some(500))
            .open()
            .map_err(|e| BackendError::Connection(format!("sled open failed: {e}")))?;

        let objects = db.open_tree("objects")
            .map_err(|e| BackendError::Connection(format!("sled objects tree failed: {e}")))?;
        let embeddings = db.open_tree("embeddings")
            .map_err(|e| BackendError::Connection(format!("sled embeddings tree failed: {e}")))?;

        let count = objects.len();
        Ok(Self { db: Arc::new(db), objects, embeddings, config, count: RwLock::new(count) })
    }

    fn id_key(id: MemoryId) -> Vec<u8> {
        id.0.as_bytes().to_vec()
    }
}

#[async_trait]
impl MemoryBackend for SledBackend {
    fn name(&self) -> &str { "sled" }
    fn is_durable(&self) -> bool { true }

    async fn store(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let id_key = Self::id_key(object.id);
        let value = serde_json::to_vec(object)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;

        // Sled operations are fast enough to run directly on the async task
        // (typically <100μs for small values).
        self.objects.insert(&id_key, value)
            .map_err(|e| BackendError::Connection(format!("sled insert failed: {e}")))?;

        if !object.embedding.data.is_empty() {
            let emb_bytes: Vec<u8> = object.embedding.data.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            self.embeddings.insert(&id_key, emb_bytes)
                .map_err(|e| BackendError::Connection(format!("sled embedding insert failed: {e}")))?;
        }

        *self.count.write() += 1;
        Ok(())
    }

    async fn get(&self, id: MemoryId) -> Result<MemoryObject, BackendError> {
        let id_key = Self::id_key(id);
        let value = self.objects.get(&id_key)
            .map_err(|e| BackendError::Connection(format!("sled get failed: {e}")))?
            .ok_or(BackendError::NotFound(id))?;

        let mut object: MemoryObject = serde_json::from_slice(&value)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;

        // Fetch embedding
        if let Ok(Some(emb_bytes)) = self.embeddings.get(&id_key) {
            let data: Vec<f32> = emb_bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            object.embedding = Embedding { data };
        }

        Ok(object)
    }

    async fn delete(&self, id: MemoryId) -> Result<(), BackendError> {
        let id_key = Self::id_key(id);
        self.objects.remove(&id_key)
            .map_err(|e| BackendError::Connection(format!("sled delete failed: {e}")))?;
        let _ = self.embeddings.remove(&id_key);
        // Decrement count — must release read lock before acquiring write lock
        let current = *self.count.read();
        *self.count.write() = current.saturating_sub(1);
        Ok(())
    }

    async fn count(&self) -> Result<usize, BackendError> {
        Ok(*self.count.read())
    }

    async fn flush(&self) -> Result<(), BackendError> {
        self.db.flush()
            .map_err(|e| BackendError::Connection(format!("sled flush failed: {e}")))?;
        Ok(())
    }
}
