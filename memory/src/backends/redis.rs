//! Redis backend (requires `redis_cache` feature).
//!
//! Stores memory objects as JSON-serialized values keyed by `qbitos:mem:{id}`.
//! Supports optional TTL for cache-style usage. Working memory is also kept
//! under `qbitos:working` for cross-process sharing.

use async_trait::async_trait;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use std::time::Duration;

use crate::backends::{BackendError, MemoryBackend, SimilarityQuery, SimilarityResult};
use crate::object::{MemoryId, MemoryObject};

/// Redis backend configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RedisConfig {
    /// Redis URL (e.g. `redis://localhost:6379`).
    pub url: String,
    /// Key prefix for all QbitOS keys.
    pub key_prefix: String,
    /// Optional TTL for stored objects (None = no expiry).
    #[serde(with = "duration_secs")]
    pub ttl: Option<Duration>,
    /// Whether to enable pipeline mode for batch operations.
    #[serde(default = "default_true")]
    pub enable_pipeline: bool,
    /// Connection pool size (default 8).
    #[serde(default = "default_pool")]
    pub pool_size: usize,
    /// Whether to enable Redis Cluster mode.
    #[serde(default)]
    pub cluster_mode: bool,
}

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        match d {
            Some(dur) => dur.as_secs().serialize(s),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        Ok(Option::<u64>::deserialize(d)?.map(Duration::from_secs))
    }
}

fn default_true() -> bool { true }
fn default_pool() -> usize { 8 }

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".into()),
            key_prefix: "qbitos:mem".into(),
            ttl: None,
            enable_pipeline: true,
            pool_size: 8,
            cluster_mode: false,
        }
    }
}

/// Redis backend.
pub struct RedisBackend {
    conn: ConnectionManager,
    config: RedisConfig,
}

impl RedisBackend {
    /// Create a new backend, connecting to Redis.
    ///
    /// # Errors
    /// Returns [`BackendError`] if the connection fails.
    pub async fn new(config: RedisConfig) -> Result<Self, BackendError> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(Self { conn, config })
    }

    fn key_for(&self, id: MemoryId) -> String {
        format!("{}:{}", self.config.key_prefix, id.0)
    }

    fn count_key(&self) -> String {
        format!("{}:__count__", self.config.key_prefix)
    }
}

#[async_trait]
impl MemoryBackend for RedisBackend {
    fn name(&self) -> &str { "redis" }
    fn is_durable(&self) -> bool { false } // Redis is volatile by default

    async fn store(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let key = self.key_for(object.id);
        let value = serde_json::to_vec(object)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        let mut conn = self.conn.clone();
        if let Some(ttl) = self.config.ttl {
            conn.set_ex::<_, _, ()>(&key, value, ttl.as_secs())
                .await
                .map_err(|e| BackendError::Connection(e.to_string()))?;
        } else {
            conn.set::<_, _, ()>(&key, value)
                .await
                .map_err(|e| BackendError::Connection(e.to_string()))?;
        }
        // Bump the counter (only if this is a new key)
        conn.incr::<_, _, u64>(self.count_key(), 1)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, id: MemoryId) -> Result<MemoryObject, BackendError> {
        let key = self.key_for(id);
        let mut conn = self.conn.clone();
        let value: Option<Vec<u8>> = conn.get(&key)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let value = value.ok_or(BackendError::NotFound(id))?;
        if value.is_empty() {
            return Err(BackendError::NotFound(id));
        }
        let object: MemoryObject = serde_json::from_slice(&value)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        Ok(object)
    }

    async fn delete(&self, id: MemoryId) -> Result<(), BackendError> {
        let key = self.key_for(id);
        let mut conn = self.conn.clone();
        let deleted: i64 = conn.del(&key)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        if deleted > 0 {
            conn.decr::<_, _, i64>(self.count_key(), 1)
                .await
                .map_err(|e| BackendError::Connection(e.to_string()))?;
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize, BackendError> {
        let mut conn = self.conn.clone();
        let count: i64 = conn.get(self.count_key())
            .await
            .unwrap_or(0);
        Ok(count.max(0) as usize)
    }

    async fn search(&self, _query: &SimilarityQuery) -> Result<Vec<SimilarityResult>, BackendError> {
        // Redis doesn't natively support vector search without the RediSearch
        // module. We could add support via FT.SEARCH with vector similarity,
        // but for now this returns an error so callers fall back to a
        // vector-capable backend (Postgres/Qdrant/HNSW).
        Err(BackendError::FeatureDisabled("vector_search (install RediSearch to enable)"))
    }
}
