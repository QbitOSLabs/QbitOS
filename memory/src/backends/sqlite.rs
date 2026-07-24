//! SQLite backend (requires `sqlite` feature).
//!
//! Embedded SQL database with full-text search (FTS5), JSON support (JSON1),
//! and vector similarity search via brute-force cosine distance.
//!
//! Perfect for:
//! - Single-node deployments without external databases
//! - Development and testing
//! - Edge devices and embedded systems
//! - Air-gapped environments
//!
//! # Schema
//!
//! ```sql
//! CREATE VIRTUAL TABLE qbitos_memories_fts USING fts5(
//!     text_content, content='qbitos_memories', content_rowid='rowid'
//! );
//! ```

use async_trait::async_trait;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::PathBuf;
use std::sync::Arc;

use crate::backends::{BackendError, BackendStats, MemoryBackend, RangeQuery, SimilarityQuery, SimilarityResult, SortOrder};
use crate::embedding::Embedding;
use crate::object::{Importance, MemoryId, MemoryObject, MemoryType};

/// SQLite backend configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SqliteConfig {
    /// Path to the SQLite database file. Use `:memory:` for in-memory.
    pub path: String,
    /// Whether to enable Write-Ahead Logging (WAL) for concurrent reads.
    pub wal_mode: bool,
    /// Cache size in KB.
    pub cache_size_kb: i64,
    /// Whether to enable FTS5 full-text search.
    pub enable_fts: bool,
    /// Synchronous mode (0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA).
    pub synchronous: i32,
    /// Page size in bytes (4096 is a good default for SSDs).
    pub page_size: u32,
}

impl Default for SqliteConfig {
    fn default() -> Self {
        Self {
            path: std::env::var("QBITOS_SQLITE_PATH")
                .unwrap_or_else(|_| "qbitos.db".into()),
            wal_mode: true,
            cache_size_kb: 64 * 1024, // 64 MB
            enable_fts: true,
            synchronous: 1, // NORMAL (safe with WAL)
            page_size: 4096,
        }
    }
}

/// SQLite backend.
pub struct SqliteBackend {
    conn: Arc<Mutex<Connection>>,
    config: SqliteConfig,
}

impl SqliteBackend {
    /// Create a new SQLite backend, running schema migrations.
    ///
    /// # Errors
    /// Returns [`BackendError`] if the connection or migration fails.
    pub fn new(config: SqliteConfig) -> Result<Self, BackendError> {
        let conn = if config.path == ":memory:" {
            Connection::open_in_memory()
        } else {
            let path = PathBuf::from(&config.path);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            Connection::open(&config.path)
        }
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Configure pragmas for performance.
        let pragmas = [
            format!("PRAGMA page_size = {}", config.page_size),
            format!("PRAGMA cache_size = -{}", config.cache_size_kb),
            format!("PRAGMA synchronous = {}", config.synchronous),
            "PRAGMA temp_store = MEMORY".into(),
            "PRAGMA mmap_size = 268435456".into(), // 256 MB
            "PRAGMA journal_size_limit = 67108864".into(), // 64 MB
        ];
        for pragma in &pragmas {
            conn.execute_batch(pragma)
                .map_err(|e| BackendError::Connection(e.to_string()))?;
        }

        if config.wal_mode {
            conn.execute_batch("PRAGMA journal_mode = WAL")
                .map_err(|e| BackendError::Connection(e.to_string()))?;
            conn.execute_batch("PRAGMA wal_autocheckpoint = 1000")
                .map_err(|e| BackendError::Connection(e.to_string()))?;
        }

        // Create the main table.
        conn.execute_batch(
            r#"CREATE TABLE IF NOT EXISTS qbitos_memories (
                id           BLOB PRIMARY KEY,
                layer        TEXT NOT NULL,
                content      TEXT NOT NULL,
                embedding    BLOB,
                importance   REAL NOT NULL DEFAULT 0.5,
                confidence   REAL NOT NULL DEFAULT 1.0,
                created_at   TEXT NOT NULL,
                accessed_at  TEXT NOT NULL,
                usage_count  INTEGER NOT NULL DEFAULT 0,
                metadata     TEXT NOT NULL DEFAULT '{}',
                tags         TEXT NOT NULL DEFAULT '[]',
                text_content TEXT,
                created_ts   INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS sqlite_mem_layer_idx ON qbitos_memories(layer);
            CREATE INDEX IF NOT EXISTS sqlite_mem_created_ts_idx ON qbitos_memories(created_ts DESC);
            CREATE INDEX IF NOT EXISTS sqlite_mem_importance_idx ON qbitos_memories(importance DESC);
            CREATE INDEX IF NOT EXISTS sqlite_mem_layer_ts_idx ON qbitos_memories(layer, created_ts DESC);
            CREATE INDEX IF NOT EXISTS sqlite_mem_layer_importance_idx ON qbitos_memories(layer, importance DESC);"#,
        )
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Create FTS5 virtual table for full-text search.
        if config.enable_fts {
            conn.execute_batch(
                r#"CREATE VIRTUAL TABLE IF NOT EXISTS qbitos_memories_fts USING fts5(
                    text_content,
                    content='qbitos_memories',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS qbitos_memories_ai AFTER INSERT ON qbitos_memories BEGIN
                    INSERT INTO qbitos_memories_fts(rowid, text_content)
                    VALUES (new.rowid, COALESCE(new.text_content, ''));
                END;

                CREATE TRIGGER IF NOT EXISTS qbitos_memories_ad AFTER DELETE ON qbitos_memories BEGIN
                    INSERT INTO qbitos_memories_fts(qbitos_memories_fts, rowid, text_content)
                    VALUES ('delete', old.rowid, COALESCE(old.text_content, ''));
                END;

                CREATE TRIGGER IF NOT EXISTS qbitos_memories_au AFTER UPDATE ON qbitos_memories BEGIN
                    INSERT INTO qbitos_memories_fts(qbitos_memories_fts, rowid, text_content)
                    VALUES ('delete', old.rowid, COALESCE(old.text_content, ''));
                    INSERT INTO qbitos_memories_fts(rowid, text_content)
                    VALUES (new.rowid, COALESCE(new.text_content, ''));
                END;"#,
            )
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        }

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            config,
        })
    }

    /// Serialize an embedding to a byte vector for storage.
    fn embedding_to_bytes(emb: &Embedding) -> Vec<u8> {
        emb.data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    /// Deserialize an embedding from bytes.
    fn bytes_to_embedding(bytes: &[u8]) -> Embedding {
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(arr)
            })
            .collect();
        Embedding { data }
    }

    /// Parse a layer string into a MemoryType.
    fn parse_layer(s: &str) -> MemoryType {
        match s {
            "sensory" => MemoryType::Sensory,
            "working" => MemoryType::Working,
            "episodic" => MemoryType::Episodic,
            "semantic" => MemoryType::Semantic,
            "procedural" => MemoryType::Procedural,
            "meta" => MemoryType::Meta,
            _ => MemoryType::Episodic,
        }
    }

    /// Parse a row into a MemoryObject.
    fn row_to_object(
        id: Vec<u8>,
        layer: String,
        content: String,
        embedding: Option<Vec<u8>>,
        importance: f64,
        confidence: f64,
        created_at: String,
        accessed_at: String,
        usage_count: i64,
        metadata: String,
    ) -> Result<MemoryObject, BackendError> {
        let uuid_bytes: [u8; 16] = id
            .as_slice()
            .try_into()
            .map_err(|_| BackendError::Serialization("invalid UUID bytes".into()))?;
        let content: serde_json::Value =
            serde_json::from_str(&content).unwrap_or(serde_json::Value::Null);
        let embedding = embedding
            .map(|b| Self::bytes_to_embedding(&b))
            .unwrap_or_else(Embedding::empty);
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null);
        let created = chrono::DateTime::parse_from_rfc3339(&created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        let accessed = chrono::DateTime::parse_from_rfc3339(&accessed_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());

        Ok(MemoryObject {
            id: MemoryId(uuid::Uuid::from_bytes(uuid_bytes)),
            layer: Self::parse_layer(&layer),
            content,
            embedding,
            importance: Importance(importance as f32),
            confidence: confidence as f32,
            created_at: created,
            last_accessed_at: accessed,
            usage_count: usage_count as u32,
            relationships: Vec::new(),
            causal_links: Vec::new(),
            metadata: serde_json::from_value(metadata).unwrap_or_default(),
        })
    }

    /// Brute-force cosine similarity search over all embeddings.
    fn brute_force_search(
        &self,
        query_vec: &[f32],
        k: usize,
        min_score: f32,
        layer_filter: Option<&str>,
    ) -> Result<Vec<SimilarityResult>, BackendError> {
        let conn = self.conn.lock();
        let mut sql = String::from(
            "SELECT id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata FROM qbitos_memories WHERE embedding IS NOT NULL",
        );
        if layer_filter.is_some() {
            sql.push_str(" AND layer = ?");
        }
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let rows = if layer_filter.is_some() {
            stmt.query_map(params![layer_filter.unwrap()], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            })
            .map_err(|e| BackendError::Connection(e.to_string()))?
        } else {
            stmt.query_map([], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            })
            .map_err(|e| BackendError::Connection(e.to_string()))?
        };

        let mut results: Vec<(f32, MemoryObject)> = Vec::new();
        for row in rows {
            let (id, layer, content, embedding_bytes, importance, confidence, created_at, accessed_at, usage_count, metadata) =
                row.map_err(|e| BackendError::Connection(e.to_string()))?;
            let emb = Self::bytes_to_embedding(&embedding_bytes.unwrap_or_default());
            let score = cosine_similarity(query_vec, &emb.data);
            if score >= min_score {
                let obj = Self::row_to_object(
                    id,
                    layer,
                    content,
                    Some(embedding_bytes.unwrap_or_default()),
                    importance,
                    confidence,
                    created_at,
                    accessed_at,
                    usage_count,
                    metadata,
                )?;
                results.push((score, obj));
            }
        }

        results.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        Ok(results
            .into_iter()
            .map(|(score, object)| SimilarityResult { object, score })
            .collect())
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[async_trait]
impl MemoryBackend for SqliteBackend {
    fn name(&self) -> &str {
        "sqlite"
    }
    fn supports_vector_search(&self) -> bool {
        true
    }
    fn is_durable(&self) -> bool {
        self.config.path != ":memory:"
    }
    fn supports_full_text_search(&self) -> bool {
        self.config.enable_fts
    }
    fn supports_range_queries(&self) -> bool {
        true
    }

    async fn store(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let conn = self.conn.lock();
        let id_bytes = *object.id.0.as_bytes();
        let layer = format!("{:?}", object.layer).to_lowercase();
        let content = serde_json::to_string(&object.content)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        let embedding = if object.embedding.data.is_empty() {
            None
        } else {
            Some(Self::embedding_to_bytes(&object.embedding))
        };
        let metadata = serde_json::to_string(&object.metadata)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        let text_content = object
            .content
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| object.content.get("content").and_then(|v| v.as_str()))
            .map(String::from);
        let created_at = object.created_at.to_rfc3339();
        let accessed_at = object.last_accessed_at.to_rfc3339();

        conn.execute(
            "INSERT OR REPLACE INTO qbitos_memories \
             (id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata, text_content) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                id_bytes.as_slice(),
                layer,
                content,
                embedding,
                object.importance.0,
                object.confidence,
                created_at,
                accessed_at,
                object.usage_count as i64,
                metadata,
                text_content,
            ],
        )
        .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn batch_store(&self, objects: &[MemoryObject]) -> Result<(), BackendError> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        for object in objects {
            let id_bytes = *object.id.0.as_bytes();
            let layer = format!("{:?}", object.layer).to_lowercase();
            let content = serde_json::to_string(&object.content)
                .map_err(|e| BackendError::Serialization(e.to_string()))?;
            let embedding = if object.embedding.data.is_empty() {
                None
            } else {
                Some(Self::embedding_to_bytes(&object.embedding))
            };
            let metadata = serde_json::to_string(&object.metadata)
                .map_err(|e| BackendError::Serialization(e.to_string()))?;
            let text_content = object
                .content
                .get("text")
                .and_then(|v| v.as_str())
                .or_else(|| object.content.get("content").and_then(|v| v.as_str()))
                .map(String::from);
            let created_at = object.created_at.to_rfc3339();
            let accessed_at = object.last_accessed_at.to_rfc3339();
            tx.execute(
                "INSERT OR REPLACE INTO qbitos_memories \
                 (id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata, text_content) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    id_bytes.as_slice(),
                    layer,
                    content,
                    embedding,
                    object.importance.0,
                    object.confidence,
                    created_at,
                    accessed_at,
                    object.usage_count as i64,
                    metadata,
                    text_content,
                ],
            )
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, id: MemoryId) -> Result<MemoryObject, BackendError> {
        let conn = self.conn.lock();
        let id_bytes = *id.0.as_bytes();
        let row = conn
            .query_row(
                "SELECT id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata \
                 FROM qbitos_memories WHERE id = ?1",
                params![id_bytes.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<Vec<u8>>>(3)?,
                        row.get::<_, f64>(4)?,
                        row.get::<_, f64>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, i64>(8)?,
                        row.get::<_, String>(9)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        match row {
            Some((id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata)) => {
                Self::row_to_object(id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata)
            }
            None => Err(BackendError::NotFound(id)),
        }
    }

    async fn delete(&self, id: MemoryId) -> Result<(), BackendError> {
        let conn = self.conn.lock();
        let id_bytes = *id.0.as_bytes();
        conn.execute(
            "DELETE FROM qbitos_memories WHERE id = ?1",
            params![id_bytes.as_slice()],
        )
        .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn count(&self) -> Result<usize, BackendError> {
        let conn = self.conn.lock();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM qbitos_memories", [], |row| row.get(0))
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(count as usize)
    }

    async fn search(&self, query: &SimilarityQuery) -> Result<Vec<SimilarityResult>, BackendError> {
        if query.embedding.data.is_empty() {
            return Err(BackendError::Serialization("empty query embedding".into()));
        }
        let layer_filter = query
            .layer_filter
            .as_ref()
            .map(|l| format!("{:?}", l).to_lowercase());
        // SQLite doesn't have native vector search, so we use brute-force.
        self.brute_force_search(
            &query.embedding.data,
            query.k,
            query.min_score,
            layer_filter.as_deref(),
        )
    }

    async fn range_query(&self, query: &RangeQuery) -> Result<Vec<MemoryObject>, BackendError> {
        let conn = self.conn.lock();
        let mut sql = String::from(
            "SELECT id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata FROM qbitos_memories WHERE 1=1",
        );
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        let mut param_idx = 1;

        if let Some(ref layer) = query.layer {
            sql.push_str(&format!(" AND layer = ?{param_idx}"));
            params_vec.push(Box::new(format!("{:?}", layer).to_lowercase()));
            param_idx += 1;
        }
        if let Some(min_imp) = query.min_importance {
            sql.push_str(&format!(" AND importance >= ?{param_idx}"));
            params_vec.push(Box::new(min_imp as f64));
            param_idx += 1;
        }
        if let Some(max_imp) = query.max_importance {
            sql.push_str(&format!(" AND importance <= ?{param_idx}"));
            params_vec.push(Box::new(max_imp as f64));
            param_idx += 1;
        }
        match query.order {
            SortOrder::NewestFirst => sql.push_str(" ORDER BY created_ts DESC"),
            SortOrder::OldestFirst => sql.push_str(" ORDER BY created_ts ASC"),
            SortOrder::ImportanceDesc => sql.push_str(" ORDER BY importance DESC"),
            SortOrder::ImportanceAsc => sql.push_str(" ORDER BY importance ASC"),
        }
        sql.push_str(&format!(" LIMIT ?{param_idx}"));
        params_vec.push(Box::new(query.limit as i64));

        let param_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            })
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let (id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata) =
                row.map_err(|e| BackendError::Connection(e.to_string()))?;
            results.push(Self::row_to_object(
                id, layer, content, embedding, importance, confidence,
                created_at, accessed_at, usage_count, metadata,
            )?);
        }
        Ok(results)
    }

    async fn full_text_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryId, f64)>, BackendError> {
        if !self.config.enable_fts {
            return Err(BackendError::FeatureDisabled("full_text_search"));
        }
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT m.id, bm25(qbitos_memories_fts) AS rank \
                 FROM qbitos_memories_fts fts \
                 JOIN qbitos_memories m ON m.rowid = fts.rowid \
                 WHERE qbitos_memories_fts MATCH ?1 \
                 ORDER BY rank LIMIT ?2",
            )
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let rows = stmt
            .query_map(params![query, limit as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let rank: f64 = row.get(1)?;
                Ok((id_bytes, rank))
            })
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let (id_bytes, rank) = row.map_err(|e| BackendError::Connection(e.to_string()))?;
            let arr: [u8; 16] = id_bytes
                .as_slice()
                .try_into()
                .map_err(|_| BackendError::Serialization("invalid UUID".into()))?;
            results.push((MemoryId(uuid::Uuid::from_bytes(arr)), rank));
        }
        Ok(results)
    }

    async fn stats(&self) -> Result<BackendStats, BackendError> {
        let count = self.count().await?;
        let size = if self.config.path != ":memory:" {
            std::fs::metadata(&self.config.path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };
        Ok(BackendStats {
            name: "sqlite".into(),
            count,
            supports_vector_search: true,
            is_durable: self.is_durable(),
            size_bytes: size,
            avg_latency_us: 0,
        })
    }

    async fn flush(&self) -> Result<(), BackendError> {
        let conn = self.conn.lock();
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::MemoryObject;

    fn make_test_backend() -> SqliteBackend {
        SqliteBackend::new(SqliteConfig {
            path: ":memory:".into(),
            wal_mode: false,
            cache_size_kb: 1024,
            enable_fts: true,
            synchronous: 0,
            page_size: 4096,
        })
        .unwrap()
    }

    fn make_test_object(id: u128, text: &str, embedding: Vec<f32>) -> MemoryObject {
        MemoryObject {
            id: MemoryId(uuid::Uuid::from_u128(id)),
            layer: MemoryType::Episodic,
            content: serde_json::json!({"text": text}),
            embedding: Embedding { data: embedding },
            importance: Importance(0.8),
            confidence: 0.9,
            created_at: chrono::Utc::now(),
            last_accessed_at: chrono::Utc::now(),
            usage_count: 0,
            relationships: Vec::new(),
            causal_links: Vec::new(),
            metadata: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn store_and_get() {
        let backend = make_test_backend();
        let obj = make_test_object(1, "hello world", vec![1.0, 0.0, 0.0]);
        backend.store(&obj).await.unwrap();
        let retrieved = backend.get(obj.id).await.unwrap();
        assert_eq!(retrieved.id, obj.id);
        assert_eq!(retrieved.content, obj.content);
    }

    #[tokio::test]
    async fn delete_works() {
        let backend = make_test_backend();
        let obj = make_test_object(2, "test", vec![0.0, 1.0, 0.0]);
        backend.store(&obj).await.unwrap();
        backend.delete(obj.id).await.unwrap();
        assert!(backend.get(obj.id).await.is_err());
    }

    #[tokio::test]
    async fn count_works() {
        let backend = make_test_backend();
        assert_eq!(backend.count().await.unwrap(), 0);
        backend.store(&make_test_object(3, "a", vec![1.0, 0.0])).await.unwrap();
        backend.store(&make_test_object(4, "b", vec![0.0, 1.0])).await.unwrap();
        assert_eq!(backend.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn vector_search_finds_similar() {
        let backend = make_test_backend();
        backend.store(&make_test_object(10, "cat", vec![1.0, 0.0, 0.0])).await.unwrap();
        backend.store(&make_test_object(11, "dog", vec![0.0, 1.0, 0.0])).await.unwrap();
        backend.store(&make_test_object(12, "kitten", vec![0.9, 0.1, 0.0])).await.unwrap();

        let query = SimilarityQuery {
            embedding: Embedding { data: vec![1.0, 0.0, 0.0] },
            k: 2,
            layer_filter: None,
            min_score: 0.5,
            metadata_filter: None,
            tags: None,
        };
        let results = backend.search(&query).await.unwrap();
        assert_eq!(results.len(), 2);
        // "cat" (exact match) should be first with score ~1.0.
        assert!(results[0].score > 0.99);
    }

    #[tokio::test]
    async fn full_text_search_works() {
        let backend = make_test_backend();
        backend.store(&make_test_object(20, "The quick brown fox", vec![])).await.unwrap();
        backend.store(&make_test_object(21, "A lazy dog sleeps", vec![])).await.unwrap();

        let results = backend.full_text_search("fox", 10).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn batch_store_in_transaction() {
        let backend = make_test_backend();
        let objects = vec![
            make_test_object(30, "first", vec![1.0]),
            make_test_object(31, "second", vec![2.0]),
            make_test_object(32, "third", vec![3.0]),
        ];
        backend.batch_store(&objects).await.unwrap();
        assert_eq!(backend.count().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn range_query_by_importance() {
        let backend = make_test_backend();
        let mut obj1 = make_test_object(40, "high", vec![]);
        obj1.importance = Importance(0.9);
        let mut obj2 = make_test_object(41, "low", vec![]);
        obj2.importance = Importance(0.1);
        backend.store(&obj1).await.unwrap();
        backend.store(&obj2).await.unwrap();

        let query = RangeQuery {
            min_importance: Some(0.5),
            ..RangeQuery::default()
        };
        let results = backend.range_query(&query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].importance.0 >= 0.5);
    }
}
