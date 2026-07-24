//! DuckDB backend (requires `duckdb` feature).
//!
//! Embedded analytical (columnar) database optimized for OLAP queries:
//! aggregation, grouping, scanning, and analytical joins.
//!
//! Perfect for:
//! - Analytical queries over memory (e.g. "average importance by layer")
//! - Batch imports and exports
//! - Time-series analysis of memory access patterns
//! - Columnar compression for large datasets
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE qbitos_memories (
//!     id           VARCHAR PRIMARY KEY,
//!     layer        VARCHAR NOT NULL,
//!     content      JSON NOT NULL,
//!     embedding    FLOAT[],
//!     importance   FLOAT NOT NULL DEFAULT 0.5,
//!     confidence   FLOAT NOT NULL DEFAULT 1.0,
//!     created_at   TIMESTAMP NOT NULL,
//!     accessed_at  TIMESTAMP NOT NULL,
//!     usage_count  INTEGER NOT NULL DEFAULT 0,
//!     metadata     JSON NOT NULL DEFAULT '{}',
//!     text_content VARCHAR
//! );
//! ```

use async_trait::async_trait;
use duckdb::{params, Connection};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

use crate::backends::{
    BackendError, BackendStats, MemoryBackend, RangeQuery, SimilarityQuery, SimilarityResult,
    SortOrder,
};
use crate::embedding::Embedding;
use crate::object::{Importance, MemoryId, MemoryObject, MemoryType};

/// DuckDB backend configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DuckdbConfig {
    /// Path to the DuckDB database file. Use `:memory:` for in-memory.
    pub path: String,
    /// Whether to enable the JSON extension.
    pub enable_json: bool,
    /// Threads for parallel query execution.
    pub threads: usize,
    /// Memory limit in bytes (0 = auto).
    pub memory_limit_bytes: u64,
}

impl Default for DuckdbConfig {
    fn default() -> Self {
        Self {
            path: std::env::var("QBITOS_DUCKDB_PATH")
                .unwrap_or_else(|_| "qbitos.duckdb".into()),
            enable_json: true,
            threads: 0, // 0 = auto-detect
            memory_limit_bytes: 0, // 0 = auto
        }
    }
}

/// DuckDB backend.
pub struct DuckdbBackend {
    conn: Arc<Mutex<Connection>>,
    config: DuckdbConfig,
}

impl DuckdbBackend {
    /// Create a new DuckDB backend, running schema migrations.
    ///
    /// # Errors
    /// Returns [`BackendError`] if the connection or migration fails.
    pub fn new(config: DuckdbConfig) -> Result<Self, BackendError> {
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

        // Configure pragmas.
        if config.threads > 0 {
            let _ = conn.execute_batch(&format!("PRAGMA threads={}", config.threads));
        }
        if config.memory_limit_bytes > 0 {
            let mb = config.memory_limit_bytes / (1024 * 1024);
            let _ = conn.execute_batch(&format!("PRAGMA memory_limit='{}MB'", mb));
        }

        // Create the main table (columnar storage).
        conn.execute_batch(
            r#"CREATE TABLE IF NOT EXISTS qbitos_memories (
                id           VARCHAR PRIMARY KEY,
                layer        VARCHAR NOT NULL,
                content      JSON NOT NULL,
                embedding    FLOAT[],
                importance   FLOAT NOT NULL DEFAULT 0.5,
                confidence   FLOAT NOT NULL DEFAULT 1.0,
                created_at   TIMESTAMP NOT NULL,
                accessed_at  TIMESTAMP NOT NULL,
                usage_count  INTEGER NOT NULL DEFAULT 0,
                metadata     JSON NOT NULL DEFAULT '{}',
                text_content VARCHAR
            );

            CREATE INDEX IF NOT EXISTS duckdb_mem_layer_idx ON qbitos_memories(layer);
            CREATE INDEX IF NOT EXISTS duckdb_mem_created_idx ON qbitos_memories(created_at DESC);
            CREATE INDEX IF NOT EXISTS duckdb_mem_importance_idx ON qbitos_memories(importance DESC);"#,
        )
        .map_err(|e| BackendError::Connection(e.to_string()))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            config,
        })
    }

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

    fn row_to_object(
        id: String,
        layer: String,
        content: String,
        embedding: Option<Vec<f32>>,
        importance: f64,
        confidence: f64,
        created_at: String,
        accessed_at: String,
        usage_count: i64,
        metadata: String,
    ) -> Result<MemoryObject, BackendError> {
        let uuid = uuid::Uuid::parse_str(&id)
            .map_err(|e| BackendError::Serialization(format!("invalid UUID: {e}")))?;
        let content: serde_json::Value =
            serde_json::from_str(&content).unwrap_or(serde_json::Value::Null);
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null);
        let created = chrono::DateTime::parse_from_rfc3339(&created_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        let accessed = chrono::DateTime::parse_from_rfc3339(&accessed_at)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        Ok(MemoryObject {
            id: MemoryId(uuid),
            layer: Self::parse_layer(&layer),
            content,
            embedding: Embedding {
                data: embedding.unwrap_or_default(),
            },
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

    /// Run an analytical aggregation query (e.g., "SELECT layer, COUNT(*), AVG(importance) FROM qbitos_memories GROUP BY layer").
    ///
    /// # Errors
    /// Returns [`BackendError`] on database errors.
    pub fn analyze(&self, sql: &str) -> Result<Vec<Vec<serde_json::Value>>, BackendError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let col_count = stmt.column_count();
        let rows = stmt
            .query_map([], |row| {
                let mut vals = Vec::with_capacity(col_count);
                for i in 0..col_count {
                    let val: serde_json::Value = row.get_ref(i).ok().and_then(|r| {
                        match r {
                            duckdb::ValueRef::Null => Some(serde_json::Value::Null),
                            duckdb::ValueRef::Boolean(b) => Some(serde_json::Value::Bool(b)),
                            duckdb::ValueRef::TinyInt(i) => Some(serde_json::Value::from(i)),
                            duckdb::ValueRef::SmallInt(i) => Some(serde_json::Value::from(i)),
                            duckdb::ValueRef::Int(i) => Some(serde_json::Value::from(i)),
                            duckdb::ValueRef::BigInt(i) => Some(serde_json::Value::from(i)),
                            duckdb::ValueRef::Float(f) => {
                                Some(serde_json::Value::from(f as f64))
                            }
                            duckdb::ValueRef::Double(f) => Some(serde_json::Value::from(f)),
                            duckdb::ValueRef::Text(s) => {
                                Some(serde_json::Value::String(String::from_utf8_lossy(s).to_string()))
                            }
                            _ => Some(serde_json::Value::Null),
                        }
                    }).unwrap_or(serde_json::Value::Null);
                    vals.push(val);
                }
                Ok(vals)
            })
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| BackendError::Connection(e.to_string()))?);
        }
        Ok(results)
    }
}

#[async_trait]
impl MemoryBackend for DuckdbBackend {
    fn name(&self) -> &str {
        "duckdb"
    }
    fn supports_vector_search(&self) -> bool {
        true // via brute-force cosine
    }
    fn is_durable(&self) -> bool {
        self.config.path != ":memory:"
    }
    fn supports_range_queries(&self) -> bool {
        true
    }

    async fn store(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let conn = self.conn.lock();
        let id = object.id.0.to_string();
        let layer = format!("{:?}", object.layer).to_lowercase();
        let content = serde_json::to_string(&object.content)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        let embedding = if object.embedding.data.is_empty() {
            None
        } else {
            Some(object.embedding.data.clone())
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
                id,
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
        let conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        for object in objects {
            let id = object.id.0.to_string();
            let layer = format!("{:?}", object.layer).to_lowercase();
            let content = serde_json::to_string(&object.content)
                .map_err(|e| BackendError::Serialization(e.to_string()))?;
            let embedding = if object.embedding.data.is_empty() {
                None
            } else {
                Some(object.embedding.data.clone())
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
                    id,
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
        let id_str = id.0.to_string();
        let result = conn
            .query_row(
                "SELECT id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata \
                 FROM qbitos_memories WHERE id = ?1",
                params![id_str],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<Vec<f32>>>(3)?,
                        row.get::<_, f64>(4)?,
                        row.get::<_, f64>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, i64>(8)?,
                        row.get::<_, String>(9)?,
                    ))
                },
            )
            .map_err(|e| match e {
                duckdb::Error::QueryReturnedNoRows => BackendError::NotFound(id),
                e => BackendError::Connection(e.to_string()),
            })?;

        Self::row_to_object(
            result.0, result.1, result.2, result.3, result.4, result.5,
            result.6, result.7, result.8, result.9,
        )
    }

    async fn delete(&self, id: MemoryId) -> Result<(), BackendError> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM qbitos_memories WHERE id = ?1",
            params![id.0.to_string()],
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
        // DuckDB supports array cosine similarity via list_cosine_distance.
        // We use brute-force with DuckDB's optimized vectorized execution.
        let conn = self.conn.lock();
        let query_vec = format!(
            "[{}]",
            query
                .embedding
                .data
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let limit = query.k as i64;
        let min_score = query.min_score as f64;

        let layer_clause = query
            .layer_filter
            .as_ref()
            .map(|l| format!("AND layer = '{}'", format!("{:?}", l).to_lowercase()))
            .unwrap_or_default();

        let sql = format!(
            r#"SELECT id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata,
                      1 - list_cosine_distance(embedding, '{query_vec}'::FLOAT[]) AS score
               FROM qbitos_memories
               WHERE embedding IS NOT NULL {layer_clause}
               AND 1 - list_cosine_distance(embedding, '{query_vec}'::FLOAT[]) >= {min_score}
               ORDER BY list_cosine_distance(embedding, '{query_vec}'::FLOAT[])
               LIMIT {limit}"#,
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<Vec<f32>>>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, f64>(10)?,
                ))
            })
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            let r = row.map_err(|e| BackendError::Connection(e.to_string()))?;
            let obj = Self::row_to_object(
                r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8, r.9,
            )?;
            results.push(SimilarityResult {
                object: obj,
                score: r.10 as f32,
            });
        }
        Ok(results)
    }

    async fn range_query(&self, query: &RangeQuery) -> Result<Vec<MemoryObject>, BackendError> {
        let conn = self.conn.lock();
        let mut sql = String::from(
            "SELECT id, layer, content, embedding, importance, confidence, created_at, accessed_at, usage_count, metadata FROM qbitos_memories WHERE 1=1",
        );
        if let Some(ref layer) = query.layer {
            sql.push_str(&format!(" AND layer = '{}'", format!("{:?}", layer).to_lowercase()));
        }
        if let Some(min_imp) = query.min_importance {
            sql.push_str(&format!(" AND importance >= {}", min_imp as f64));
        }
        if let Some(max_imp) = query.max_importance {
            sql.push_str(&format!(" AND importance <= {}", max_imp as f64));
        }
        match query.order {
            SortOrder::NewestFirst => sql.push_str(" ORDER BY created_at DESC"),
            SortOrder::OldestFirst => sql.push_str(" ORDER BY created_at ASC"),
            SortOrder::ImportanceDesc => sql.push_str(" ORDER BY importance DESC"),
            SortOrder::ImportanceAsc => sql.push_str(" ORDER BY importance ASC"),
        }
        sql.push_str(&format!(" LIMIT {}", query.limit));

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<Vec<f32>>>(3)?,
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
            let r = row.map_err(|e| BackendError::Connection(e.to_string()))?;
            results.push(Self::row_to_object(
                r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8, r.9,
            )?);
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
            name: "duckdb".into(),
            count,
            supports_vector_search: true,
            is_durable: self.is_durable(),
            size_bytes: size,
            avg_latency_us: 0,
        })
    }
}
