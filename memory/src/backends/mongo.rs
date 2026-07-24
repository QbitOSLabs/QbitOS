//! MongoDB backend (requires `mongodb` feature).
//!
//! Stores memory objects in a MongoDB collection with:
//! - Atlas Vector Search (cosine similarity via `$vectorSearch` aggregation)
//! - Full-text search via text indexes
//! - Rich JSONB-native queries via the aggregation pipeline
//! - TTL indexes for automatic expiration
//! - Compound indexes for common query patterns
//!
//! Perfect for:
//! - Document-oriented workloads with nested JSON
//! - Multi-region deployments with MongoDB Atlas
//! - Applications already using MongoDB

use async_trait::async_trait;
use mongodb::bson::{doc, Document};
use mongodb::options::{
    CreateCollectionOptions, IndexOptions, VectorSearchOptions,
};
use mongodb::{Client, Collection, Database, IndexModel};
use std::sync::Arc;

use crate::backends::{BackendError, BackendStats, MemoryBackend, RangeQuery, SimilarityQuery, SimilarityResult, SortOrder};
use crate::embedding::Embedding;
use crate::object::{Importance, MemoryId, MemoryObject, MemoryType};

/// MongoDB backend configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MongoConfig {
    /// MongoDB connection URL.
    pub url: String,
    /// Database name.
    pub database: String,
    /// Collection name for memory objects.
    pub collection: String,
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Whether to auto-create indexes on connect.
    pub auto_create_indexes: bool,
}

impl Default for MongoConfig {
    fn default() -> Self {
        Self {
            url: std::env::var("MONGODB_URL")
                .unwrap_or_else(|_| "mongodb://localhost:27017".into()),
            database: "qbitos".into(),
            collection: "memories".into(),
            embedding_dim: 384,
            auto_create_indexes: true,
        }
    }
}

/// MongoDB backend.
pub struct MongoBackend {
    collection: Collection<Document>,
    client: Arc<Client>,
    database: Arc<Database>,
    config: MongoConfig,
}

impl MongoBackend {
    /// Create a new MongoDB backend, connecting and creating indexes.
    ///
    /// # Errors
    /// Returns [`BackendError`] on connection or index creation failure.
    pub async fn new(config: MongoConfig) -> Result<Self, BackendError> {
        let client = Client::with_uri_str(&config.url)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let database = client.database(&config.database);
        let collection = database.collection(&config.collection);

        let backend = Self {
            collection: collection.clone(),
            client: Arc::new(client),
            database: Arc::new(database),
            config,
        };

        if backend.config.auto_create_indexes {
            backend.create_indexes().await?;
        }

        Ok(backend)
    }

    async fn create_indexes(&self) -> Result<(), BackendError> {
        // Layer index
        let layer_idx = IndexModel::builder()
            .keys(doc! { "layer": 1 })
            .options(IndexOptions::builder().name("layer_idx".into()).build())
            .build();
        self.collection
            .create_index(layer_idx)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Importance index (descending)
        let importance_idx = IndexModel::builder()
            .keys(doc! { "importance": -1 })
            .options(IndexOptions::builder().name("importance_idx".into()).build())
            .build();
        self.collection
            .create_index(importance_idx)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Created-at index (descending)
        let created_idx = IndexModel::builder()
            .keys(doc! { "created_at": -1 })
            .options(IndexOptions::builder().name("created_at_idx".into()).build())
            .build();
        self.collection
            .create_index(created_idx)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Agent ID index (sparse)
        let agent_idx = IndexModel::builder()
            .keys(doc! { "agent_id": 1 })
            .options(
                IndexOptions::builder()
                    .name("agent_id_idx".into())
                    .sparse(true)
                    .build(),
            )
            .build();
        self.collection
            .create_index(agent_idx)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Compound: layer + created_at
        let layer_created_idx = IndexModel::builder()
            .keys(doc! { "layer": 1, "created_at": -1 })
            .options(
                IndexOptions::builder()
                    .name("layer_created_idx".into())
                    .build(),
            )
            .build();
        self.collection
            .create_index(layer_created_idx)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Compound: layer + importance
        let layer_imp_idx = IndexModel::builder()
            .keys(doc! { "layer": 1, "importance": -1 })
            .options(
                IndexOptions::builder()
                    .name("layer_importance_idx".into())
                    .build(),
            )
            .build();
        self.collection
            .create_index(layer_imp_idx)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        // Text index for full-text search
        let text_idx = IndexModel::builder()
            .keys(doc! { "text_content": "text" })
            .options(
                IndexOptions::builder()
                    .name("text_content_fts_idx".into())
                    .build(),
            )
            .build();
        let _ = self.collection.create_index(text_idx).await;

        // TTL index for auto-expiration (if expires_at is set)
        let ttl_idx = IndexModel::builder()
            .keys(doc! { "expires_at": 1 })
            .options(
                IndexOptions::builder()
                    .name("ttl_idx".into())
                    .expire_after(std::time::Duration::from_secs(0))
                    .build(),
            )
            .build();
        let _ = self.collection.create_index(ttl_idx).await;

        Ok(())
    }

    fn object_to_doc(object: &MemoryObject) -> Result<Document, BackendError> {
        let mut doc = Document::new();
        doc.insert("_id", object.id.0.to_string());
        doc.insert("layer", format!("{:?}", object.layer).to_lowercase());
        doc.insert(
            "content",
            mongodb::bson::to_bson(&object.content)
                .map_err(|e| BackendError::Serialization(e.to_string()))?,
        );
        if !object.embedding.data.is_empty() {
            doc.insert(
                "embedding",
                object.embedding.data.iter().map(|f| *f as f64).collect::<Vec<_>>(),
            );
        }
        doc.insert("importance", object.importance.0 as f64);
        doc.insert("confidence", object.confidence as f64);
        doc.insert("created_at", object.created_at.timestamp());
        doc.insert("accessed_at", object.last_accessed_at.timestamp());
        doc.insert("usage_count", object.usage_count as i64);
        doc.insert(
            "metadata",
            mongodb::bson::to_bson(&object.metadata)
                .map_err(|e| BackendError::Serialization(e.to_string()))?,
        );
        // Extract text content for full-text search.
        if let Some(text) = object
            .content
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| object.content.get("content").and_then(|v| v.as_str()))
        {
            doc.insert("text_content", text);
        }
        Ok(doc)
    }

    fn doc_to_object(doc: &Document) -> Result<MemoryObject, BackendError> {
        let id_str = doc
            .get_str("_id")
            .map_err(|e| BackendError::Serialization(format!("missing _id: {e}")))?;
        let id = uuid::Uuid::parse_str(id_str)
            .map_err(|e| BackendError::Serialization(format!("invalid UUID: {e}")))?;
        let layer_str = doc
            .get_str("layer")
            .map_err(|e| BackendError::Serialization(format!("missing layer: {e}")))?;
        let layer = match layer_str {
            "sensory" => MemoryType::Sensory,
            "working" => MemoryType::Working,
            "episodic" => MemoryType::Episodic,
            "semantic" => MemoryType::Semantic,
            "procedural" => MemoryType::Procedural,
            "meta" => MemoryType::Meta,
            _ => MemoryType::Episodic,
        };
        let content_bson = doc
            .get("content")
            .ok_or_else(|| BackendError::Serialization("missing content".into()))?;
        let content: serde_json::Value = mongodb::bson::from_bson(content_bson.clone())
            .unwrap_or(serde_json::Value::Null);
        let embedding: Vec<f32> = doc
            .get_array("embedding")
            .ok()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();
        let importance = doc
            .get_f64("importance")
            .unwrap_or(0.5) as f32;
        let confidence = doc
            .get_f64("confidence")
            .unwrap_or(1.0) as f32;
        let created_ts = doc.get_i64("created_at").unwrap_or(0);
        let accessed_ts = doc.get_i64("accessed_at").unwrap_or(0);
        let usage_count = doc.get_i64("usage_count").unwrap_or(0) as u32;
        let metadata_bson = doc.get("metadata").cloned().unwrap_or_default();
        let metadata: serde_json::Value =
            mongodb::bson::from_bson(metadata_bson).unwrap_or(serde_json::Value::Null);

        Ok(MemoryObject {
            id: MemoryId(id),
            layer,
            content,
            embedding: Embedding { data: embedding },
            importance: Importance(importance),
            confidence,
            created_at: chrono::DateTime::from_timestamp(created_ts, 0).unwrap_or_else(chrono::Utc::now),
            last_accessed_at: chrono::DateTime::from_timestamp(accessed_ts, 0).unwrap_or_else(chrono::Utc::now),
            usage_count,
            relationships: Vec::new(),
            causal_links: Vec::new(),
            metadata: serde_json::from_value(metadata).unwrap_or_default(),
        })
    }
}

#[async_trait]
impl MemoryBackend for MongoBackend {
    fn name(&self) -> &str {
        "mongodb"
    }
    fn supports_vector_search(&self) -> bool {
        true
    }
    fn is_durable(&self) -> bool {
        true
    }
    fn supports_full_text_search(&self) -> bool {
        true
    }
    fn supports_range_queries(&self) -> bool {
        true
    }

    async fn store(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let doc = Self::object_to_doc(object)?;
        self.collection
            .insert_one(doc)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn batch_store(&self, objects: &[MemoryObject]) -> Result<(), BackendError> {
        let docs: Vec<Document> = objects
            .iter()
            .map(Self::object_to_doc)
            .collect::<Result<_, _>>()?;
        self.collection
            .insert_many(docs)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, id: MemoryId) -> Result<MemoryObject, BackendError> {
        let filter = doc! { "_id": id.0.to_string() };
        let doc = self
            .collection
            .find_one(filter)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?
            .ok_or(BackendError::NotFound(id))?;
        Self::doc_to_object(&doc)
    }

    async fn delete(&self, id: MemoryId) -> Result<(), BackendError> {
        let filter = doc! { "_id": id.0.to_string() };
        self.collection
            .delete_one(filter)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn count(&self) -> Result<usize, BackendError> {
        let count = self
            .collection
            .count_documents(None)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(count as usize)
    }

    async fn search(&self, query: &SimilarityQuery) -> Result<Vec<SimilarityResult>, BackendError> {
        if query.embedding.data.is_empty() {
            return Err(BackendError::Serialization("empty query embedding".into()));
        }
        // Use MongoDB Atlas Vector Search via the $vectorSearch aggregation stage.
        // Requires a vector_index defined on the collection (see MongoDB Atlas docs).
        let query_vec: Vec<f64> = query.embedding.data.iter().map(|f| *f as f64).collect();
        let limit = query.k as i64;

        // Build the $vectorSearch aggregation pipeline.
        let pipeline = vec![
            doc! {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": (limit * 10) as i64,
                    "limit": limit,
                }
            },
            doc! {
                "$addFields": {
                    "score": { "$meta": "vectorSearchScore" }
                }
            },
        ];

        let mut cursor = self
            .collection
            .aggregate(pipeline)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let mut results = Vec::new();
        while let Some(doc) = cursor
            .next()
            .await
        {
            let doc = doc.map_err(|e| BackendError::Connection(e.to_string()))?;
            let score = doc.get_f64("score").unwrap_or(0.0) as f32;
            let obj = Self::doc_to_object(&doc)?;
            if score >= query.min_score {
                results.push(SimilarityResult {
                    object: obj,
                    score,
                });
            }
        }
        Ok(results)
    }

    async fn range_query(&self, query: &RangeQuery) -> Result<Vec<MemoryObject>, BackendError> {
        let mut filter = Document::new();
        if let Some(ref layer) = query.layer {
            filter.insert("layer", format!("{:?}", layer).to_lowercase());
        }
        if let Some(min_imp) = query.min_importance {
            filter.insert("importance", doc! { "$gte": min_imp as f64 });
        }
        if let Some(max_imp) = query.max_importance {
            let entry = filter
                .entry("importance")
                .or_insert_with(|| mongodb::bson::Bson::Document(Document::new()));
            if let mongodb::bson::Bson::Document(d) = entry {
                d.insert("$lte", max_imp as f64);
            }
        }
        if let Some(since) = query.since {
            filter.insert("created_at", doc! { "$gte": since.timestamp() });
        }
        if let Some(until) = query.until {
            let entry = filter
                .entry("created_at")
                .or_insert_with(|| mongodb::bson::Bson::Document(Document::new()));
            if let mongodb::bson::Bson::Document(d) = entry {
                d.insert("$lt", until.timestamp());
            }
        }

        let sort = match query.order {
            SortOrder::NewestFirst => doc! { "created_at": -1 },
            SortOrder::OldestFirst => doc! { "created_at": 1 },
            SortOrder::ImportanceDesc => doc! { "importance": -1 },
            SortOrder::ImportanceAsc => doc! { "importance": 1 },
        };

        let options = mongodb::options::FindOptions::builder()
            .sort(sort)
            .limit(query.limit as i64)
            .build();

        let mut cursor = self
            .collection
            .find(filter)
            .with_options(options)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let mut results = Vec::new();
        while let Some(doc) = cursor.next().await {
            let doc = doc.map_err(|e| BackendError::Connection(e.to_string()))?;
            results.push(Self::doc_to_object(&doc)?);
        }
        Ok(results)
    }

    async fn full_text_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryId, f64)>, BackendError> {
        let filter = doc! { "$text": { "$search": query } };
        let options = mongodb::options::FindOptions::builder()
            .projection(doc! { "score": { "$meta": "textScore" }, "_id": 1 })
            .sort(doc! { "score": { "$meta": "textScore" } })
            .limit(limit as i64)
            .build();

        let mut cursor = self
            .collection
            .find(filter)
            .with_options(options)
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let mut results = Vec::new();
        while let Some(doc) = cursor.next().await {
            let doc = doc.map_err(|e| BackendError::Connection(e.to_string()))?;
            let id_str = doc
                .get_str("_id")
                .map_err(|e| BackendError::Serialization(e.to_string()))?;
            let id = uuid::Uuid::parse_str(id_str)
                .map_err(|e| BackendError::Serialization(e.to_string()))?;
            let score = doc.get_f64("score").unwrap_or(0.0);
            results.push((MemoryId(id), score));
        }
        Ok(results)
    }

    async fn stats(&self) -> Result<BackendStats, BackendError> {
        let count = self.count().await?;
        Ok(BackendStats {
            name: "mongodb".into(),
            count,
            supports_vector_search: true,
            is_durable: true,
            size_bytes: 0,
            avg_latency_us: 0,
        })
    }
}
