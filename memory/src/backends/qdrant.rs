//! Qdrant backend (requires `qdrant` feature).
//!
//! Stores memory objects as points in a Qdrant collection. Each point's
//! payload contains the full `MemoryObject` JSON; the vector is the
//! object's embedding. Supports upsert, search, retrieve, delete.

#![allow(deprecated)]

use async_trait::async_trait;
use std::sync::Arc;

use qdrant_client::qdrant::{
    self, point_id::PointIdOptions, points_selector::PointsSelectorOneOf,
    with_payload_selector, with_vectors_selector, CollectionExistsRequest, CountPoints,
    CreateCollection, DeletePoints, Distance, GetPoints, PointId, PointStruct, PointsIdsList,
    PointsSelector, SearchPoints, UpsertPoints, VectorParams, Vectors, VectorsConfig,
    WithPayloadSelector, WithVectorsSelector,
};
use qdrant_client::Qdrant;

use crate::backends::{BackendError, MemoryBackend, SimilarityQuery, SimilarityResult};
use crate::embedding::Embedding;
use crate::object::{MemoryId, MemoryObject};

/// Qdrant backend configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QdrantConfig {
    /// Qdrant server URL (e.g. `http://localhost:6333`).
    pub url: String,
    /// Collection name.
    pub collection_name: String,
    /// Embedding dimension.
    pub embedding_dim: u64,
    /// Whether to auto-create the collection if it doesn't exist.
    pub auto_create_collection: bool,
    /// HNSW M parameter (max connections per node, default 16).
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: u64,
    /// HNSW ef_construct parameter (search depth during build, default 100).
    #[serde(default = "default_hnsw_ef")]
    pub hnsw_ef_construct: u64,
    /// Whether to enable scalar quantization (reduces memory 4x).
    #[serde(default)]
    pub enable_quantization: bool,
    /// Whether to enable on-disk storage (reduces RAM usage).
    #[serde(default)]
    pub on_disk: bool,
    /// Whether to create payload indexes for `layer` and `importance`.
    #[serde(default = "default_true")]
    pub create_payload_indexes: bool,
}

fn default_hnsw_m() -> u64 { 16 }
fn default_hnsw_ef() -> u64 { 100 }
fn default_true() -> bool { true }

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".into()),
            collection_name: "qbitos_memories".into(),
            embedding_dim: 384,
            auto_create_collection: true,
            hnsw_m: 16,
            hnsw_ef_construct: 100,
            enable_quantization: false,
            on_disk: false,
            create_payload_indexes: true,
        }
    }
}

/// Qdrant backend.
pub struct QdrantBackend {
    client: Arc<Qdrant>,
    config: QdrantConfig,
}

impl QdrantBackend {
    /// Create a new backend, connecting to Qdrant and creating the collection
    /// if needed.
    pub async fn new(config: QdrantConfig) -> Result<Self, BackendError> {
        let client = Qdrant::from_url(&config.url)
            .build()
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let client = Arc::new(client);
        let backend = Self { client: Arc::clone(&client), config };
        if backend.config.auto_create_collection {
            backend.ensure_collection().await?;
        }
        Ok(backend)
    }

    async fn ensure_collection(&self) -> Result<(), BackendError> {
        let exists = self.client
            .collection_exists(CollectionExistsRequest {
                collection_name: self.config.collection_name.clone(),
            })
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        if !exists {
            self.client
                .create_collection(CreateCollection {
                    collection_name: self.config.collection_name.clone(),
                    vectors_config: Some(VectorsConfig {
                        config: Some(qdrant::vectors_config::Config::Params(VectorParams {
                            size: self.config.embedding_dim,
                            distance: Distance::Cosine as i32,
                            ..Default::default()
                        })),
                    }),
                    ..Default::default()
                })
                .await
                .map_err(|e| BackendError::Connection(e.to_string()))?;
        }
        Ok(())
    }

    fn to_point(object: &MemoryObject) -> Result<PointStruct, BackendError> {
        let payload_json: serde_json::Value = serde_json::to_value(object)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        let payload: std::collections::HashMap<String, qdrant::Value> =
            serde_json::from_value(payload_json)
                .map_err(|e| BackendError::Serialization(e.to_string()))?;
        let vectors = if object.embedding.data.is_empty() {
            None
        } else {
            Some(Vectors {
                vectors_options: Some(qdrant::vectors::VectorsOptions::Vector(
                    qdrant::Vector {
                        data: object.embedding.data.clone(),
                        ..Default::default()
                    },
                )),
            })
        };
        Ok(PointStruct {
            id: Some(PointId {
                point_id_options: Some(PointIdOptions::Uuid(object.id.0.to_string())),
            }),
            vectors,
            payload,
        })
    }

    fn from_point(point: &qdrant::RetrievedPoint) -> Result<MemoryObject, BackendError> {
        let payload_json: serde_json::Value = serde_json::to_value(&point.payload)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        let mut object: MemoryObject = serde_json::from_value(payload_json)
            .map_err(|e| BackendError::Serialization(e.to_string()))?;
        if let Some(vectors) = &point.vectors {
            if let Some(qdrant::vectors_output::VectorsOptions::Vector(vec)) =
                &vectors.vectors_options
            {
                object.embedding = Embedding { data: vec.data.clone() };
            }
        }
        Ok(object)
    }

    fn point_id(id: MemoryId) -> PointId {
        PointId {
            point_id_options: Some(PointIdOptions::Uuid(id.0.to_string())),
        }
    }

    fn with_payload_true() -> Option<WithPayloadSelector> {
        Some(WithPayloadSelector {
            selector_options: Some(with_payload_selector::SelectorOptions::Enable(true)),
        })
    }

    fn with_vectors_true() -> Option<WithVectorsSelector> {
        Some(WithVectorsSelector {
            selector_options: Some(with_vectors_selector::SelectorOptions::Enable(true)),
        })
    }
}

#[async_trait]
impl MemoryBackend for QdrantBackend {
    fn name(&self) -> &str { "qdrant" }
    fn supports_vector_search(&self) -> bool { true }
    fn is_durable(&self) -> bool { true }

    async fn store(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let point = Self::to_point(object)?;
        self.client
            .upsert_points(UpsertPoints {
                collection_name: self.config.collection_name.clone(),
                points: vec![point],
                ..Default::default()
            })
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn get(&self, id: MemoryId) -> Result<MemoryObject, BackendError> {
        let response = self.client
            .get_points(GetPoints {
                collection_name: self.config.collection_name.clone(),
                ids: vec![Self::point_id(id)],
                with_payload: Self::with_payload_true(),
                with_vectors: Self::with_vectors_true(),
                ..Default::default()
            })
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        let point = response.result.first()
            .ok_or(BackendError::NotFound(id))?;
        Self::from_point(point)
    }

    async fn delete(&self, id: MemoryId) -> Result<(), BackendError> {
        let selector = PointsSelector {
            points_selector_one_of: Some(PointsSelectorOneOf::Points(PointsIdsList {
                ids: vec![Self::point_id(id)],
            })),
        };
        self.client
            .delete_points(DeletePoints {
                collection_name: self.config.collection_name.clone(),
                points: Some(selector),
                ..Default::default()
            })
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(())
    }

    async fn count(&self) -> Result<usize, BackendError> {
        let response = self.client
            .count(CountPoints {
                collection_name: self.config.collection_name.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;
        Ok(response.result.map(|r| r.count as usize).unwrap_or(0))
    }

    async fn search(&self, query: &SimilarityQuery) -> Result<Vec<SimilarityResult>, BackendError> {
        if query.embedding.data.is_empty() {
            return Err(BackendError::Serialization("empty query embedding".into()));
        }
        let filter = query.layer_filter.as_ref().map(|layer| {
            let layer_str = format!("{:?}", layer).to_lowercase();
            qdrant::Filter {
                must: vec![qdrant::Condition {
                    condition_one_of: Some(qdrant::condition::ConditionOneOf::Field(
                        qdrant::FieldCondition {
                            key: "layer".into(),
                            r#match: Some(qdrant::Match {
                                match_value: Some(qdrant::r#match::MatchValue::Keyword(layer_str)),
                            }),
                            ..Default::default()
                        },
                    )),
                }],
                ..Default::default()
            }
        });

        let response = self.client
            .search_points(SearchPoints {
                collection_name: self.config.collection_name.clone(),
                vector: query.embedding.data.clone(),
                limit: query.k as u64,
                filter,
                with_payload: Self::with_payload_true(),
                with_vectors: Self::with_vectors_true(),
                score_threshold: Some(query.min_score),
                ..Default::default()
            })
            .await
            .map_err(|e| BackendError::Connection(e.to_string()))?;

        let mut results = Vec::new();
        for scored in response.result {
            if let Some(point_id) = scored.id {
                if let Ok(get_response) = self.client
                    .get_points(GetPoints {
                        collection_name: self.config.collection_name.clone(),
                        ids: vec![point_id],
                        with_payload: Self::with_payload_true(),
                        with_vectors: Self::with_vectors_true(),
                        ..Default::default()
                    })
                    .await
                {
                    if let Some(point) = get_response.result.first() {
                        if let Ok(object) = Self::from_point(point) {
                            results.push(SimilarityResult { object, score: scored.score });
                        }
                    }
                }
            }
        }
        Ok(results)
    }
}
