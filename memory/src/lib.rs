//! QbitOS Memory Fabric
//!
//! Biological-inspired, multi-layer memory system. Each layer has distinct
//! retention characteristics and storage backends:
//!
//! | Layer | Purpose | Backend | Retention |
//! |-------|---------|---------|-----------|
//! | L0 Sensory | Raw perception (frames, audio) | In-memory ring buffer | <1s |
//! | L1 Working | Current task, active reasoning | In-memory | <5 min |
//! | L2 Episodic | Experiences, outcomes | RocksDB + embeddings | days–weeks |
//! | L3 Semantic | Facts, concepts, relationships | Postgres + pgvector | permanent |
//! | L4 Procedural | Skills, workflows | Skill genome (qbitos-skills) | permanent |
//! | L5 Meta | Learning strategies, confidence | Postgres | permanent |
//!
//! Every memory object carries the same envelope (see [`MemoryObject`]) so
//! consolidation can move data between layers without re-encoding.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod advanced_search;
pub mod backends;
pub mod consolidation;
pub mod embedding;
pub mod episodic;
pub mod hnsw;
pub mod meta;
pub mod object;
pub mod procedural;
pub mod semantic;
pub mod sensory;
pub mod streaming_hnsw;
pub mod tenant;
pub mod working;

pub use backends::{BackendError, MemoryBackend, SharedBackend, SimilarityQuery, SimilarityResult};
#[cfg(feature = "postgres-backed")]
pub use backends::{PostgresBackend, postgres::PostgresConfig};
#[cfg(feature = "qdrant-backed")]
pub use backends::{QdrantBackend, qdrant::QdrantConfig};
#[cfg(feature = "redis-backed")]
pub use backends::{RedisBackend, redis::RedisConfig};
#[cfg(feature = "rocksdb-backed")]
pub use backends::{RocksdbBackend, rocksdb::RocksdbConfig};
#[cfg(feature = "mongodb-backed")]
pub use backends::{MongoBackend, mongo::MongoConfig};
#[cfg(feature = "duckdb-backed")]
pub use backends::{DuckdbBackend, duckdb::DuckdbConfig};
pub use consolidation::ConsolidationEngine;
pub use embedding::{Embedding, EmbeddingModel, EmbeddingModelStub, SharedEmbeddingModel};
pub use episodic::EpisodicMemory;
pub use hnsw::HnswIndex;
pub use meta::MetaMemory;
pub use object::{Importance, MemoryId, MemoryObject, MemoryRelationship, MemoryType};
pub use procedural::ProceduralMemory;
pub use semantic::SemanticMemory;
pub use sensory::{SensoryMemory, SensorySlot};
pub use streaming_hnsw::{SharedStreamingHnsw, StreamingHnsw, StreamingHnswConfig, StreamingHnswError, VectorId};
pub use tenant::{
    TenantAction, TenantConfig, TenantContext, TenantId,
    TenantRegistry, SharedTenantRegistry,
};
pub use working::WorkingMemory;

/// Re-export of the layers as a single composite fabric.
pub use fabric::MemoryFabric;

mod fabric;
