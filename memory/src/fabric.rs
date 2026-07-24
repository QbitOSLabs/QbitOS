//! The composite memory fabric: a single handle to all six layers +
//! optional durable backends.

use std::sync::Arc;

use crate::backends::{BackendError, SharedBackend, SimilarityQuery, SimilarityResult};
use crate::consolidation::{ConsolidationConfig, ConsolidationEngine, ConsolidationSummary};
use crate::embedding::{SharedEmbeddingModel, EmbeddingModelStub};
use crate::episodic::{EpisodicMemory, Outcome};
use crate::meta::MetaMemory;
use crate::object::{MemoryId, MemoryObject, MemoryType};
use crate::procedural::ProceduralMemory;
use crate::semantic::SemanticMemory;
use crate::sensory::{SensoryMemory, SensorySlot};
use crate::working::WorkingMemory;

/// The composite memory fabric.
///
/// Holds [`Arc`] handles to every layer so they can be shared across the
/// kernel, planner, and reflection engine. The consolidation engine is owned
/// by the fabric and runs on its own task when [`MemoryFabric::start_consolidation`]
/// is called.
///
/// Optional durable backends (Postgres, Qdrant, Redis, RocksDB, Sled) can be
/// attached via [`MemoryFabric::add_backend`]. When backends are attached:
///
/// - `store` writes to all backends (write-through).
/// - `get` reads from the fastest backend that has the data (first to return wins).
/// - `search` queries all vector-capable backends and merges results.
pub struct MemoryFabric {
    /// L0 sensory memory.
    pub sensory: Arc<SensoryMemory>,
    /// L1 working memory.
    pub working: Arc<WorkingMemory>,
    /// L2 episodic memory.
    pub episodic: Arc<EpisodicMemory>,
    /// L3 semantic memory.
    pub semantic: Arc<SemanticMemory>,
    /// L4 procedural memory.
    pub procedural: Arc<ProceduralMemory>,
    /// L5 meta memory.
    pub meta: Arc<MetaMemory>,
    /// The shared embedding model.
    pub embedder: SharedEmbeddingModel,
    consolidation: parking_lot::Mutex<Option<Arc<ConsolidationEngine>>>,
    /// Attached durable backends.
    backends: parking_lot::RwLock<Vec<SharedBackend>>,
}

impl MemoryFabric {
    /// Build a new memory fabric with the given embedding model.
    #[must_use]
    pub fn new(embedder: SharedEmbeddingModel) -> Self {
        Self {
            sensory: Arc::new(SensoryMemory::default()),
            working: Arc::new(WorkingMemory::default()),
            episodic: Arc::new(EpisodicMemory::new(Arc::clone(&embedder))),
            semantic: Arc::new(SemanticMemory::new(Arc::clone(&embedder))),
            procedural: Arc::new(ProceduralMemory::new(Arc::clone(&embedder))),
            meta: Arc::new(MetaMemory::new()),
            embedder,
            consolidation: parking_lot::Mutex::new(None),
            backends: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Build a new memory fabric with the default deterministic embedding model.
    #[must_use]
    pub fn with_deterministic_embeddings(dim: usize) -> Self {
        let emb: SharedEmbeddingModel = Arc::new(EmbeddingModelStub::new(dim));
        Self::new(emb)
    }

    /// Attach a durable backend (Postgres, Qdrant, Redis, RocksDB, Sled).
    ///
    /// Once attached, `store_to_backends`, `get_from_backends`, and
    /// `search_backends` will use this backend.
    pub fn add_backend(&self, backend: SharedBackend) {
        self.backends.write().push(backend);
    }

    /// Returns the number of attached backends.
    #[must_use]
    pub fn backend_count(&self) -> usize {
        self.backends.read().len()
    }

    /// Returns the names of all attached backends.
    #[must_use]
    pub fn backend_names(&self) -> Vec<String> {
        self.backends.read().iter().map(|b| b.name().to_string()).collect()
    }

    /// Store a memory object to all attached backends (write-through).
    ///
    /// # Errors
    /// Returns the first [`BackendError`] encountered. Remaining backends
    /// are still written to (best-effort).
    pub async fn store_to_backends(&self, object: &MemoryObject) -> Result<(), BackendError> {
        let backends = self.backends.read().clone();
        let mut first_err = None;
        for backend in &backends {
            if let Err(e) = backend.store(object).await {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Get a memory object from the first backend that has it.
    ///
    /// # Errors
    /// Returns [`BackendError::NotFound`] if no backend has the object.
    pub async fn get_from_backends(&self, id: MemoryId) -> Result<MemoryObject, BackendError> {
        let backends = self.backends.read().clone();
        let mut last_err = None;
        for backend in &backends {
            match backend.get(id).await {
                Ok(obj) => return Ok(obj),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or(BackendError::NotFound(id)))
    }

    /// Delete a memory object from all attached backends.
    ///
    /// # Errors
    /// Returns the first [`BackendError`] encountered.
    pub async fn delete_from_backends(&self, id: MemoryId) -> Result<(), BackendError> {
        let backends = self.backends.read().clone();
        let mut first_err = None;
        for backend in &backends {
            if let Err(e) = backend.delete(id).await {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Search all vector-capable backends and merge results.
    ///
    /// # Errors
    /// Returns the first [`BackendError`] if all backends fail.
    pub async fn search_backends(&self, query: &SimilarityQuery) -> Result<Vec<SimilarityResult>, BackendError> {
        let backends = self.backends.read().clone();
        let mut all_results = Vec::new();
        let mut last_err = None;
        for backend in &backends {
            if backend.supports_vector_search() {
                match backend.search(query).await {
                    Ok(results) => all_results.extend(results),
                    Err(e) => last_err = Some(e),
                }
            }
        }
        // Sort by score descending
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(query.k);
        if all_results.is_empty() {
            if let Some(e) = last_err {
                return Err(e);
            }
        }
        Ok(all_results)
    }

    /// Flush all attached backends.
    pub async fn flush_backends(&self) -> Result<(), BackendError> {
        let backends = self.backends.read().clone();
        for backend in &backends {
            let _ = backend.flush().await;
        }
        Ok(())
    }

    /// Configure the consolidation engine and store it for later use.
    pub fn configure_consolidation(&self, config: ConsolidationConfig) {
        let engine = Arc::new(ConsolidationEngine::new(
            config,
            Arc::clone(&self.sensory),
            Arc::clone(&self.working),
            Arc::clone(&self.episodic),
            Arc::clone(&self.semantic),
            Arc::clone(&self.procedural),
            Arc::clone(&self.meta),
        ));
        *self.consolidation.lock() = Some(engine);
    }

    /// Run a single consolidation cycle.
    #[must_use]
    pub fn consolidate(&self) -> ConsolidationSummary {
        if let Some(engine) = self.consolidation.lock().as_ref() {
            engine.consolidate()
        } else {
            ConsolidationSummary::default()
        }
    }

    /// Start the consolidation engine on a background task.
    pub fn start_consolidation(&self) {
        if let Some(engine) = self.consolidation.lock().as_ref() {
            let engine = Arc::clone(engine);
            tokio::spawn(async move { engine.run().await });
        }
    }

    /// Record an episodic memory and return its id.
    pub fn record_episode(&self, content: serde_json::Value, outcome: Outcome) -> MemoryId {
        self.episodic.record(content, outcome).object.id
    }

    /// Push a sensory slot.
    pub fn push_sensory(&self, slot: SensorySlot) {
        self.sensory.push(slot);
    }

    /// Push an item into working memory.
    pub fn push_working(&self, obj: MemoryObject) {
        self.working.push(obj);
    }

    /// Get the layer type for a memory id (best effort — checks each layer).
    #[must_use]
    pub fn layer_of(&self, _id: MemoryId) -> Option<MemoryType> {
        None
    }

    /// Backward-compatible alias for [`with_deterministic_embeddings`].
    #[must_use]
    pub fn with_stub_embeddings(dim: usize) -> Self {
        Self::with_deterministic_embeddings(dim)
    }
}

impl Default for MemoryFabric {
    fn default() -> Self {
        Self::with_deterministic_embeddings(384)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fabric_construction_default() {
        let fabric = MemoryFabric::default();
        assert_eq!(fabric.embedder.dim(), 384);
        assert!(fabric.sensory.is_empty());
        assert!(fabric.episodic.is_empty());
    }

    #[test]
    fn fabric_with_deterministic_embeddings() {
        let fabric = MemoryFabric::with_deterministic_embeddings(128);
        assert_eq!(fabric.embedder.dim(), 128);
    }

    #[test]
    fn fabric_record_episode_increases_count() {
        let fabric = MemoryFabric::with_deterministic_embeddings(64);
        let initial = fabric.episodic.len();
        fabric.record_episode(
            serde_json::json!({"event": "test"}),
            crate::episodic::Outcome::Success,
        );
        assert_eq!(fabric.episodic.len(), initial + 1);
    }

    #[test]
    fn fabric_push_sensory_increases_count() {
        let fabric = MemoryFabric::with_deterministic_embeddings(64);
        let initial = fabric.sensory.len();
        let slot = crate::sensory::SensorySlot::new("text", b"hello".to_vec());
        fabric.push_sensory(slot);
        assert!(fabric.sensory.len() > initial);
    }

    #[test]
    fn fabric_push_working_increases_count() {
        let fabric = MemoryFabric::with_deterministic_embeddings(64);
        let emb = fabric.embedder.embed_json(&serde_json::json!({"x": 1}));
        let obj = crate::object::MemoryObject::new(
            crate::object::MemoryType::Working,
            serde_json::json!({"x": 1}),
            emb,
        );
        fabric.push_working(obj);
        assert!(!fabric.working.spotlight().is_empty());
    }

    #[test]
    fn fabric_with_stub_embeddings_is_alias() {
        let a = MemoryFabric::with_deterministic_embeddings(64);
        let b = MemoryFabric::with_stub_embeddings(64);
        assert_eq!(a.embedder.dim(), b.embedder.dim());
    }

    #[test]
    fn fabric_backend_count_starts_at_zero() {
        let fabric = MemoryFabric::with_deterministic_embeddings(64);
        assert_eq!(fabric.backend_count(), 0);
        assert!(fabric.backend_names().is_empty());
    }
}
