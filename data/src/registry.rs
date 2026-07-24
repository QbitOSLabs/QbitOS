//! The dataset registry.
//!
//! Holds a collection of [`DatasetLoader`]s and dispatches based on file
//! extension. Custom loaders can be registered at runtime.

use ahash::AHashMap;
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

use crate::loader::SharedLoader;

/// Errors emitted by the registry.
#[derive(Debug, Error)]
pub enum RegistryError {
    /// No loader matches the given path.
    #[error("no loader registered for path: {0}")]
    NoLoader(String),
    /// A loader with the same id is already registered.
    #[error("loader `{0}` is already registered")]
    Duplicate(String),
}

/// The dataset registry.
#[derive(Clone)]
pub struct DatasetRegistry {
    loaders: Arc<RwLock<AHashMap<String, SharedLoader>>>,
}

impl DatasetRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loaders: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    /// Create a registry pre-populated with all default-enabled loaders.
    #[must_use]
    pub fn with_default_loaders() -> Self {
        let registry = Self::new();
        registry.register_defaults();
        registry
    }

    /// Register all default-enabled loaders (controlled by feature flags).
    pub fn register_defaults(&self) {
        // Always-on (default features)
        #[cfg(feature = "text")]
        self.register(Arc::new(crate::loaders::text::TextLoader::new()));
        #[cfg(feature = "json")]
        self.register(Arc::new(crate::loaders::json::JsonLoader::new()));
        #[cfg(feature = "jsonl")]
        self.register(Arc::new(crate::loaders::jsonl::JsonlLoader::new()));
        self.register(Arc::new(crate::loaders::csv::CsvLoader::new()));
        #[cfg(feature = "yaml")]
        self.register(Arc::new(crate::loaders::yaml::YamlLoader::new()));
        #[cfg(feature = "xml")]
        self.register(Arc::new(crate::loaders::xml::XmlLoader::new()));
        #[cfg(feature = "source-code")]
        self.register(Arc::new(crate::loaders::source_code::SourceCodeLoader::new()));
        #[cfg(feature = "conversation")]
        self.register(Arc::new(crate::loaders::conversation::ConversationLoader::new()));
        #[cfg(feature = "instruction")]
        self.register(Arc::new(crate::loaders::instruction::InstructionLoader::new()));
        #[cfg(feature = "function-call")]
        self.register(Arc::new(crate::loaders::function_call::FunctionCallLoader::new()));
        #[cfg(feature = "agent-trajectory")]
        self.register(Arc::new(crate::loaders::agent_trajectory::AgentTrajectoryLoader::new()));
        #[cfg(feature = "preference")]
        self.register(Arc::new(crate::loaders::preference::PreferenceLoader::new()));
        #[cfg(feature = "knowledge-graph")]
        self.register(Arc::new(crate::loaders::knowledge_graph::KnowledgeGraphLoader::new()));
        #[cfg(feature = "vector")]
        self.register(Arc::new(crate::loaders::vector::VectorDatasetLoader::new()));
        #[cfg(feature = "time-series")]
        self.register(Arc::new(crate::loaders::time_series::TimeSeriesLoader::new()));
        #[cfg(feature = "tabular")]
        self.register(Arc::new(crate::loaders::tabular::TabularLoader::new()));

        // Optional format features
        #[cfg(feature = "parquet")]
        self.register(Arc::new(crate::loaders::parquet::ParquetLoader::new()));
        #[cfg(feature = "arrow")]
        self.register(Arc::new(crate::loaders::arrow::ArrowLoader::new()));
        #[cfg(feature = "sqlite")]
        self.register(Arc::new(crate::loaders::sqlite::SqliteLoader::new()));
        #[cfg(feature = "excel")]
        self.register(Arc::new(crate::loaders::excel::ExcelLoader::new()));
        #[cfg(feature = "pdf")]
        self.register(Arc::new(crate::loaders::pdf::PdfLoader::new()));
        #[cfg(feature = "html")]
        self.register(Arc::new(crate::loaders::html::HtmlLoader::new()));
        #[cfg(feature = "git")]
        self.register(Arc::new(crate::loaders::git_repo::GitRepoLoader::new()));
        #[cfg(feature = "image-loader")]
        self.register(Arc::new(crate::loaders::image_loader::ImageLoader::new()));
        #[cfg(feature = "audio-loader")]
        self.register(Arc::new(crate::loaders::audio_loader::AudioLoader::new()));
        #[cfg(feature = "video-loader")]
        self.register(Arc::new(crate::loaders::video_loader::VideoLoader::new()));
        #[cfg(feature = "ocr-loader")]
        self.register(Arc::new(crate::loaders::ocr_loader::OcrLoader::new()));
        #[cfg(feature = "lmdb-loader")]
        self.register(Arc::new(crate::loaders::lmdb_loader::LmdbLoader::new()));
        #[cfg(feature = "hdf5-loader")]
        self.register(Arc::new(crate::loaders::hdf5_loader::Hdf5Loader::new()));
        #[cfg(feature = "webdataset")]
        self.register(Arc::new(crate::loaders::webdataset::WebDatasetLoader::new()));
        #[cfg(feature = "huggingface")]
        self.register(Arc::new(crate::loaders::huggingface::HuggingFaceLoader::new()));
        #[cfg(feature = "object-storage")]
        self.register(Arc::new(crate::loaders::object_storage::ObjectStorageLoader::new()));
    }

    /// Register a loader. Overwrites any existing loader with the same id.
    pub fn register(&self, loader: SharedLoader) {
        let id = loader.id().to_string();
        self.loaders.write().insert(id, loader);
    }

    /// Look up a loader by id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<SharedLoader> {
        self.loaders.read().get(id).cloned()
    }

    /// Select a loader for the given path. Iterates registered loaders and
    /// returns the most specific match.
    ///
    /// Specificity is determined by the number of extensions the loader
    /// handles (fewer = more specific) — this lets conversation/instruction/
    /// agent-trajectory loaders win over the generic JSONL loader when
    /// the filename matches.
    ///
    /// # Errors
    /// Returns [`RegistryError::NoLoader`] if no loader matches.
    pub fn select(&self, path: &Path) -> Result<SharedLoader, RegistryError> {
        let loaders = self.loaders.read();
        let mut candidates: Vec<SharedLoader> = loaders.values()
            .filter(|l| l.matches(path))
            .cloned()
            .collect();
        // Sort by specificity: loaders with fewer extensions (more specific)
        // come first. Loaders whose id is one of the "specific" ids always win.
        let priority = |id: &str| -> u32 {
            match id {
                "conversation" | "instruction" | "function-call" | "agent-trajectory"
                | "preference" | "knowledge-graph" | "vector" | "time-series"
                | "tabular" | "source-code" | "git" | "huggingface" | "object-storage"
                | "image" | "audio" | "video" | "ocr" | "pdf" | "html" | "excel"
                | "sqlite" | "parquet" | "arrow" | "lmdb" | "hdf5" | "webdataset" => 0,
                _ => 1,
            }
        };
        candidates.sort_by_key(|l| priority(l.id()));
        candidates.into_iter().next().ok_or_else(|| RegistryError::NoLoader(path.display().to_string()))
    }

    /// List all registered loaders.
    #[must_use]
    pub fn list(&self) -> Vec<SharedLoader> {
        self.loaders.read().values().cloned().collect()
    }

    /// Number of registered loaders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.loaders.read().len()
    }

    /// Returns `true` if no loaders are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.loaders.read().is_empty()
    }
}

impl Default for DatasetRegistry {
    fn default() -> Self {
        Self::with_default_loaders()
    }
}
