//! Hugging Face Hub loader (requires `huggingface` feature).
//!
//! Downloads and loads datasets from the Hugging Face Hub. Uses the `hf-hub`
//! crate to resolve `dataset:repo` paths.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// HuggingFace loader.
pub struct HuggingFaceLoader {
    cache_dir: PathBuf,
}

impl Default for HuggingFaceLoader {
    fn default() -> Self {
        let cache_dir = std::env::var("HF_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/hf-cache"));
        Self { cache_dir }
    }
}

impl HuggingFaceLoader {
    /// Create a new loader with a custom cache directory.
    #[must_use]
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }
}

#[async_trait]
impl DatasetLoader for HuggingFaceLoader {
    fn id(&self) -> &str { "huggingface" }
    fn name(&self) -> &str { "Hugging Face Hub (datasets:repo)" }
    fn extensions(&self) -> &[&str] { &["hf"] }

    fn matches(&self, path: &Path) -> bool {
        path.to_string_lossy().starts_with("hf:")
            || path.to_string_lossy().starts_with("datasets:")
    }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let path_str = path.to_string_lossy().to_string();
        let repo = path_str.strip_prefix("hf:").or_else(|| path_str.strip_prefix("datasets:")).unwrap();
        let (repo_id, revision) = match repo.split_once('@') {
            Some((r, v)) => (r.to_string(), v.to_string()),
            None => (repo.to_string(), "main".to_string()),
        };
        let (repo_id, file) = match repo_id.split_once(':') {
            Some((r, f)) => (r.to_string(), f.to_string()),
            None => (repo_id, "data/train-00000-of-00001.parquet".to_string()),
        };

        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_cache_dir(self.cache_dir.clone())
            .build()
            .map_err(|e| DatasetError::Custom(format!("hf-hub error: {e}")))?;
        let local_path = api.dataset(repo_id.clone()).get(&file).await
            .map_err(|e| DatasetError::Custom(format!("hf-hub download error: {e}")))?;

        // Delegate to the appropriate loader based on file extension
        let registry = crate::registry::DatasetRegistry::with_default_loaders();
        let loader = registry.select(&local_path)?;
        let mut stream = loader.load_with_options(&local_path, options).await?;
        // Post-process: attach HF metadata to each record
        let mut records = Vec::new();
        while let Some(mut record) = stream.next_record().await? {
            record.metadata.insert("hf_repo".into(), serde_json::Value::String(repo_id.clone()));
            record.metadata.insert("hf_revision".into(), serde_json::Value::String(revision.clone()));
            record.metadata.insert("hf_file".into(), serde_json::Value::String(file.clone()));
            record.source = path_str.clone();
            records.push(record);
        }
        Ok(Box::new(crate::streaming::MemoryRecordStream::new(records)))
    }
}
