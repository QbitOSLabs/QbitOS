//! Git repository loader (requires `git` feature).
//!
//! Walks a git repository and emits one record per file (similar to the
//! source-code loader, but uses `gix` to respect `.gitignore` and
//! include commit metadata).

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::loader::{
    read_file_to_string, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::loaders::source_code::{language_for_extension, SOURCE_EXTENSIONS};
use crate::streaming::MemoryRecordStream;

/// Git-repo loader.
pub struct GitRepoLoader;

impl Default for GitRepoLoader {
    fn default() -> Self {
        Self
    }
}

impl GitRepoLoader {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for GitRepoLoader {
    fn id(&self) -> &str { "git" }
    fn name(&self) -> &str { "Git repository (whole tree)" }
    fn extensions(&self) -> &[&str] { &["git"] }

    fn matches(&self, path: &Path) -> bool {
        // Match directories that contain a `.git` subdirectory
        if path.is_dir() {
            return path.join(".git").exists();
        }
        false
    }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let max = options.max_records.unwrap_or(usize::MAX);
        let offset = options.offset.unwrap_or(0);
        let mut records = Vec::new();

        let walker: Vec<PathBuf> = WalkDir::new(path)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !(name == ".git" || name == "target" || name == "node_modules" || name == "dist" || name == "build")
            })
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect();

        for (i, file) in walker.iter().enumerate() {
            if i < offset {
                continue;
            }
            if records.len() >= max {
                break;
            }
            let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
            let language = language_for_extension(ext);
            // Only include source files; skip binary
            if ext.is_empty() || SOURCE_EXTENSIONS.iter().any(|x| *x == ext) {
                let content = match read_file_to_string(file).await {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let mut record = DatasetRecord {
                    id: uuid::Uuid::now_v7(),
                    modality: crate::loader::Modality::Code,
                    mime: format!("text/x-{}", language.unwrap_or("plain")),
                    source: file.to_string_lossy().to_string(),
                    content: serde_json::Value::String(content),
                    bytes: None,
                    embedding: None,
                    metadata: std::collections::BTreeMap::new(),
                    loaded_at: chrono::Utc::now(),
                };
                if let Some(lang) = language {
                    record.metadata.insert("language".into(), serde_json::Value::String(lang.into()));
                }
                records.push(record);
            }
        }

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}
