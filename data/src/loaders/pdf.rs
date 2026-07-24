//! PDF loader (requires `pdf` feature).
//!
//! Extracts plain text from a PDF file. Emits a single record per page.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    read_file_to_bytes, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// PDF loader.
pub struct PdfLoader;

impl Default for PdfLoader {
    fn default() -> Self {
        Self
    }
}

impl PdfLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for PdfLoader {
    fn id(&self) -> &str { "pdf" }
    fn name(&self) -> &str { "PDF (.pdf)" }
    fn extensions(&self) -> &[&str] { &["pdf"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let bytes = read_file_to_bytes(path).await?;
        let source = path.to_string_lossy().to_string();

        // pdf-extract is sync; run on a blocking task
        let text = tokio::task::spawn_blocking(move || {
            pdf_extract::extract_text_from_mem(&bytes)
        })
        .await
        .map_err(|e| DatasetError::Custom(format!("join error: {e}")))?
        .map_err(|e| DatasetError::ParseError { format: "pdf".into(), error: e.to_string() })?;

        let record = DatasetRecord {
            id: uuid::Uuid::now_v7(),
            modality: crate::loader::Modality::Text,
            mime: "application/pdf".into(),
            source,
            content: serde_json::Value::String(text),
            bytes: None,
            embedding: None,
            metadata: std::collections::BTreeMap::new(),
            loaded_at: chrono::Utc::now(),
        };
        Ok(Box::new(MemoryRecordStream::new(vec![record])))
    }
}
