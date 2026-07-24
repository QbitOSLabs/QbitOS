//! HTML loader (requires `html` feature).
//!
//! Extracts text content from HTML files. Strips tags and emits one record
//! per page, with metadata for `<title>` and `<meta>`.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    read_file_to_string, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// HTML loader.
pub struct HtmlLoader;

impl Default for HtmlLoader {
    fn default() -> Self {
        Self
    }
}

impl HtmlLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for HtmlLoader {
    fn id(&self) -> &str { "html" }
    fn name(&self) -> &str { "HTML (.html, .htm)" }
    fn extensions(&self) -> &[&str] { &["html", "htm", "xhtml"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let content = read_file_to_string(path).await?;
        let source = path.to_string_lossy().to_string();

        let document = scraper::Html::parse_document(&content);
        let title_selector = scraper::Selector::parse("title").unwrap();
        let title = document.select(&title_selector)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        // Extract main text content
        let body_selector = scraper::Selector::parse("body").unwrap();
        let body_text = document.select(&body_selector)
            .next()
            .map(|e| e.text().collect::<String>())
            .unwrap_or_default();

        let mut record = DatasetRecord::text(source, body_text);
        record.mime = "text/html".into();
        record.metadata.insert("title".into(), serde_json::Value::String(title));
        Ok(Box::new(MemoryRecordStream::new(vec![record])))
    }
}
