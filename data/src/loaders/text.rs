//! Plain-text loader: `.txt` and `.md`.
//!
//! Reads the file as UTF-8 and emits a single text record. Optionally,
//! for Markdown, splits the file into sections by `##` headings and emits
//! one record per section.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    read_file_to_string, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Plain-text loader for `.txt` and `.md` files.
pub struct TextLoader {
    split_markdown_sections: bool,
}

impl TextLoader {
    /// Create a new text loader.
    #[must_use]
    pub fn new() -> Self {
        Self { split_markdown_sections: true }
    }

    /// Disable Markdown section splitting.
    #[must_use]
    pub fn no_markdown_split(mut self) -> Self {
        self.split_markdown_sections = false;
        self
    }
}

impl Default for TextLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DatasetLoader for TextLoader {
    fn id(&self) -> &str { "text" }
    fn name(&self) -> &str { "Plain text (.txt, .md)" }
    fn extensions(&self) -> &[&str] { &["txt", "md", "markdown", "text"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let content = read_file_to_string(path).await?;
        let source = path.to_string_lossy().to_string();
        let mime = if path.extension().and_then(|e| e.to_str()) == Some("md") {
            "text/markdown"
        } else {
            "text/plain"
        };

        let records = if self.split_markdown_sections && mime == "text/markdown" {
            split_markdown_sections(&source, &content)
        } else {
            vec![DatasetRecord {
                id: uuid::Uuid::now_v7(),
                modality: crate::loader::Modality::Text,
                mime: mime.into(),
                source: source.clone(),
                content: serde_json::Value::String(content),
                bytes: None,
                embedding: None,
                metadata: std::collections::BTreeMap::new(),
                loaded_at: chrono::Utc::now(),
            }]
        };

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}

/// Split a Markdown document into sections by `##` (level-2) headings.
fn split_markdown_sections(source: &str, content: &str) -> Vec<DatasetRecord> {
    let mut records = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_body = String::new();

    for line in content.lines() {
        if line.starts_with("## ") {
            // Flush the previous section
            if let Some(heading) = current_heading.take() {
                records.push(DatasetRecord {
                    id: uuid::Uuid::now_v7(),
                    modality: crate::loader::Modality::Text,
                    mime: "text/markdown".into(),
                    source: source.to_string(),
                    content: serde_json::Value::String(format!("# {heading}\n\n{current_body}")),
                    bytes: None,
                    embedding: None,
                    metadata: std::collections::BTreeMap::from([(
                        "section".into(),
                        serde_json::Value::String(heading),
                    )]),
                    loaded_at: chrono::Utc::now(),
                });
                current_body.clear();
            }
            current_heading = Some(line.trim_start_matches("## ").trim().to_string());
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }
    if let Some(heading) = current_heading {
        records.push(DatasetRecord {
            id: uuid::Uuid::now_v7(),
            modality: crate::loader::Modality::Text,
            mime: "text/markdown".into(),
            source: source.to_string(),
            content: serde_json::Value::String(format!("# {heading}\n\n{current_body}")),
            bytes: None,
            embedding: None,
            metadata: std::collections::BTreeMap::from([(
                "section".into(),
                serde_json::Value::String(heading),
            )]),
            loaded_at: chrono::Utc::now(),
        });
    }
    if records.is_empty() {
        records.push(DatasetRecord::text(source, content));
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn load_text_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp.as_file(), "hello world").unwrap();
        let loader = TextLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let record = stream.next_record().await.unwrap().unwrap();
        assert_eq!(record.modality, crate::loader::Modality::Text);
        assert!(record.content.as_str().unwrap().contains("hello world"));
    }

    #[tokio::test]
    async fn load_markdown_splits_sections() {
        let tmp = tempfile::Builder::new().suffix(".md").tempfile().unwrap();
        writeln!(tmp.as_file(), "# Title\n\n## Section 1\n\nbody 1\n\n## Section 2\n\nbody 2").unwrap();
        let loader = TextLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let mut count = 0;
        while stream.next_record().await.unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 2);
    }
}
