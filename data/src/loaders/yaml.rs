//! YAML loader: `.yaml` / `.yml`.
//!
//! Reads a YAML file. Like JSON, if the top-level value is a sequence,
//! each item becomes a record; otherwise the whole document is one record.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    read_file_to_string, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// YAML loader.
pub struct YamlLoader;

impl Default for YamlLoader {
    fn default() -> Self {
        Self
    }
}

impl YamlLoader {
    /// Create a new YAML loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for YamlLoader {
    fn id(&self) -> &str { "yaml" }
    fn name(&self) -> &str { "YAML (.yaml, .yml)" }
    fn extensions(&self) -> &[&str] { &["yaml", "yml"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let content = read_file_to_string(path).await?;
        let value: serde_json::Value = serde_yaml::from_str(&content)
            .map_err(|e| DatasetError::ParseError { format: "yaml".into(), error: e.to_string() })?;
        let source = path.to_string_lossy().to_string();

        let records: Vec<DatasetRecord> = match value {
            serde_json::Value::Array(arr) => {
                arr.into_iter().map(|v| DatasetRecord::json(source.clone(), v)).collect()
            }
            other => vec![DatasetRecord::json(source, other)],
        };

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_yaml() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "name: Alice\nage: 30\n").unwrap();
        let loader = YamlLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.content["name"], "Alice");
        assert_eq!(r.content["age"], 30);
    }
}
