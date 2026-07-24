//! JSON loader: `.json`.
//!
//! Reads a single JSON file. If the top-level value is an array, each
//! element becomes a record; otherwise the whole document is one record.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    read_file_to_string, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// JSON loader for `.json` files.
pub struct JsonLoader;

impl Default for JsonLoader {
    fn default() -> Self {
        Self
    }
}

impl JsonLoader {
    /// Create a new JSON loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for JsonLoader {
    fn id(&self) -> &str { "json" }
    fn name(&self) -> &str { "JSON (.json)" }
    fn extensions(&self) -> &[&str] { &["json"] }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let content = read_file_to_string(path).await?;
        let value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| DatasetError::ParseError { format: "json".into(), error: e.to_string() })?;
        let source = path.to_string_lossy().to_string();

        let records: Vec<DatasetRecord> = match value {
            serde_json::Value::Array(arr) => {
                arr.into_iter()
                    .skip(options.offset.unwrap_or(0))
                    .take(options.max_records.unwrap_or(usize::MAX))
                    .map(|v| DatasetRecord::json(source.clone(), v))
                    .collect()
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
    async fn load_json_array() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), r#"[{"a":1},{"a":2},{"a":3}]"#).unwrap();
        let loader = JsonLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let mut count = 0;
        while stream.next_record().await.unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn load_json_object() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), r#"{"a":1,"b":2}"#).unwrap();
        let loader = JsonLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.content["a"], 1);
    }
}
