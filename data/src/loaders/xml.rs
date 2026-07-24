//! XML loader: `.xml`.
//!
//! Reads an XML file and converts it to JSON via `quick-xml`. Each top-level
//! repeating element becomes a record.

use async_trait::async_trait;
use quick_xml::de::from_str;
use std::path::Path;

use crate::loader::{
    read_file_to_string, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// XML loader.
pub struct XmlLoader;

impl Default for XmlLoader {
    fn default() -> Self {
        Self
    }
}

impl XmlLoader {
    /// Create a new XML loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for XmlLoader {
    fn id(&self) -> &str { "xml" }
    fn name(&self) -> &str { "XML (.xml)" }
    fn extensions(&self) -> &[&str] { &["xml", "rss", "atom"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let content = read_file_to_string(path).await?;
        let value: serde_json::Value = from_str(&content)
            .map_err(|e| DatasetError::ParseError { format: "xml".into(), error: e.to_string() })?;
        let source = path.to_string_lossy().to_string();

        // Try to find a top-level array field (common pattern: <items><item>...</item></items>)
        let records = flatten_xml_array(&source, value);
        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}

/// If the JSON-converted XML has a single top-level object whose only field
/// is an array, treat that array as the record list. Otherwise emit the
/// whole document as one record.
fn flatten_xml_array(source: &str, value: serde_json::Value) -> Vec<DatasetRecord> {
    match value {
        serde_json::Value::Array(arr) => {
            arr.into_iter().map(|v| DatasetRecord::json(source.to_string(), v)).collect()
        }
        serde_json::Value::Object(map) => {
            // Look for the first array value
            for (_, v) in map.clone() {
                if let serde_json::Value::Array(arr) = v {
                    return arr.into_iter()
                        .map(|item| DatasetRecord::json(source.to_string(), item))
                        .collect();
                }
            }
            vec![DatasetRecord::json(source.to_string(), serde_json::Value::Object(map))]
        }
        other => vec![DatasetRecord::json(source.to_string(), other)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_xml() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), r#"<?xml version="1.0"?>
<items>
  <item><name>Alice</name><age>30</age></item>
  <item><name>Bob</name><age>25</age></item>
</items>"#).unwrap();
        let loader = XmlLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        // The XML has an items wrapper containing an item array
        assert!(r.content.is_object());
    }
}
