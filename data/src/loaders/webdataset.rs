//! WebDataset loader (requires `webdataset` feature).
//!
//! WebDataset stores samples in `.tar` shards, where each sample consists
//! of multiple files with a common basename (e.g. `001.jpg`, `001.json`,
//! `001.txt`). This loader streams samples from a shard.

use async_trait::async_trait;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::loader::{
    read_file_to_bytes, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// WebDataset loader.
pub struct WebDatasetLoader;

impl Default for WebDatasetLoader {
    fn default() -> Self {
        Self
    }
}

impl WebDatasetLoader {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for WebDatasetLoader {
    fn id(&self) -> &str { "webdataset" }
    fn name(&self) -> &str { "WebDataset (.tar shards)" }
    fn extensions(&self) -> &[&str] { &["tar", "tgz", "tar.gz"] }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let bytes = read_file_to_bytes(path).await?;
        let source = path.to_string_lossy().to_string();

        let cursor = std::io::Cursor::new(bytes);
        let mut archive = tar::Archive::new(cursor);

        // Group entries by basename
        let mut samples: BTreeMap<String, BTreeMap<String, Vec<u8>>> = BTreeMap::new();
        for entry in archive.entries()
            .map_err(|e| DatasetError::ParseError { format: "tar".into(), error: e.to_string() })?
        {
            let mut entry = entry.map_err(|e| DatasetError::ParseError { format: "tar".into(), error: e.to_string() })?;
            let path_str = entry.path()
                .map_err(|e| DatasetError::ParseError { format: "tar".into(), error: e.to_string() })?
                .to_string_lossy()
                .to_string();
            // Extract basename and extension
            let basename = path_str.rsplit_once('.').map(|(base, _)| base).unwrap_or(&path_str).to_string();
            let extension = path_str.rsplit_once('.').map(|(_, ext)| ext.to_string()).unwrap_or_default();
            let mut data = Vec::new();
            entry.read_to_end(&mut data).map_err(DatasetError::Io)?;
            samples.entry(basename).or_default().insert(extension, data);
        }

        let max = options.max_records.unwrap_or(usize::MAX);
        let offset = options.offset.unwrap_or(0);
        let mut records = Vec::new();
        for (i, (basename, files)) in samples.into_iter().enumerate() {
            if i < offset {
                continue;
            }
            if records.len() >= max {
                break;
            }
            let mut record = DatasetRecord::json(source.clone(), serde_json::json!({"basename": basename}));
            record.modality = crate::loader::Modality::Binary;
            // Attach the first file as bytes (typically the image)
            if let Some((ext, data)) = files.iter().next() {
                record.bytes = Some(data.clone());
                record.mime = mime_guess::from_ext(ext).first_or_octet_stream().to_string();
            }
            // Attach metadata for all extensions
            let exts: Vec<String> = files.keys().cloned().collect();
            record.metadata.insert("extensions".into(), serde_json::Value::Array(
                exts.into_iter().map(serde_json::Value::String).collect(),
            ));
            records.push(record);
        }

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}
