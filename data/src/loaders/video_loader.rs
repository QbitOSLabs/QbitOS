//! Video loader (requires `video-loader` feature).
//!
//! Loads video files and emits one record per file with raw bytes. Does not
//! decode frames — that would require ffmpeg. Use the `qbitos-vision` crate
//! for frame extraction.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    read_file_to_bytes, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Video loader.
pub struct VideoLoader;

impl Default for VideoLoader {
    fn default() -> Self {
        Self
    }
}

impl VideoLoader {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for VideoLoader {
    fn id(&self) -> &str { "video" }
    fn name(&self) -> &str { "Video (.mp4, .avi, .mkv, .mov, .webm)" }
    fn extensions(&self) -> &[&str] { &["mp4", "avi", "mkv", "mov", "webm", "flv", "wmv", "m4v"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let bytes = read_file_to_bytes(path).await?;
        let source = path.to_string_lossy().to_string();
        let mime = mime_guess::from_path(path).first_or_octet_stream().to_string();

        let mut record = DatasetRecord::binary(source, mime, bytes);
        record.modality = crate::loader::Modality::Video;
        Ok(Box::new(MemoryRecordStream::new(vec![record])))
    }
}
