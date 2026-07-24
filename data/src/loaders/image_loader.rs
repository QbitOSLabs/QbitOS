//! Image loader (requires `image-loader` feature).
//!
//! Loads images and emits one record per file, with the raw bytes attached.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    read_file_to_bytes, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Image loader.
pub struct ImageLoader;

impl Default for ImageLoader {
    fn default() -> Self {
        Self
    }
}

impl ImageLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for ImageLoader {
    fn id(&self) -> &str { "image" }
    fn name(&self) -> &str { "Image (.jpg, .png, .webp, .gif, .bmp)" }
    fn extensions(&self) -> &[&str] { &["jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "tif", "ico"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let bytes = read_file_to_bytes(path).await?;
        let source = path.to_string_lossy().to_string();
        let mime = mime_guess::from_path(path).first_or_octet_stream().to_string();

        // Decode to get dimensions and attach as metadata
        let dims = image::load_from_memory(&bytes).ok().map(|img| {
            (img.width(), img.height())
        });

        let mut record = DatasetRecord::binary(source, mime, bytes);
        record.modality = crate::loader::Modality::Image;
        if let Some((w, h)) = dims {
            record.metadata.insert("width".into(), serde_json::Value::Number(serde_json::Number::from(w)));
            record.metadata.insert("height".into(), serde_json::Value::Number(serde_json::Number::from(h)));
        }
        Ok(Box::new(MemoryRecordStream::new(vec![record])))
    }
}
