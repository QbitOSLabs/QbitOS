//! Arrow IPC loader (requires `arrow` feature).

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Arrow loader.
pub struct ArrowLoader;

impl Default for ArrowLoader {
    fn default() -> Self { Self }
}

impl ArrowLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self { Self }
}

#[async_trait]
impl DatasetLoader for ArrowLoader {
    fn id(&self) -> &str { "arrow" }
    fn name(&self) -> &str { "Arrow IPC (.arrow, .feather)" }
    fn extensions(&self) -> &[&str] { &["arrow", "feather"] }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let path = path.to_path_buf();
        let offset = options.offset.unwrap_or(0);
        let max_records = options.max_records;
        let source = path.to_string_lossy().to_string();

        let records = tokio::task::spawn_blocking(move || -> Result<Vec<DatasetRecord>, DatasetError> {
            let file = std::fs::File::open(&path).map_err(DatasetError::Io)?;
            let reader = arrow::ipc::reader::FileReader::try_new(Arc::new(file), None)
                .map_err(|e| DatasetError::ParseError { format: "arrow".into(), error: e.to_string() })?;

            let mut records = Vec::new();
            let mut global_idx = 0usize;
            let max = max_records.unwrap_or(usize::MAX);
            for batch in reader {
                let batch = batch.map_err(|e| DatasetError::ParseError { format: "arrow".into(), error: e.to_string() })?;
                let num_rows = batch.num_rows();
                for row_idx in 0..num_rows {
                    if global_idx < offset { global_idx += 1; continue; }
                    if records.len() >= max { break; }
                    let row = crate::loaders::parquet::arrow_row_to_json_pub(&batch, row_idx);
                    let mut record = DatasetRecord::json(source.clone(), row);
                    record.modality = crate::loader::Modality::Tabular;
                    records.push(record);
                    global_idx += 1;
                }
                if records.len() >= max { break; }
            }
            Ok(records)
        })
        .await
        .map_err(|e| DatasetError::Custom(format!("join error: {e}")))??;

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}
