//! HDF5 loader (requires `hdf5-loader` feature).
//!
//! Reads datasets from an HDF5 file. Each top-level dataset becomes a
//! stream of records.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// HDF5 loader.
pub struct Hdf5Loader;

impl Default for Hdf5Loader {
    fn default() -> Self {
        Self
    }
}

impl Hdf5Loader {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for Hdf5Loader {
    fn id(&self) -> &str { "hdf5" }
    fn name(&self) -> &str { "HDF5 (.h5, .hdf5)" }
    fn extensions(&self) -> &[&str] { &["h5", "hdf5"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let file = hdf5::File::open(path)
            .map_err(|e| DatasetError::ParseError { format: "hdf5".into(), error: e.to_string() })?;
        let source = path.to_string_lossy().to_string();

        let mut records = Vec::new();
        // List all datasets in the file
        let datasets = file.member_names()
            .map_err(|e| DatasetError::ParseError { format: "hdf5".into(), error: e.to_string() })?;
        for name in datasets {
            if let Ok(ds) = file.dataset(&name) {
                if let Ok(data) = ds.read_1d::<f64>() {
                    for (i, v) in data.iter().enumerate() {
                        let mut record = DatasetRecord::json(
                            source.clone(),
                            serde_json::json!({"dataset": name, "index": i, "value": v}),
                        );
                        record.modality = crate::loader::Modality::TimeSeries;
                        records.push(record);
                    }
                }
            }
        }

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}
