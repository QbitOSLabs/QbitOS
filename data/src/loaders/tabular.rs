//! Tabular dataset loader.
//!
//! Loads tabular data from CSV/TSV/JSON. Acts as a thin alias for the CSV
//! loader for plain `.csv` files, but also supports `.arrow` and `.parquet`
//! when those features are enabled.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecordStream, LoadOptions,
};

/// Tabular loader.
pub struct TabularLoader;

impl Default for TabularLoader {
    fn default() -> Self {
        Self
    }
}

impl TabularLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for TabularLoader {
    fn id(&self) -> &str { "tabular" }
    fn name(&self) -> &str { "Tabular (CSV, TSV, JSON tables)" }
    fn extensions(&self) -> &[&str] { &["csv", "tsv", "json"] }

    fn matches(&self, path: &Path) -> bool {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name.contains("table") || name.contains("tabular") || name.contains("data.csv")
    }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "csv" | "tsv" => {
                let csv_loader = super::csv::CsvLoader::new();
                csv_loader.load_with_options(path, options).await
            }
            "json" => {
                let json_loader = super::json::JsonLoader::new();
                json_loader.load_with_options(path, options).await
            }
            _ => Err(DatasetError::UnsupportedFormat(ext.to_string())),
        }
    }
}
