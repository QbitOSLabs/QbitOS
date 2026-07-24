//! Excel loader (requires `excel` feature).
//!
//! Reads `.xlsx` and `.xls` files using `calamine`. Emits one record per row.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Excel loader.
pub struct ExcelLoader;

impl Default for ExcelLoader {
    fn default() -> Self { Self }
}

impl ExcelLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self { Self }
}

#[async_trait]
impl DatasetLoader for ExcelLoader {
    fn id(&self) -> &str { "excel" }
    fn name(&self) -> &str { "Excel (.xlsx, .xls)" }
    fn extensions(&self) -> &[&str] { &["xlsx", "xls", "xlsm", "xlsb"] }

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
            use calamine::{open_workbook_auto, Data, Reader};
            let max = max_records.unwrap_or(usize::MAX);

            let mut wb = open_workbook_auto(&path)
                .map_err(|e| DatasetError::ParseError { format: "excel".into(), error: e.to_string() })?;
            let sheet = wb.worksheet_range_at(0)
                .ok_or_else(|| DatasetError::ParseError { format: "excel".into(), error: "no sheet".into() })?
                .map_err(|e| DatasetError::ParseError { format: "excel".into(), error: e.to_string() })?;

            let mut rows = sheet.rows().peekable();
            let header: Option<Vec<String>> = rows.next().map(|r| {
                r.iter().map(|c| match c {
                    Data::String(s) => s.clone(),
                    _ => format!("{c}"),
                }).collect()
            });

            let mut records = Vec::new();
            let mut global_idx = 0usize;
            for row in rows {
                if global_idx < offset { global_idx += 1; continue; }
                if records.len() >= max { break; }
                let mut map = serde_json::Map::new();
                for (i, cell) in row.iter().enumerate() {
                    let key = header.as_ref().and_then(|h| h.get(i).cloned()).unwrap_or_else(|| format!("col_{i}"));
                    let value = match cell {
                        Data::Int(n) => serde_json::Value::Number((*n).into()),
                        Data::Float(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
                        Data::String(s) => serde_json::Value::String(s.clone()),
                        Data::DateTime(d) => serde_json::Value::String(format!("{d:?}")),
                        Data::Bool(b) => serde_json::Value::Bool(*b),
                        Data::Error(e) => serde_json::Value::String(format!("error: {e:?}")),
                        Data::Empty => serde_json::Value::Null,
                        Data::DurationIso(s) => serde_json::Value::String(s.clone()),
                        Data::DateTimeIso(s) => serde_json::Value::String(s.clone()),
                    };
                    map.insert(key, value);
                }
                let mut record = DatasetRecord::json(source.clone(), serde_json::Value::Object(map));
                record.modality = crate::loader::Modality::Tabular;
                records.push(record);
                global_idx += 1;
            }
            Ok(records)
        })
        .await
        .map_err(|e| DatasetError::Custom(format!("join error: {e}")))??;

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}
