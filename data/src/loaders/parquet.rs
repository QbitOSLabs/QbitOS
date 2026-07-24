//! Parquet loader (requires `parquet` feature).
//!
//! Reads Parquet files using `arrow-rs` and emits one record per row.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Parquet loader.
pub struct ParquetLoader;

impl Default for ParquetLoader {
    fn default() -> Self { Self }
}

impl ParquetLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self { Self }
}

#[async_trait]
impl DatasetLoader for ParquetLoader {
    fn id(&self) -> &str { "parquet" }
    fn name(&self) -> &str { "Parquet (.parquet)" }
    fn extensions(&self) -> &[&str] { &["parquet", "pq"] }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let path = path.to_path_buf();
        let offset = options.offset.unwrap_or(0);
        let max_records = options.max_records;
        let source = path.to_string_lossy().to_string();

        // Run the blocking parquet read on a spawn_blocking task
        let records = tokio::task::spawn_blocking(move || -> Result<Vec<DatasetRecord>, DatasetError> {
            let file = std::fs::File::open(&path)
                .map_err(DatasetError::Io)?;
            let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .map_err(|e| DatasetError::ParseError { format: "parquet".into(), error: e.to_string() })?
                .build()
                .map_err(|e| DatasetError::ParseError { format: "parquet".into(), error: e.to_string() })?;

            let mut records = Vec::new();
            let mut global_idx = 0usize;
            let max = max_records.unwrap_or(usize::MAX);
            for batch in reader {
                let batch = batch.map_err(|e| DatasetError::ParseError { format: "parquet".into(), error: e.to_string() })?;
                let num_rows = batch.num_rows();
                for row_idx in 0..num_rows {
                    if global_idx < offset {
                        global_idx += 1;
                        continue;
                    }
                    if records.len() >= max { break; }
                    let json = arrow_batch_row_to_json(&batch, row_idx);
                    let mut record = DatasetRecord::json(source.clone(), json);
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

/// Convert a single row of an Arrow RecordBatch to a JSON object.
fn arrow_batch_row_to_json(batch: &arrow::array::RecordBatch, row: usize) -> serde_json::Value {
    use arrow::array::*;
    use arrow::datatypes::*;
    let schema = batch.schema();
    let mut map = serde_json::Map::new();
    for (i, col) in batch.columns().iter().enumerate() {
        let name = schema.field(i).name().to_string();
        let value = if col.is_null(row) {
            serde_json::Value::Null
        } else {
            match col.data_type() {
                DataType::Int8 => {
                    let a = col.as_any().downcast_ref::<Int8Array>().map(|a| i64::from(a.value(row))).unwrap_or(0);
                    serde_json::Value::Number(a.into())
                }
                DataType::Int16 => {
                    let a = col.as_any().downcast_ref::<Int16Array>().map(|a| i64::from(a.value(row))).unwrap_or(0);
                    serde_json::Value::Number(a.into())
                }
                DataType::Int32 => {
                    let a = col.as_any().downcast_ref::<Int32Array>().map(|a| i64::from(a.value(row))).unwrap_or(0);
                    serde_json::Value::Number(a.into())
                }
                DataType::Int64 => {
                    let a = col.as_any().downcast_ref::<Int64Array>().map(|a| a.value(row)).unwrap_or(0);
                    serde_json::Value::Number(a.into())
                }
                DataType::UInt8 => {
                    let a = col.as_any().downcast_ref::<UInt8Array>().map(|a| i64::from(a.value(row))).unwrap_or(0);
                    serde_json::Value::Number(a.into())
                }
                DataType::UInt16 => {
                    let a = col.as_any().downcast_ref::<UInt16Array>().map(|a| i64::from(a.value(row))).unwrap_or(0);
                    serde_json::Value::Number(a.into())
                }
                DataType::UInt32 => {
                    let a = col.as_any().downcast_ref::<UInt32Array>().map(|a| i64::from(a.value(row))).unwrap_or(0);
                    serde_json::Value::Number(a.into())
                }
                DataType::UInt64 => {
                    let a = col.as_any().downcast_ref::<UInt64Array>().map(|a| a.value(row) as i64).unwrap_or(0);
                    serde_json::Value::Number(a.into())
                }
                DataType::Float32 => {
                    let a = col.as_any().downcast_ref::<Float32Array>().map(|a| a.value(row)).unwrap_or(0.0);
                    serde_json::Number::from_f64(f64::from(a)).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)
                }
                DataType::Float64 => {
                    let a = col.as_any().downcast_ref::<Float64Array>().map(|a| a.value(row)).unwrap_or(0.0);
                    serde_json::Number::from_f64(a).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)
                }
                DataType::Boolean => {
                    let a = col.as_any().downcast_ref::<BooleanArray>().map(|a| a.value(row)).unwrap_or(false);
                    serde_json::Value::Bool(a)
                }
                DataType::Utf8 => {
                    let a = col.as_any().downcast_ref::<StringArray>().and_then(|a| a.value(row).into()).unwrap_or("");
                    serde_json::Value::String(a.to_string())
                }
                DataType::LargeUtf8 => {
                    let a = col.as_any().downcast_ref::<LargeStringArray>().and_then(|a| a.value(row).into()).unwrap_or("");
                    serde_json::Value::String(a.to_string())
                }
                _ => serde_json::Value::String(format!("{:?}", col.data_type())),
            }
        };
        map.insert(name, value);
    }
    serde_json::Value::Object(map)
}

/// Re-export the row-to-JSON helper for the Arrow loader.
pub(crate) fn arrow_row_to_json_pub(batch: &arrow::array::RecordBatch, row: usize) -> serde_json::Value {
    arrow_batch_row_to_json(batch, row)
}
