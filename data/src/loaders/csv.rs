//! CSV loader: `.csv`.
//!
//! Reads a CSV file and emits one record per row. Each record's content is
//! a JSON object mapping column names to cell values.

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::schema::{DatasetSchema, FieldSchema, FieldType};

/// CSV loader for `.csv` files.
pub struct CsvLoader;

impl Default for CsvLoader {
    fn default() -> Self {
        Self
    }
}

impl CsvLoader {
    /// Create a new CSV loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for CsvLoader {
    fn id(&self) -> &str { "csv" }
    fn name(&self) -> &str { "CSV (.csv, .tsv)" }
    fn extensions(&self) -> &[&str] { &["csv", "tsv"] }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let file = tokio::fs::File::open(path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                DatasetError::NotFound(path.display().to_string())
            } else {
                DatasetError::Io(e)
            }
        })?;
        let delim = options.csv_delimiter.unwrap_or_else(|| {
            if path.extension().and_then(|e| e.to_str()) == Some("tsv") { '\t' } else { ',' }
        });
        let reader = csv::ReaderBuilder::new()
            .delimiter(delim as u8)
            .has_headers(true)
            .flexible(true)
            .from_reader(file.into_std().await);

        Ok(Box::new(CsvStream {
            reader,
            source: path.to_string_lossy().to_string(),
            headers: None,
            schema: None,
            offset: options.offset.unwrap_or(0),
            max_records: options.max_records,
            records_seen: 0,
            finished: false,
        }))
    }
}

/// Streaming CSV reader.
pub struct CsvStream {
    reader: csv::Reader<std::fs::File>,
    source: String,
    headers: Option<csv::StringRecord>,
    schema: Option<Arc<DatasetSchema>>,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for CsvStream {
    async fn next_record(&mut self) -> Result<Option<DatasetRecord>, DatasetError> {
        if self.finished {
            return Ok(None);
        }
        // Capture headers on first call
        if self.headers.is_none() {
            let headers = self.reader.headers()?.clone();
            // Build schema
            let fields: Vec<FieldSchema> = headers.iter().map(|h| FieldSchema {
                name: h.to_string(),
                field_type: FieldType::Text,
                nullable: true,
                description: None,
            }).collect();
            self.schema = Some(Arc::new(DatasetSchema { fields }));
            self.headers = Some(headers);
        }

        let headers = self.headers.clone().unwrap();
        let mut record = csv::StringRecord::new();
        loop {
            if !self.reader.read_record(&mut record)? {
                self.finished = true;
                return Ok(None);
            }
            self.records_seen += 1;
            if self.records_seen <= self.offset {
                continue;
            }
            if let Some(max) = self.max_records {
                if self.records_seen > self.offset + max {
                    self.finished = true;
                    return Ok(None);
                }
            }
            let mut map = serde_json::Map::new();
            for (i, header) in headers.iter().enumerate() {
                let value = record.get(i).unwrap_or("");
                // Try to parse as number, then bool, then fall back to string
                let parsed: serde_json::Value = if let Ok(n) = value.parse::<i64>() {
                    serde_json::Value::Number(n.into())
                } else if let Ok(f) = value.parse::<f64>() {
                    serde_json::Number::from_f64(f).map(serde_json::Value::Number).unwrap_or_else(|| serde_json::Value::String(value.into()))
                } else if value.eq_ignore_ascii_case("true") {
                    serde_json::Value::Bool(true)
                } else if value.eq_ignore_ascii_case("false") {
                    serde_json::Value::Bool(false)
                } else {
                    serde_json::Value::String(value.into())
                };
                map.insert(header.to_string(), parsed);
            }
            return Ok(Some(DatasetRecord {
                id: uuid::Uuid::now_v7(),
                modality: crate::loader::Modality::Tabular,
                mime: "text/csv".into(),
                source: self.source.clone(),
                content: serde_json::Value::Object(map),
                bytes: None,
                embedding: None,
                metadata: std::collections::BTreeMap::new(),
                loaded_at: chrono::Utc::now(),
            }));
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen
    }

    fn schema(&self) -> Option<&DatasetSchema> {
        self.schema.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_csv() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "name,age\nAlice,30\nBob,25\n").unwrap();
        let loader = CsvLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let r1 = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r1.content["name"], "Alice");
        assert_eq!(r1.content["age"], 30);
        let r2 = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r2.content["name"], "Bob");
    }

    #[tokio::test]
    async fn load_tsv() {
        let tmp = tempfile::Builder::new().suffix(".tsv").tempfile().unwrap();
        std::fs::write(tmp.path(), "name\tage\nAlice\t30\n").unwrap();
        let loader = CsvLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.content["name"], "Alice");
    }
}
