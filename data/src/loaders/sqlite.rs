//! SQLite loader (requires `sqlite` feature).
//!
//! Reads rows from a SQLite database table.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// SQLite loader.
pub struct SqliteLoader;

impl Default for SqliteLoader {
    fn default() -> Self { Self }
}

impl SqliteLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self { Self }
}

#[async_trait]
impl DatasetLoader for SqliteLoader {
    fn id(&self) -> &str { "sqlite" }
    fn name(&self) -> &str { "SQLite (.sqlite, .db)" }
    fn extensions(&self) -> &[&str] { &["sqlite", "db", "sqlite3"] }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let path = path.to_path_buf();
        let table = options.extra.get("table").and_then(|v| v.as_str()).map(String::from);
        let limit = options.max_records.unwrap_or(usize::MAX);
        let offset = options.offset.unwrap_or(0);
        let source = path.to_string_lossy().to_string();

        let records = tokio::task::spawn_blocking(move || -> Result<Vec<DatasetRecord>, DatasetError> {
            let conn = rusqlite::Connection::open(&path)
                .map_err(|e| DatasetError::ParseError { format: "sqlite".into(), error: e.to_string() })?;

            // Determine the table to read
            let table_name = if let Some(t) = table { t } else {
                let mut stmt = conn.prepare(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1",
                ).map_err(|e| DatasetError::ParseError { format: "sqlite".into(), error: e.to_string() })?;
                stmt.query_row([], |row| row.get::<_, String>(0))
                    .unwrap_or_else(|_| "main".into())
            };

            // Validate table_name to prevent SQL injection — only allow
            // alphanumeric + underscore (valid SQLite identifiers).
            let valid = !table_name.is_empty()
                && !table_name.starts_with(|c: char| c.is_ascii_digit())
                && table_name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if !valid {
                return Err(DatasetError::ParseError {
                    format: "sqlite".into(),
                    error: format!("invalid table name: {table_name:?} (must be a valid identifier)"),
                });
            }

            let query = format!("SELECT * FROM \"{table_name}\" LIMIT {limit} OFFSET {offset}");
            let mut stmt = conn.prepare(&query)
                .map_err(|e| DatasetError::ParseError { format: "sqlite".into(), error: e.to_string() })?;

            let col_count = stmt.column_count();
            let col_names: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();

            let rows = stmt.query([])
                .map_err(|e| DatasetError::ParseError { format: "sqlite".into(), error: e.to_string() })?;

            let mut records = Vec::new();
            let mut rows = rows;
            loop {
                let row_opt = rows.next();
                match row_opt {
                    Ok(Some(row)) => {
                        let mut map = serde_json::Map::new();
                        for i in 0..col_count {
                            let name = col_names.get(i).cloned().unwrap_or_else(|| format!("col_{i}"));
                            let value: rusqlite::Result<String> = row.get::<_, String>(i);
                            let value = match value {
                                Ok(s) => serde_json::Value::String(s),
                                Err(_) => {
                                    if let Ok(i_val) = row.get::<_, i64>(i) {
                                        serde_json::Value::Number(i_val.into())
                                    } else if let Ok(f) = row.get::<_, f64>(i) {
                                        serde_json::Number::from_f64(f).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)
                                    } else if let Ok(b) = row.get::<_, bool>(i) {
                                        serde_json::Value::Bool(b)
                                    } else {
                                        serde_json::Value::Null
                                    }
                                }
                            };
                            map.insert(name, value);
                        }
                        let mut record = DatasetRecord::json(source.clone(), serde_json::Value::Object(map));
                        record.modality = crate::loader::Modality::Tabular;
                        records.push(record);
                    }
                    Ok(None) => break,
                    Err(e) => return Err(DatasetError::ParseError { format: "sqlite".into(), error: e.to_string() }),
                }
            }
            Ok(records)
        })
        .await
        .map_err(|e| DatasetError::Custom(format!("join error: {e}")))??;

        Ok(Box::new(MemoryRecordStream::new(records)))
    }
}
