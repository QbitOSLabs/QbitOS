//! LMDB loader (requires `lmdb-loader` feature).
//!
//! Reads all key-value pairs from an LMDB database.

use async_trait::async_trait;
use std::path::Path;

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// LMDB loader.
pub struct LmdbLoader;

impl Default for LmdbLoader {
    fn default() -> Self {
        Self
    }
}

impl LmdbLoader {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for LmdbLoader {
    fn id(&self) -> &str { "lmdb" }
    fn name(&self) -> &str { "LMDB (.mdb, .lmdb)" }
    fn extensions(&self) -> &[&str] { &["mdb", "lmdb"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let env = unsafe {
            lmdb_rkv::Environment::new()
                .set_max_dbs(10)
                .open(path)
                .map_err(|e| DatasetError::ParseError { format: "lmdb".into(), error: e.to_string() })?
        };
        let db = env.open_db(None)
            .map_err(|e| DatasetError::ParseError { format: "lmdb".into(), error: e.to_string() })?;

        // Read all key-value pairs (LMDB doesn't support streaming across tasks,
        // so we read them all up-front)
        let txn = env.begin_ro_txn()
            .map_err(|e| DatasetError::ParseError { format: "lmdb".into(), error: e.to_string() })?;
        let mut records = Vec::new();
        let mut cursor = txn.open_ro_cursor(db)
            .map_err(|e| DatasetError::ParseError { format: "lmdb".into(), error: e.to_string() })?;
        for (k, v) in cursor.iter() {
            let key_str = String::from_utf8_lossy(k).to_string();
            let value_str = String::from_utf8_lossy(v).to_string();
            let mut record = DatasetRecord::json(
                path.to_string_lossy().to_string(),
                serde_json::json!({"key": key_str, "value": value_str}),
            );
            record.modality = crate::loader::Modality::Json;
            records.push(record);
        }
        drop(cursor);
        txn.commit().map_err(|e| DatasetError::ParseError { format: "lmdb".into(), error: e.to_string() })?;

        Ok(Box::new(crate::streaming::MemoryRecordStream::new(records)))
    }
}
