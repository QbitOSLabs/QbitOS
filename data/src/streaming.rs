//! Streaming helpers.
//!
//! [`MemoryRecordStream`] holds a list of pre-loaded records and yields them
//! one at a time. Useful for testing and for loaders that read the whole
//! file into memory (e.g. JSON, YAML).

use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::loader::{DatasetError, DatasetRecord, DatasetRecordStream};
use crate::schema::DatasetSchema;

/// A stream backed by an in-memory vector of records.
pub struct MemoryRecordStream {
    records: Vec<DatasetRecord>,
    pos: AtomicUsize,
    schema: Option<DatasetSchema>,
}

impl MemoryRecordStream {
    /// Create a new stream from a vector of records.
    #[must_use]
    pub fn new(records: Vec<DatasetRecord>) -> Self {
        Self { records, pos: AtomicUsize::new(0), schema: None }
    }

    /// Attach a schema.
    #[must_use]
    pub fn with_schema(mut self, schema: DatasetSchema) -> Self {
        self.schema = Some(schema);
        self
    }
}

#[async_trait]
impl DatasetRecordStream for MemoryRecordStream {
    async fn next_record(&mut self) -> Result<Option<DatasetRecord>, DatasetError> {
        let pos = self.pos.fetch_add(1, Ordering::SeqCst);
        Ok(self.records.get(pos).cloned())
    }

    fn records_seen(&self) -> usize {
        self.pos.load(Ordering::SeqCst)
    }

    fn schema(&self) -> Option<&DatasetSchema> {
        self.schema.as_ref()
    }
}

/// A stream that wraps a `tokio::io::AsyncBufRead` and yields one record per
/// line (used by JSONL and CSV loaders).
pub struct LineStream<R> {
    reader: R,
    line_buf: String,
    records_seen: usize,
    finished: bool,
}

impl<R> LineStream<R> {
    /// Create a new line stream.
    #[must_use]
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_buf: String::new(),
            records_seen: 0,
            finished: false,
        }
    }
}

use tokio::io::{AsyncBufRead, AsyncBufReadExt};

#[async_trait]
impl<R> DatasetRecordStream for LineStream<R>
where
    R: AsyncBufRead + Unpin + Send,
{
    async fn next_record(&mut self) -> Result<Option<DatasetRecord>, DatasetError> {
        if self.finished {
            return Ok(None);
        }
        loop {
            self.line_buf.clear();
            let n = self.reader.read_line(&mut self.line_buf).await?;
            if n == 0 {
                self.finished = true;
                return Ok(None);
            }
            let line = self.line_buf.trim();
            if line.is_empty() {
                continue;
            }
            self.records_seen += 1;
            return Ok(Some(DatasetRecord::text("line", line.to_string())));
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen
    }
}
