//! Knowledge-graph loader.
//!
//! Supports:
//!
//! - **JSON Graph**: `{"nodes": [...], "edges": [...]}`
//! - **RDF/Turtle**: text-based triple format
//! - **GraphML**: XML-based graph format
//! - **NTriples**: `<subject> <predicate> <object> .` per line

use async_trait::async_trait;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};

/// Knowledge-graph loader.
pub struct KnowledgeGraphLoader;

impl Default for KnowledgeGraphLoader {
    fn default() -> Self {
        Self
    }
}

impl KnowledgeGraphLoader {
    /// Create a new loader instance with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for KnowledgeGraphLoader {
    fn id(&self) -> &str { "knowledge-graph" }
    fn name(&self) -> &str { "Knowledge graph (JSON Graph, RDF, NTriples, GraphML)" }
    fn extensions(&self) -> &[&str] { &["nt", "n3", "ttl", "rdf", "graphml", "json"] }

    fn matches(&self, path: &Path) -> bool {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        matches!(ext, "nt" | "n3" | "ttl" | "rdf" | "graphml")
            || path.file_name().and_then(|n| n.to_str()).map(|n| n.contains("graph") || n.contains("kg")).unwrap_or(false)
    }

    async fn load_with_options(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "nt" | "n3" | "ttl" | "rdf" => {
                // NTriples: line-based
                let file = tokio::fs::File::open(path).await.map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        DatasetError::NotFound(path.display().to_string())
                    } else {
                        DatasetError::Io(e)
                    }
                })?;
                Ok(Box::new(NTriplesStream {
                    reader: BufReader::new(file),
                    source: path.to_string_lossy().to_string(),
                    line_buf: String::new(),
                    offset: options.offset.unwrap_or(0),
                    max_records: options.max_records,
                    records_seen: 0,
                    finished: false,
                }))
            }
            "graphml" => {
                // Use the XML loader's output and emit one record per <edge>
                let content = crate::loader::read_file_to_string(path).await?;
                let records = parse_graphml(&path.to_string_lossy(), &content);
                Ok(Box::new(crate::streaming::MemoryRecordStream::new(records)))
            }
            _ => {
                // JSON Graph: {nodes: [...], edges: [...]} → emit one record per edge
                let content = crate::loader::read_file_to_string(path).await?;
                let value: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| DatasetError::ParseError { format: "json".into(), error: e.to_string() })?;
                let source = path.to_string_lossy().to_string();
                let records = parse_json_graph(source, value);
                Ok(Box::new(crate::streaming::MemoryRecordStream::new(records)))
            }
        }
    }
}

/// Streaming NTriples reader.
pub struct NTriplesStream {
    reader: tokio::io::BufReader<tokio::fs::File>,
    source: String,
    line_buf: String,
    offset: usize,
    max_records: Option<usize>,
    records_seen: usize,
    finished: bool,
}

#[async_trait]
impl DatasetRecordStream for NTriplesStream {
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
            if line.is_empty() || line.starts_with('#') {
                continue;
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
            if let Some(record) = parse_ntriples_line(self.source.clone(), line) {
                return Ok(Some(record));
            }
        }
    }

    fn records_seen(&self) -> usize {
        self.records_seen
    }
}

/// Parse a single NTriples line: `<subject> <predicate> <object> .`
fn parse_ntriples_line(source: String, line: &str) -> Option<DatasetRecord> {
    let line = line.trim_end_matches('.').trim();
    let parts: Vec<&str> = line.splitn(3, ' ').collect();
    if parts.len() != 3 {
        return None;
    }
    let strip = |s: &str| -> String {
        let s = s.trim();
        if s.starts_with('<') && s.ends_with('>') {
            s[1..s.len() - 1].to_string()
        } else if s.starts_with('"') && s.ends_with('"') {
            s[1..s.len() - 1].to_string()
        } else {
            s.to_string()
        }
    };
    let mut record = DatasetRecord::json(source, serde_json::json!({
        "subject": strip(parts[0]),
        "predicate": strip(parts[1]),
        "object": strip(parts[2]),
    }));
    record.modality = crate::loader::Modality::Triple;
    Some(record)
}

/// Parse a JSON Graph document: emit one record per edge.
fn parse_json_graph(source: String, value: serde_json::Value) -> Vec<DatasetRecord> {
    let mut records = Vec::new();
    if let Some(edges) = value.get("edges").and_then(|e| e.as_array()) {
        for edge in edges {
            let mut record = DatasetRecord::json(source.clone(), edge.clone());
            record.modality = crate::loader::Modality::Triple;
            records.push(record);
        }
    }
    if records.is_empty() {
        // Fall back to emitting the whole document as one record
        records.push(DatasetRecord::json(source, value));
    }
    records
}

/// Parse a GraphML document: extract <edge> elements.
fn parse_graphml(source: &str, content: &str) -> Vec<DatasetRecord> {
    // Regex-based extraction for common triple patterns
    let re = regex::Regex::new(r"<edge\s+([^>]*)>").ok();
    let mut records = Vec::new();
    if let Some(re) = re {
        for caps in re.captures_iter(content) {
            let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let mut edge = serde_json::Map::new();
            for attr_pair in attrs.split_whitespace() {
                if let Some((k, v)) = attr_pair.split_once('=') {
                    let v = v.trim_matches('"');
                    edge.insert(k.to_string(), serde_json::Value::String(v.into()));
                }
            }
            let mut record = DatasetRecord::json(source.to_string(), serde_json::Value::Object(edge));
            record.modality = crate::loader::Modality::Triple;
            records.push(record);
        }
    }
    if records.is_empty() {
        records.push(DatasetRecord::text(source.to_string(), content.to_string()));
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn load_ntriples() {
        let tmp = tempfile::Builder::new().suffix(".nt").tempfile().unwrap();
        std::fs::write(tmp.path(), "<a> <b> <c> .\n<x> <y> <z> .\n").unwrap();
        let loader = KnowledgeGraphLoader::new();
        let mut stream = loader.load(tmp.path()).await.unwrap();
        let r = stream.next_record().await.unwrap().unwrap();
        assert_eq!(r.modality, crate::loader::Modality::Triple);
        assert_eq!(r.content["subject"], "a");
        assert_eq!(r.content["predicate"], "b");
        assert_eq!(r.content["object"], "c");
    }
}
