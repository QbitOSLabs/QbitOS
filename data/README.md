# qbitos-data

**Unified dataset loader for QbitOS.**

A single [`DatasetLoader`] trait + registry that imports training and
evaluation data from **40+ formats** with zero boilerplate. Every loader
emits a uniform stream of [`DatasetRecord`]s, so downstream training
pipelines can treat every dataset the same way regardless of its on-disk
format.

## Supported formats

| Category | Format | Feature | Extension(s) |
|----------|--------|---------|--------------|
| Text | Plain text, Markdown | `text` | `.txt`, `.md` |
| Structured | JSON | `json` | `.json` |
| Structured | JSON Lines / NDJSON | `jsonl` | `.jsonl`, `.ndjson` |
| Structured | CSV, TSV | (default) | `.csv`, `.tsv` |
| Structured | YAML | `yaml` | `.yaml`, `.yml` |
| Structured | XML | `xml` | `.xml`, `.rss`, `.atom` |
| Tabular | Parquet | `parquet` | `.parquet`, `.pq` |
| Tabular | Arrow IPC | `arrow` | `.arrow`, `.feather` |
| Tabular | SQLite | `sqlite` | `.sqlite`, `.db`, `.sqlite3` |
| Tabular | Excel | `excel` | `.xlsx`, `.xls`, `.xlsm` |
| Documents | PDF | `pdf` | `.pdf` |
| Documents | HTML | `html` | `.html`, `.htm` |
| Code | 50+ languages | `source-code` | `.py`, `.rs`, `.cpp`, `.js`, `.go`, ... |
| Code | Git repository | `git` | directory with `.git/` |
| Multimodal | Image | `image-loader` | `.jpg`, `.png`, `.webp`, `.gif`, `.bmp` |
| Multimodal | Audio | `audio-loader` | `.wav`, `.mp3`, `.flac`, `.ogg`, `.aac` |
| Multimodal | Video | `video-loader` | `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm` |
| Multimodal | OCR pairs | `ocr-loader` | JSONL with `image`+`text` |
| Conversation | ShareGPT, ChatML, OpenAI, Llama | `conversation` | `.jsonl`, `.json` |
| Instruction | Alpaca, OpenAI JSONL, Dolly | `instruction` | `.jsonl`, `.json` |
| Agent | Function calling | `function-call` | `.jsonl`, `.json` |
| Agent | Trajectories (goal→plan→act→observe) | `agent-trajectory` | `.jsonl`, `.json` |
| Preference | DPO, RLHF chosen/rejected | `preference` | `.jsonl`, `.json` |
| Knowledge graph | RDF, NTriples, Turtle, GraphML, JSON Graph | `knowledge-graph` | `.nt`, `.ttl`, `.rdf`, `.graphml` |
| Vector | Embeddings + metadata | `vector` | `.jsonl`, `.json` |
| Time-series | CSV/JSONL with timestamp | `time-series` | `.csv`, `.jsonl` |
| Tabular | Tables | `tabular` | `.csv`, `.tsv`, `.json` |
| Archives | TAR, TAR.GZ, TGZ (WebDataset shards) | `webdataset` | `.tar`, `.tgz`, `.tar.gz` |
| Specialty | LMDB | `lmdb-loader` | `.mdb`, `.lmdb` |
| Specialty | HDF5 | `hdf5-loader` | `.h5`, `.hdf5` |
| Cloud | Hugging Face Hub | `huggingface` | `hf:dataset/repo[:file]` |
| Cloud | S3 / MinIO | `object-storage` | `s3://bucket/key`, `minio://bucket/key` |

That's **33 distinct formats** out of the box. Combined with the
synthetic data generator and the dataset catalog, that's everything
listed in the QbitOS dataset requirement table.

## Quick start

```bash
# List all registered loaders
cargo run -p qbitos-data -- list-loaders

# Inspect a dataset (format + schema + sample count)
cargo run -p qbitos-data -- inspect data.jsonl

# Load a dataset and print records as JSONL
cargo run -p qbitos-data -- load data.csv --max 10 --offset 0

# Generate synthetic data for testing
cargo run -p qbitos-data -- synthetic --kind conversation --count 100
```

## Library usage

```rust
use qbitos_data::{DatasetLoader, DatasetRegistry, LoadOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(std::path::Path::new("data.jsonl"))?;

    // Detect schema without loading all records
    let schema = loader.detect_schema(std::path::Path::new("data.jsonl")).await?;
    println!("Schema: {:#?}", schema);

    // Stream records one at a time
    let opts = LoadOptions::new().with_max_records(100).with_offset(0);
    let mut stream = loader.load_with_options(std::path::Path::new("data.jsonl"), &opts).await?;
    while let Some(record) = stream.next_record().await? {
        println!("{:?}", record.modality);
        println!("{:?}", record.content);
    }
    Ok(())
}
```

## Feature flags

By default, only the lightweight loaders (no system dependencies) are
enabled:

```toml
[features]
default = [
    "text", "json", "jsonl", "csv", "yaml", "xml",
    "source-code", "conversation", "instruction",
    "function-call", "agent-trajectory", "preference",
    "knowledge-graph", "vector", "time-series", "tabular",
    "archive", "synthetic", "streaming",
]
```

To enable all formats:

```toml
qbitos-data = { path = "../data", features = ["all-formats"] }
```

Or pick individual heavy formats:

```toml
qbitos-data = { path = "../data", features = ["parquet", "sqlite", "pdf", "html", "image-loader"] }
```

## Uniform record envelope

Every loader returns a `DatasetRecord` with the same shape:

```rust
pub struct DatasetRecord {
    pub id: Uuid,
    pub modality: Modality,                  // Text, Json, Code, Image, ...
    pub mime: String,                        // "application/json", "image/png", ...
    pub source: String,                      // file path or URL
    pub content: serde_json::Value,          // always populated
    pub bytes: Option<Vec<u8>>,              // for binary blobs (images, audio)
    pub embedding: Option<Vec<f32>>,         // for vector datasets
    pub metadata: BTreeMap<String, Value>,   // free-form (language, dimensions, ...)
    pub loaded_at: DateTime<Utc>,
}
```

This means a training pipeline that consumes `DatasetRecord`s can handle
**every** format without format-specific code.

## The dataset catalog

The [`DatasetCatalog`] keeps track of all datasets known to the system:
their paths, formats, schemas, and statistics. Use it to build a
"Hugging Face Hub"-style browser of local datasets.

```rust
use qbitos_data::DatasetCatalog;

let catalog = DatasetCatalog::new();
catalog.register("alpaca", "data/alpaca.jsonl", "jsonl", vec!["instruction", "training"]);
catalog.register("dolly", "data/dolly.jsonl", "jsonl", vec!["instruction", "training"]);

for entry in catalog.list() {
    println!("{} ({}) — {} bytes", entry.name, entry.format, entry.size_bytes.unwrap_or(0));
}
```

## Synthetic data generator

For bootstrapping the cognitive kernel before real training data is
available:

```rust
use qbitos_data::{SyntheticGenerator, SyntheticSpec, SyntheticKind};

let spec = SyntheticSpec {
    kind: SyntheticKind::Conversation,
    count: 1000,
    seed: 42,
};
let records = SyntheticGenerator::new(spec).generate();
```

Kinds: `RandomJson`, `Conversation`, `Trajectory`, `Preference`, `Numbers`.

## License

Apache-2.0. See [../LICENSE](../LICENSE).
