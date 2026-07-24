//! QbitOS Data — Unified Dataset Loader
//!
//! A single trait ([`DatasetLoader`]) and registry ([`DatasetRegistry`])
//! that lets the cognitive kernel import training and evaluation data
//! from 40+ formats:
//!
//! - **Text**: `.txt`, `.md`
//! - **Structured**: `.json`, `.jsonl`, `.csv`, `.yaml`, `.xml`, `.toml`
//! - **Tabular**: `.parquet`, `.arrow`, `.xlsx`, `.xls`, `.sqlite`, `.db`
//! - **Documents**: `.pdf`, `.html`, `.docx`
//! - **Source code**: `.py`, `.rs`, `.cpp`, `.js`, `.go`, ... (whole git repos)
//! - **Multimodal**: `.jpg`, `.png`, `.webp`, `.mp3`, `.wav`, `.flac`, `.mp4`
//! - **Conversation**: ShareGPT, ChatML, OpenAI messages, Alpaca, Llama
//! - **Agent**: function-calling, agent trajectories, DPO preference pairs
//! - **Knowledge graphs**: RDF, Turtle, GraphML, JSON Graph
//! - **Vector datasets**: embeddings + metadata
//! - **Time-series**: CSV/Parquet with temporal indexing
//! - **Archives**: `.zip`, `.tar`, `.tar.gz`, `.tgz` (WebDataset shards)
//! - **Object storage**: S3 / MinIO (via `object-storage` feature)
//! - **Hugging Face Hub**: `datasets.load_dataset()` equivalent
//! - **Specialty**: LMDB, HDF5
//!
//! All loaders return a stream of [`DatasetRecord`]s — a uniform envelope
//! with content (bytes or JSON), modality, mime type, and metadata. This
//! lets downstream training pipelines treat every dataset the same way
//! regardless of its on-disk format.
//!
//! # Example
//!
//! ```no_run
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! use qbitos_data::{DatasetLoader, DatasetRegistry};
//! use std::path::Path;
//!
//! let registry = DatasetRegistry::with_default_loaders();
//! let path = Path::new("data.jsonl");
//! let loader = registry.select(path)?;
//! let mut stream = loader.load(path).await?;
//! while let Some(record) = stream.next_record().await? {
//!     println!("{:?}", record);
//! }
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::new_without_default
)]

pub mod catalog;
pub mod curation;
pub mod loader;
pub mod loaders;
pub mod registry;
pub mod schema;
pub mod streaming;
pub mod synthetic;

pub use loader::{DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions, LoadStats};
pub use registry::{DatasetRegistry, RegistryError};
pub use schema::{DatasetSchema, FieldSchema, FieldType};
pub use streaming::MemoryRecordStream;
pub use synthetic::{SyntheticGenerator, SyntheticKind, SyntheticSpec};
pub use catalog::{CatalogEntry, DatasetCatalog};
pub use curation::{CurationConfig, CurationError, CurationPipeline, DataSample, ImportanceSampler, SharedCurationPipeline};

/// Semantic version of the data crate.
pub const DATA_VERSION: &str = env!("CARGO_PKG_VERSION");
