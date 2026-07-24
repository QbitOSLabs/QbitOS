//! Object storage loader (requires `object-storage` feature).
//!
//! Loads datasets from S3 / MinIO. Supports `s3://bucket/key` and
//! `minio://bucket/key` URLs. Uses `reqwest` to talk to the S3-compatible
//! API.
//!
//! Note: this is a minimal implementation that supports unauthenticated
//! reads from public buckets. For production use, add SigV4 signing.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use crate::loader::{
    DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Object-storage loader.
pub struct ObjectStorageLoader {
    endpoint: Option<String>,
    access_key: Option<String>,
    secret_key: Option<String>,
    region: String,
}

impl Default for ObjectStorageLoader {
    fn default() -> Self {
        Self {
            endpoint: std::env::var("S3_ENDPOINT").ok(),
            access_key: std::env::var("S3_ACCESS_KEY").ok(),
            secret_key: std::env::var("S3_SECRET_KEY").ok(),
            region: std::env::var("S3_REGION").unwrap_or_else(|_| "us-east-1".into()),
        }
    }
}

impl ObjectStorageLoader {
    /// Create a new loader with explicit credentials.
    #[must_use]
    pub fn new(endpoint: String, access_key: String, secret_key: String, region: String) -> Self {
        Self {
            endpoint: Some(endpoint),
            access_key: Some(access_key),
            secret_key: Some(secret_key),
            region,
        }
    }
}

#[async_trait]
impl DatasetLoader for ObjectStorageLoader {
    fn id(&self) -> &str { "object-storage" }
    fn name(&self) -> &str { "Object storage (S3, MinIO)" }
    fn extensions(&self) -> &[&str] { &["s3", "minio"] }

    fn matches(&self, path: &Path) -> bool {
        let s = path.to_string_lossy();
        s.starts_with("s3://") || s.starts_with("minio://")
    }

    async fn load_with_options(
        self: &Self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let url = path.to_string_lossy().to_string();
        let (scheme, rest) = url.split_once("://").ok_or_else(|| DatasetError::Custom("invalid URL".into()))?;
        let (bucket, key) = rest.split_once('/').ok_or_else(|| DatasetError::Custom("missing key".into()))?;

        let endpoint = self.endpoint.clone().unwrap_or_else(|| match scheme {
            "s3" => format!("https://s3.{}.amazonaws.com", self.region),
            "minio" => "http://localhost:9000".into(),
            _ => format!("https://{scheme}"),
        });

        let object_url = format!("{endpoint}/{bucket}/{key}");
        let client = reqwest::Client::new();
        let response = client.get(&object_url).send().await
            .map_err(|e| DatasetError::Custom(format!("object-storage error: {e}")))?;
        if !response.status().is_success() {
            return Err(DatasetError::Custom(format!("object-storage returned {}", response.status())));
        }
        let bytes = response.bytes().await
            .map_err(|e| DatasetError::Custom(format!("object-storage body error: {e}")))?;

        // Save to a temp file and delegate to the appropriate loader
        let tmp_dir = std::env::temp_dir();
        let filename = key.rsplit_once('/').map(|(_, name)| name).unwrap_or(key);
        let tmp_path: PathBuf = tmp_dir.join(format!("qbitos-os-{}-{filename}", uuid::Uuid::now_v7()));
        tokio::fs::write(&tmp_path, &bytes).await?;
        let mime = mime_guess::from_path(&tmp_path).first_or_octet_stream().to_string();

        let mut record = DatasetRecord::binary(url, mime, bytes.to_vec());
        record.metadata.insert("source_bucket".into(), serde_json::Value::String(bucket.into()));
        record.metadata.insert("source_key".into(), serde_json::Value::String(key.into()));
        record.metadata.insert("endpoint".into(), serde_json::Value::String(endpoint));
        Ok(Box::new(MemoryRecordStream::new(vec![record])))
    }
}
