//! Integration tests for the memory backends.
//!
//! Tests the RocksDB backend (embedded, no external service needed) end-to-end.
//! Postgres/Qdrant/Redis tests are gated behind `#[ignore]` so they only run
//! when the service is available (`cargo test -- --ignored`).

#[cfg(feature = "rocksdb-backed")]
mod rocksdb_tests {
    use qbitos_memory::backends::MemoryBackend;
    use qbitos_memory::{RocksdbBackend, RocksdbConfig};
    use qbitos_memory::embedding::Embedding;
    use qbitos_memory::object::{Importance, MemoryObject, MemoryType};

    fn make_test_object(content: &str, embedding: Vec<f32>) -> MemoryObject {
        let mut obj = MemoryObject::new(
            MemoryType::Episodic,
            serde_json::json!({"content": content}),
            Embedding::new(embedding),
        );
        obj.importance = Importance(0.8);
        obj
    }

    #[tokio::test]
    async fn rocksdb_store_and_get() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = RocksdbConfig {
            path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let backend = RocksdbBackend::new(config).unwrap();

        let obj = make_test_object("hello world", vec![0.1, 0.2, 0.3]);
        let id = obj.id;

        backend.store(&obj).await.unwrap();
        let retrieved = backend.get(id).await.unwrap();
        assert_eq!(retrieved.content["content"], "hello world");
        assert_eq!(retrieved.embedding.data, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn rocksdb_delete() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = RocksdbConfig {
            path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let backend = RocksdbBackend::new(config).unwrap();

        let obj = make_test_object("to delete", vec![1.0, 0.0]);
        let id = obj.id;

        backend.store(&obj).await.unwrap();
        let retrieved = backend.get(id).await.unwrap();
        assert_eq!(retrieved.content["content"], "to delete");

        backend.delete(id).await.unwrap();
        assert!(backend.get(id).await.is_err());
    }

    #[tokio::test]
    async fn rocksdb_count() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = RocksdbConfig {
            path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let backend = RocksdbBackend::new(config).unwrap();

        for i in 0..5 {
            let obj = make_test_object(&format!("item {i}"), vec![i as f32]);
            backend.store(&obj).await.unwrap();
        }
        assert_eq!(backend.count().await.unwrap(), 5);
    }

    #[tokio::test]
    async fn rocksdb_persists_across_reopen() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().to_path_buf();

        let obj = make_test_object("persistent", vec![0.5, 0.5]);
        let id = obj.id;

        {
            let config = RocksdbConfig {
                path: path.clone(),
                ..Default::default()
            };
            let backend = RocksdbBackend::new(config).unwrap();
            backend.store(&obj).await.unwrap();
            backend.flush().await.unwrap();
            drop(backend);
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        {
            let config = RocksdbConfig {
                path: path.clone(),
                ..Default::default()
            };
            let backend = RocksdbBackend::new(config).unwrap();
            let retrieved = backend.get(id).await.unwrap();
            assert_eq!(retrieved.content["content"], "persistent");
            assert_eq!(retrieved.embedding.data, vec![0.5, 0.5]);
        }
    }

    #[tokio::test]
    async fn rocksdb_properties() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = RocksdbConfig {
            path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let backend = RocksdbBackend::new(config).unwrap();
        assert!(backend.is_durable());
        assert_eq!(backend.name(), "rocksdb");
    }
}

#[cfg(feature = "postgres-backed")]
mod postgres_tests {
    use qbitos_memory::backends::MemoryBackend;
    use qbitos_memory::{PostgresBackend, PostgresConfig};
    use qbitos_memory::embedding::Embedding;
    use qbitos_memory::object::{Importance, MemoryObject, MemoryType};

    #[tokio::test]
    #[ignore = "requires Postgres with pgvector — set DATABASE_URL to run"]
    async fn postgres_store_and_get() {
        let url = std::env::var("DATABASE_URL").unwrap_or_else(|_| "postgres://localhost/qbitos".into());
        let config = PostgresConfig { url, embedding_dim: 4, ..Default::default() };
        let backend = match PostgresBackend::new(config).await {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping postgres test: {e}");
                return;
            }
        };
        let obj = MemoryObject::new(
            MemoryType::Episodic,
            serde_json::json!({"test": "postgres"}),
            Embedding::new(vec![0.1, 0.2, 0.3, 0.4]),
        );
        obj.importance = Importance(0.9);
        let id = obj.id;

        backend.store(&obj).await.unwrap();
        let retrieved = backend.get(id).await.unwrap();
        assert_eq!(retrieved.content["test"], "postgres");
        assert_eq!(retrieved.embedding.data, vec![0.1, 0.2, 0.3, 0.4]);

        backend.delete(id).await.unwrap();
    }
}

#[cfg(feature = "qdrant-backed")]
mod qdrant_tests {
    use qbitos_memory::backends::{MemoryBackend, SimilarityQuery};
    use qbitos_memory::{QdrantBackend, QdrantConfig};
    use qbitos_memory::embedding::Embedding;
    use qbitos_memory::object::{Importance, MemoryObject, MemoryType};

    #[tokio::test]
    #[ignore = "requires Qdrant — set QDRANT_URL to run"]
    async fn qdrant_store_and_search() {
        let url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".into());
        let config = QdrantConfig { url, embedding_dim: 4, ..Default::default() };
        let backend = match QdrantBackend::new(config).await {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping qdrant test: {e}");
                return;
            }
        };
        let obj = MemoryObject::new(
            MemoryType::Episodic,
            serde_json::json!({"test": "qdrant"}),
            Embedding::new(vec![0.1, 0.2, 0.3, 0.4]),
        );
        obj.importance = Importance(0.9);
        let id = obj.id;

        backend.store(&obj).await.unwrap();
        let retrieved = backend.get(id).await.unwrap();
        assert_eq!(retrieved.content["test"], "qdrant");

        let results = backend.search(&SimilarityQuery {
            embedding: Embedding::new(vec![0.1, 0.2, 0.3, 0.4]),
            k: 1,
            layer_filter: None,
            min_score: 0.0,
        }).await.unwrap();
        assert!(!results.is_empty());

        backend.delete(id).await.unwrap();
    }
}

#[cfg(feature = "redis-backed")]
mod redis_tests {
    use qbitos_memory::backends::MemoryBackend;
    use qbitos_memory::{RedisBackend, RedisConfig};
    use qbitos_memory::embedding::Embedding;
    use qbitos_memory::object::{Importance, MemoryObject, MemoryType};

    #[tokio::test]
    #[ignore = "requires Redis — set REDIS_URL to run"]
    async fn redis_store_and_get() {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".into());
        let config = RedisConfig { url, ..Default::default() };
        let backend = match RedisBackend::new(config).await {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Skipping redis test: {e}");
                return;
            }
        };
        let obj = MemoryObject::new(
            MemoryType::Working,
            serde_json::json!({"test": "redis"}),
            Embedding::new(vec![0.1, 0.2]),
        );
        obj.importance = Importance(0.7);
        let id = obj.id;

        backend.store(&obj).await.unwrap();
        let retrieved = backend.get(id).await.unwrap();
        assert_eq!(retrieved.content["test"], "redis");

        backend.delete(id).await.unwrap();
    }
}

/// Test the multi-backend fabric integration.
mod fabric_tests {
    use qbitos_memory::MemoryFabric;
    use qbitos_memory::backends::MemoryBackend;
    use qbitos_memory::embedding::Embedding;
    use qbitos_memory::object::{Importance, MemoryObject, MemoryType};

    #[tokio::test]
    async fn fabric_backend_management() {
        let fabric = MemoryFabric::with_deterministic_embeddings(4);
        assert_eq!(fabric.backend_count(), 0);
        assert!(fabric.backend_names().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "rocksdb-backed")]
    async fn fabric_with_rocksdb_backend() {
        use qbitos_memory::backends::SharedBackend;
        use qbitos_memory::{RocksdbBackend, RocksdbConfig};
        use std::sync::Arc;

        let tmp = tempfile::TempDir::new().unwrap();
        let rocksdb_config = RocksdbConfig {
            path: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let rocksdb = RocksdbBackend::new(rocksdb_config).unwrap();

        let fabric = MemoryFabric::with_deterministic_embeddings(4);
        fabric.add_backend(Arc::new(rocksdb) as SharedBackend);

        assert_eq!(fabric.backend_count(), 1);
        assert_eq!(fabric.backend_names(), vec!["rocksdb"]);

        let mut obj = MemoryObject::new(
            MemoryType::Episodic,
            serde_json::json!({"test": "fabric"}),
            Embedding::new(vec![0.1, 0.2, 0.3, 0.4]),
        );
        obj.importance = Importance(0.8);
        let id = obj.id;

        fabric.store_to_backends(&obj).await.unwrap();

        let retrieved = fabric.get_from_backends(id).await.unwrap();
        assert_eq!(retrieved.content["test"], "fabric");
        assert_eq!(retrieved.embedding.data, vec![0.1, 0.2, 0.3, 0.4]);

        fabric.delete_from_backends(id).await.unwrap();
        assert!(fabric.get_from_backends(id).await.is_err());
    }
}
