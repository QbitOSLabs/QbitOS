//! Integration tests for qbitos-data.
//!
//! Verifies that every default-enabled loader can be selected by file
//! extension and produces records from a sample file.

use std::path::PathBuf;
use qbitos_data::{DatasetLoader, DatasetRecordStream, DatasetRegistry, LoadOptions};

fn write_tmp(name: &str, content: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("qbitos-data-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, content).unwrap();
    path
}

#[tokio::test]
async fn loads_jsonl_with_correct_modality() {
    let path = write_tmp("sample.jsonl", r#"{"a":1}
{"a":2}
{"a":3}
"#);
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "jsonl");
    let mut stream = loader.load(&path).await.unwrap();
    let r = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r.modality, qbitos_data::loader::Modality::Json);
    assert_eq!(r.content["a"], 1);
}

#[tokio::test]
async fn loads_csv_with_schema() {
    let path = write_tmp("sample.csv", "name,age\nAlice,30\nBob,25\n");
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "csv");
    let schema = loader.detect_schema(&path).await.unwrap();
    assert!(schema.field("name").is_some());
    assert!(schema.field("age").is_some());
}

#[tokio::test]
async fn loads_yaml() {
    let path = write_tmp("sample.yaml", "name: Alice\nage: 30\n");
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "yaml");
    let mut stream = loader.load(&path).await.unwrap();
    let r = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r.content["name"], "Alice");
}

#[tokio::test]
async fn loads_json_array_as_multiple_records() {
    let path = write_tmp("array.json", r#"[{"x":1},{"x":2},{"x":3}]"#);
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "json");
    let mut stream = loader.load(&path).await.unwrap();
    let mut count = 0;
    while stream.next_record().await.unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 3);
}

#[tokio::test]
async fn loads_markdown_split_into_sections() {
    let path = write_tmp("doc.md", "# Title\n\n## Section 1\n\nbody 1\n\n## Section 2\n\nbody 2");
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "text");
    let mut stream = loader.load(&path).await.unwrap();
    let mut count = 0;
    while stream.next_record().await.unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 2);
}

#[tokio::test]
async fn loads_python_source_code() {
    let path = write_tmp("hello.py", "def hello():\n    print('hi')\n");
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "source-code");
    let mut stream = loader.load(&path).await.unwrap();
    let r = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r.modality, qbitos_data::loader::Modality::Code);
    assert_eq!(r.metadata.get("language").and_then(|v| v.as_str()), Some("python"));
}

#[tokio::test]
async fn loads_sharegpt_conversation() {
    // Filename must contain "conversation" so the conversation loader matches first
    let path = write_tmp("conversation.jsonl",
        r#"{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"hello"}]}"#);
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "conversation");
    let mut stream = loader.load(&path).await.unwrap();
    let r = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r.modality, qbitos_data::loader::Modality::Conversation);
    assert_eq!(r.content.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn loads_alpaca_instruction() {
    let path = write_tmp("alpaca-test.jsonl",
        r#"{"instruction":"Say hi","input":"","output":"Hi!"}"#);
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "instruction");
    let mut stream = loader.load(&path).await.unwrap();
    let r = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r.content["instruction"], "Say hi");
    assert_eq!(r.content["output"], "Hi!");
}

#[tokio::test]
async fn loads_ntriples() {
    let path = write_tmp("triples.nt", "<a> <b> <c> .\n<x> <y> <z> .\n");
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "knowledge-graph");
    let mut stream = loader.load(&path).await.unwrap();
    let r = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r.modality, qbitos_data::loader::Modality::Triple);
    assert_eq!(r.content["subject"], "a");
}

#[tokio::test]
async fn loads_vector_dataset() {
    let path = write_tmp("vector-test.jsonl",
        r#"{"id":"v1","embedding":[0.1,0.2,0.3],"metadata":{"label":"cat"}}"#);
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    assert_eq!(loader.id(), "vector");
    let mut stream = loader.load(&path).await.unwrap();
    let r = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r.modality, qbitos_data::loader::Modality::Vector);
    assert_eq!(r.embedding.unwrap(), vec![0.1, 0.2, 0.3]);
}

#[tokio::test]
async fn pagination_works() {
    let path = write_tmp("paginated.jsonl",
        "{\"i\":1}\n{\"i\":2}\n{\"i\":3}\n{\"i\":4}\n{\"i\":5}\n");
    let registry = DatasetRegistry::with_default_loaders();
    let loader = registry.select(&path).unwrap();
    let opts = LoadOptions { offset: Some(1), max_records: Some(2), ..Default::default() };
    let mut stream = loader.load_with_options(&path, &opts).await.unwrap();
    let r1 = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r1.content["i"], 2);
    let r2 = stream.next_record().await.unwrap().unwrap();
    assert_eq!(r2.content["i"], 3);
    assert!(stream.next_record().await.unwrap().is_none());
}

#[test]
fn registry_lists_all_default_loaders() {
    let registry = DatasetRegistry::with_default_loaders();
    let ids: Vec<String> = registry.list().into_iter().map(|l| l.id().to_string()).collect();
    assert!(ids.contains(&"text".to_string()));
    assert!(ids.contains(&"json".to_string()));
    assert!(ids.contains(&"jsonl".to_string()));
    assert!(ids.contains(&"csv".to_string()));
    assert!(ids.contains(&"yaml".to_string()));
    assert!(ids.contains(&"xml".to_string()));
    assert!(ids.contains(&"source-code".to_string()));
    assert!(ids.contains(&"conversation".to_string()));
    assert!(ids.contains(&"instruction".to_string()));
    assert!(ids.contains(&"function-call".to_string()));
    assert!(ids.contains(&"agent-trajectory".to_string()));
    assert!(ids.contains(&"preference".to_string()));
    assert!(ids.contains(&"knowledge-graph".to_string()));
    assert!(ids.contains(&"vector".to_string()));
    assert!(ids.contains(&"time-series".to_string()));
    assert!(ids.contains(&"tabular".to_string()));
}
