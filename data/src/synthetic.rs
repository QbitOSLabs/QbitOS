//! Synthetic data generators.
//!
//! Generates synthetic datasets for testing and bootstrapping. Useful for
//! validating the cognitive kernel without requiring real training data.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::loader::{DatasetRecord, Modality};

/// A synthetic data specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticSpec {
    /// Generator kind.
    pub kind: SyntheticKind,
    /// Number of records to generate.
    pub count: usize,
    /// Random seed.
    pub seed: u64,
}

/// The kind of synthetic data to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticKind {
    /// Random JSON objects.
    RandomJson,
    /// Random conversation turns.
    Conversation,
    /// Random agent trajectories.
    Trajectory,
    /// Random preference pairs.
    Preference,
    /// Random numbers (for time-series).
    Numbers,
}

/// The synthetic generator.
pub struct SyntheticGenerator {
    spec: SyntheticSpec,
    rng: parking_lot::Mutex<rand::rngs::SmallRng>,
}

impl SyntheticGenerator {
    /// Create a new generator.
    #[must_use]
    pub fn new(spec: SyntheticSpec) -> Self {
        use rand::SeedableRng;
        let seed = spec.seed;
        Self {
            spec,
            rng: parking_lot::Mutex::new(rand::rngs::SmallRng::seed_from_u64(seed)),
        }
    }

    /// Generate `count` records.
    #[must_use]
    pub fn generate(&self) -> Vec<DatasetRecord> {
        let mut out = Vec::with_capacity(self.spec.count);
        let mut rng = self.rng.lock();
        for i in 0..self.spec.count {
            let record = match self.spec.kind {
                SyntheticKind::RandomJson => {
                    let n_keys = rng.r#gen_range(1..5);
                    let mut map = serde_json::Map::new();
                    for j in 0..n_keys {
                        let key = format!("field_{j}");
                        let val: i64 = rng.r#gen_range(0..1000);
                        map.insert(key, serde_json::Value::Number(val.into()));
                    }
                    DatasetRecord {
                        id: uuid::Uuid::now_v7(),
                        modality: Modality::Json,
                        mime: "application/json".into(),
                        source: format!("synthetic://random_json/{i}"),
                        content: serde_json::Value::Object(map),
                        bytes: None,
                        embedding: None,
                        metadata: std::collections::BTreeMap::new(),
                        loaded_at: chrono::Utc::now(),
                    }
                }
                SyntheticKind::Conversation => {
                    let turns = vec![
                        serde_json::json!({"role": "user", "content": format!("Question {i}")}),
                        serde_json::json!({"role": "assistant", "content": format!("Answer {i}")}),
                    ];
                    DatasetRecord {
                        id: uuid::Uuid::now_v7(),
                        modality: Modality::Conversation,
                        mime: "application/json".into(),
                        source: format!("synthetic://conversation/{i}"),
                        content: serde_json::Value::Array(turns),
                        bytes: None,
                        embedding: None,
                        metadata: std::collections::BTreeMap::new(),
                        loaded_at: chrono::Utc::now(),
                    }
                }
                SyntheticKind::Trajectory => {
                    DatasetRecord {
                        id: uuid::Uuid::now_v7(),
                        modality: Modality::Trajectory,
                        mime: "application/json".into(),
                        source: format!("synthetic://trajectory/{i}"),
                        content: serde_json::json!({
                            "goal": format!("goal_{i}"),
                            "steps": [
                                {"action": "observe", "result": "ok"},
                                {"action": "plan", "result": "ok"},
                                {"action": "execute", "result": "ok"},
                            ],
                        }),
                        bytes: None,
                        embedding: None,
                        metadata: std::collections::BTreeMap::new(),
                        loaded_at: chrono::Utc::now(),
                    }
                }
                SyntheticKind::Preference => {
                    DatasetRecord {
                        id: uuid::Uuid::now_v7(),
                        modality: Modality::Preference,
                        mime: "application/json".into(),
                        source: format!("synthetic://preference/{i}"),
                        content: serde_json::json!({
                            "prompt": format!("prompt_{i}"),
                            "chosen": format!("good response {i}"),
                            "rejected": format!("bad response {i}"),
                        }),
                        bytes: None,
                        embedding: None,
                        metadata: std::collections::BTreeMap::new(),
                        loaded_at: chrono::Utc::now(),
                    }
                }
                SyntheticKind::Numbers => {
                    let v: f64 = rng.r#gen_range(-1.0..1.0);
                    DatasetRecord {
                        id: uuid::Uuid::now_v7(),
                        modality: Modality::TimeSeries,
                        mime: "application/json".into(),
                        source: format!("synthetic://numbers/{i}"),
                        content: serde_json::json!({"t": i, "v": v}),
                        bytes: None,
                        embedding: None,
                        metadata: std::collections::BTreeMap::new(),
                        loaded_at: chrono::Utc::now(),
                    }
                }
            };
            out.push(record);
        }
        out
    }
}
