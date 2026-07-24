//! Dataset schema: describes the fields of a dataset.
//!
//! Used by the catalog and by downstream training pipelines to know what
//! columns / fields to expect.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The type of a field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    /// String.
    Text,
    /// Integer.
    Integer,
    /// Float.
    Float,
    /// Boolean.
    Boolean,
    /// JSON object.
    Object,
    /// JSON array.
    Array,
    /// Binary blob.
    Bytes,
    /// Embedding vector (Vec<f32>).
    Vector,
    /// Nested record.
    Nested(DatasetSchema),
    /// Unknown / null.
    Null,
}

impl FieldType {
    /// Infer a field type from a JSON value.
    #[must_use]
    pub fn infer_from(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(_) => Self::Boolean,
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    Self::Integer
                } else {
                    Self::Float
                }
            }
            serde_json::Value::String(_) => Self::Text,
            serde_json::Value::Array(_) => Self::Array,
            serde_json::Value::Object(_) => Self::Object,
        }
    }
}

/// A single field in a dataset schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldSchema {
    /// Field name.
    pub name: String,
    /// Field type.
    pub field_type: FieldType,
    /// Whether the field is nullable.
    pub nullable: bool,
    /// Optional description.
    pub description: Option<String>,
}

/// A dataset schema: an ordered list of fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DatasetSchema {
    /// Field list.
    pub fields: Vec<FieldSchema>,
}

impl DatasetSchema {
    /// Create an empty schema.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Infer a schema from a JSON value (typically a record's content).
    #[must_use]
    pub fn infer_from(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Object(map) => {
                let fields = map.iter().map(|(k, v)| FieldSchema {
                    name: k.clone(),
                    field_type: FieldType::infer_from(v),
                    nullable: v.is_null(),
                    description: None,
                }).collect();
                Self { fields }
            }
            _ => Self::empty(),
        }
    }

    /// Look up a field by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&FieldSchema> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Number of fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Returns `true` if the schema is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Build a name → type map.
    #[must_use]
    pub fn type_map(&self) -> AHashMap<String, FieldType> {
        self.fields.iter().map(|f| (f.name.clone(), f.field_type.clone())).collect()
    }
}

/// A shared schema handle.
pub type SharedSchema = Arc<DatasetSchema>;
