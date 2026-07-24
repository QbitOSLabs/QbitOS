//! Skill signatures — type signatures for skill inputs and outputs.

use serde::{Deserialize, Serialize};

/// A type signature for a skill input or output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    /// The type name (e.g. "string", "number", "object", "array").
    pub type_name: String,
    /// Optional sub-signature for object properties or array items.
    pub items: Option<Box<Signature>>,
    /// Optional description.
    pub description: Option<String>,
    /// Whether this parameter is required.
    pub required: bool,
}

impl Signature {
    /// Create a string signature.
    #[must_use]
    pub fn string(description: impl Into<String>) -> Self {
        Self {
            type_name: "string".into(),
            items: None,
            description: Some(description.into()),
            required: true,
        }
    }

    /// Create a number signature.
    #[must_use]
    pub fn number(description: impl Into<String>) -> Self {
        Self {
            type_name: "number".into(),
            items: None,
            description: Some(description.into()),
            required: true,
        }
    }

    /// Create an object signature.
    #[must_use]
    pub fn object(description: impl Into<String>) -> Self {
        Self {
            type_name: "object".into(),
            items: None,
            description: Some(description.into()),
            required: true,
        }
    }

    /// Create an array signature with the given item type.
    #[must_use]
    pub fn array(items: Signature, description: impl Into<String>) -> Self {
        Self {
            type_name: "array".into(),
            items: Some(Box::new(items)),
            description: Some(description.into()),
            required: true,
        }
    }

    /// Validate a JSON value against this signature (best-effort).
    #[must_use]
    pub fn validate(&self, value: &serde_json::Value) -> bool {
        match self.type_name.as_str() {
            "string" => value.is_string(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "object" => value.is_object(),
            "array" => value.is_array() && self.items.as_ref().map_or(true, |sig| {
                value.as_array().unwrap().iter().all(|v| sig.validate(v))
            }),
            _ => true,
        }
    }
}

/// A complete skill signature (input + output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSignature {
    /// The skill name.
    pub name: String,
    /// Input signature.
    pub input: Signature,
    /// Output signature.
    pub output: Signature,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_signature_validates_strings() {
        let sig = Signature::string("a string");
        assert!(sig.validate(&serde_json::json!("hello")));
        assert!(!sig.validate(&serde_json::json!(42)));
    }

    #[test]
    fn array_signature_validates_items() {
        let sig = Signature::array(Signature::number("a number"), "list of numbers");
        assert!(sig.validate(&serde_json::json!([1, 2, 3])));
        assert!(!sig.validate(&serde_json::json!(["a", "b"])));
    }

    #[test]
    fn object_signature_validates_objects() {
        let sig = Signature::object("an object");
        assert!(sig.validate(&serde_json::json!({"key": "value"})));
        assert!(!sig.validate(&serde_json::json!("not an object")));
    }
}
