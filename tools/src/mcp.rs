//! MCP tool descriptors (Model Context Protocol 2026-07-28 spec).
//!
//! Implements the Model Context Protocol's tool descriptor format, including:
//! - JSON Schema for input/output validation
//! - Tool annotations (read-only, destructive, network access, etc.)
//! - A tool router that dispatches calls with automatic validation
//! - Composition plans for multi-step tool pipelines with dependency resolution

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Errors emitted by MCP tools.
#[derive(Debug, Error)]
pub enum McpToolError {
    /// The requested tool was not found in the router.
    #[error("tool `{0}` not found")]
    NotFound(String),
    /// The input failed JSON Schema validation.
    #[error("input validation failed: {0}")]
    InvalidInput(String),
    /// The output failed JSON Schema validation.
    #[error("output validation failed: {0}")]
    InvalidOutput(String),
    /// The tool execution itself failed.
    #[error("execution error: {0}")]
    Execution(String),
}

/// A JSON Schema descriptor for tool input/output validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// The schema type: "object", "string", "number", "boolean", or "array".
    #[serde(rename = "type", default = "default_schema_type")]
    pub schema_type: String,
    /// Optional human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// For object types: property schemas keyed by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JsonSchema>>>,
    /// Required property names (for object types).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
    /// For array types: the schema for each item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonSchema>>,
    /// Optional enum constraint: the value must be one of these.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#enum: Option<Vec<Value>>,
    /// Whether additional properties are allowed on objects.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub additional_properties: bool,
}

fn default_true() -> bool {
    true
}

fn is_true(b: &bool) -> bool {
    *b
}

fn default_schema_type() -> String {
    "object".into()
}

impl JsonSchema {
    /// Create a new object schema.
    #[must_use]
    pub fn object() -> Self {
        Self {
            schema_type: "object".into(),
            description: None,
            properties: Some(HashMap::new()),
            required: Vec::new(),
            items: None,
            r#enum: None,
            additional_properties: false,
        }
    }

    /// Create a new string schema with the given description.
    #[must_use]
    pub fn string(d: impl Into<String>) -> Self {
        Self {
            schema_type: "string".into(),
            description: Some(d.into()),
            properties: None,
            required: Vec::new(),
            items: None,
            r#enum: None,
            additional_properties: true,
        }
    }

    /// Add a property schema (for object types).
    #[must_use]
    pub fn with_property(mut self, name: impl Into<String>, schema: JsonSchema) -> Self {
        self.properties
            .get_or_insert_with(HashMap::new)
            .insert(name.into(), Box::new(schema));
        self
    }

    /// Mark a property as required.
    #[must_use]
    pub fn with_required(mut self, name: impl Into<String>) -> Self {
        self.required.push(name.into());
        self
    }

    /// Validate a JSON value against this schema.
    ///
    /// # Errors
    /// Returns a string describing the validation failure.
    pub fn validate(&self, value: &Value) -> Result<(), String> {
        match self.schema_type.as_str() {
            "object" => {
                let obj = match value {
                    Value::Object(map) => map,
                    _ => return Err("expected object".into()),
                };
                for r in &self.required {
                    if !obj.contains_key(r) {
                        return Err(format!("missing: {r}"));
                    }
                }
                if let Some(props) = &self.properties {
                    for (k, v) in obj {
                        if let Some(s) = props.get(k) {
                            s.validate(v)?;
                        } else if !self.additional_properties {
                            return Err(format!("unexpected: {k}"));
                        }
                    }
                }
                Ok(())
            }
            "string" => match value {
                Value::String(_) => Ok(()),
                _ => Err("expected string".into()),
            },
            "number" => match value {
                Value::Number(_) => Ok(()),
                _ => Err("expected number".into()),
            },
            "boolean" => match value {
                Value::Bool(_) => Ok(()),
                _ => Err("expected boolean".into()),
            },
            "array" => {
                if let Value::Array(a) = value {
                    if let Some(items) = &self.items {
                        for (i, item) in a.iter().enumerate() {
                            items.validate(item).map_err(|e| format!("item {i}: {e}"))?;
                        }
                    }
                    Ok(())
                } else {
                    Err("expected array".into())
                }
            }
            _ => Ok(()),
        }
    }
}

/// A tool descriptor: name, description, input/output schemas, and annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// The tool's unique name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's input.
    pub input_schema: JsonSchema,
    /// JSON Schema for the tool's output.
    pub output_schema: JsonSchema,
    /// Tool annotations (behavioral hints for the agent).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<ToolAnnotation>,
}

/// Behavioral annotations that describe a tool's side effects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolAnnotation {
    /// The tool only reads data (no side effects).
    ReadOnly,
    /// The tool destroys data.
    Destructive,
    /// The tool accesses the network.
    NetworkAccess,
    /// The tool accesses the filesystem.
    FilesystemAccess,
    /// The tool executes arbitrary code.
    CodeExecution,
    /// The tool is idempotent (repeated calls produce the same result).
    Idempotent,
}

/// Trait that every MCP tool must implement.
#[async_trait]
pub trait McpTool: Send + Sync {
    /// Returns the tool's descriptor.
    fn descriptor(&self) -> &ToolDescriptor;
    /// Execute the tool with the given input.
    ///
    /// # Errors
    /// Returns an error if the tool execution fails.
    async fn call(&self, input: Value) -> Result<Value, McpToolError>;
}

/// A router that dispatches tool calls by name.
pub struct ToolRouter {
    tools: HashMap<String, Arc<dyn McpTool>>,
}

impl ToolRouter {
    /// Create a new empty router.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Arc<dyn McpTool>) {
        self.tools
            .insert(tool.descriptor().name.clone(), tool);
    }

    /// List all registered tool descriptors.
    #[must_use]
    pub fn list(&self) -> Vec<&ToolDescriptor> {
        self.tools.values().map(|t| t.descriptor()).collect()
    }

    /// Dispatch a tool call by name, with automatic input/output validation.
    ///
    /// # Errors
    /// Returns `NotFound` if the tool doesn't exist.
    /// Returns `InvalidInput` if the input fails schema validation.
    /// Returns `InvalidOutput` if the output fails schema validation.
    /// Returns `Execution` if the tool itself fails.
    pub async fn dispatch(&self, name: &str, input: Value) -> Result<Value, McpToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| McpToolError::NotFound(name.to_string()))?;
        tool.descriptor()
            .input_schema
            .validate(&input)
            .map_err(McpToolError::InvalidInput)?;
        let output = tool.call(input).await?;
        tool.descriptor()
            .output_schema
            .validate(&output)
            .map_err(McpToolError::InvalidOutput)?;
        Ok(output)
    }

    /// Dispatch multiple tool calls sequentially. Returns a result for each call.
    pub async fn dispatch_parallel(
        &self,
        calls: Vec<(String, Value)>,
    ) -> Vec<Result<Value, McpToolError>> {
        let mut results = Vec::with_capacity(calls.len());
        for (name, input) in calls {
            results.push(self.dispatch(&name, input).await);
        }
        results
    }
}

impl Default for ToolRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// A composition plan: a multi-step tool pipeline with dependency resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionPlan {
    /// The steps in the pipeline.
    pub steps: Vec<CompositionStep>,
}

/// A single step in a composition plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionStep {
    /// Unique step id (referenced by other steps' `depends_on`).
    pub id: String,
    /// The tool to call.
    pub tool: String,
    /// Input JSON (may contain `${step_id.field}` references).
    pub input: Value,
    /// Step ids that must complete before this step can run.
    #[serde(default)]
    pub depends_on: Vec<String>,
}

impl CompositionPlan {
    /// Execute the composition plan, resolving dependencies in topological order.
    ///
    /// # Errors
    /// Returns `Execution` if a cycle is detected or any step fails.
    pub async fn execute(&self, router: &ToolRouter) -> Result<HashMap<String, Value>, McpToolError> {
        let mut outputs: HashMap<String, Value> = HashMap::new();
        let mut completed: HashMap<String, ()> = HashMap::new();
        while completed.len() < self.steps.len() {
            let mut progressed = false;
            for step in &self.steps {
                if completed.contains_key(&step.id) {
                    continue;
                }
                if !step.depends_on.iter().all(|d| completed.contains_key(d)) {
                    continue;
                }
                let resolved = substitute(&step.input, &outputs);
                let output = router.dispatch(&step.tool, resolved).await?;
                outputs.insert(step.id.clone(), output);
                completed.insert(step.id.clone(), ());
                progressed = true;
            }
            if !progressed {
                return Err(McpToolError::Execution("cycle".into()));
            }
        }
        Ok(outputs)
    }
}

/// Recursively substitute `${step_id.field}` references in a JSON value.
fn substitute(value: &Value, outputs: &HashMap<String, Value>) -> Value {
    match value {
        Value::String(s) if s.starts_with("${") && s.ends_with('}') => {
            let path = &s[2..s.len() - 1];
            let mut parts = path.splitn(2, '.');
            let sid = parts.next().unwrap_or("");
            let field = parts.next();
            if let Some(o) = outputs.get(sid) {
                if let Some(f) = field {
                    o.get(f).cloned().unwrap_or(Value::Null)
                } else {
                    o.clone()
                }
            } else {
                Value::Null
            }
        }
        Value::Object(o) => {
            let mut m = serde_json::Map::new();
            for (k, v) in o {
                m.insert(k.clone(), substitute(v, outputs));
            }
            Value::Object(m)
        }
        Value::Array(a) => Value::Array(a.iter().map(|v| substitute(v, outputs)).collect()),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Echo {
        descriptor: ToolDescriptor,
    }
    impl Echo {
        fn new() -> Self {
            Self {
                descriptor: ToolDescriptor {
                    name: "echo".into(),
                    description: "echo".into(),
                    input_schema: JsonSchema::object()
                        .with_property("msg", JsonSchema::string("msg"))
                        .with_required("msg"),
                    output_schema: JsonSchema::object()
                        .with_property("echoed", JsonSchema::string("out"))
                        .with_required("echoed"),
                    annotations: vec![ToolAnnotation::ReadOnly],
                },
            }
        }
    }
    #[async_trait]
    impl McpTool for Echo {
        fn descriptor(&self) -> &ToolDescriptor {
            &self.descriptor
        }
        async fn call(&self, input: Value) -> Result<Value, McpToolError> {
            Ok(json!({"echoed": input["msg"]}))
        }
    }

    #[tokio::test]
    async fn dispatch() {
        let mut r = ToolRouter::new();
        r.register(Arc::new(Echo::new()));
        let o = r.dispatch("echo", json!({"msg":"hi"})).await.unwrap();
        assert_eq!(o["echoed"], "hi");
    }

    #[tokio::test]
    async fn not_found() {
        let r = ToolRouter::new();
        assert!(matches!(r.dispatch("x", json!({})).await, Err(McpToolError::NotFound(_))));
    }

    #[tokio::test]
    async fn parallel() {
        let mut r = ToolRouter::new();
        r.register(Arc::new(Echo::new()));
        let res = r
            .dispatch_parallel(vec![
                ("echo".into(), json!({"msg":"a"})),
                ("echo".into(), json!({"msg":"b"})),
            ])
            .await;
        assert_eq!(res.len(), 2);
        assert!(res[0].is_ok());
    }

    #[tokio::test]
    async fn composition() {
        let mut r = ToolRouter::new();
        r.register(Arc::new(Echo::new()));
        let p = CompositionPlan {
            steps: vec![
                CompositionStep {
                    id: "s1".into(),
                    tool: "echo".into(),
                    input: json!({"msg":"hi"}),
                    depends_on: vec![],
                },
                CompositionStep {
                    id: "s2".into(),
                    tool: "echo".into(),
                    input: json!({"msg":"${s1.echoed}"}),
                    depends_on: vec!["s1".into()],
                },
            ],
        };
        let o = p.execute(&r).await.unwrap();
        assert_eq!(o["s2"]["echoed"], "hi");
    }

    #[test]
    fn schema_validate() {
        let s = JsonSchema::object()
            .with_property("x", JsonSchema::string("x"))
            .with_required("x");
        assert!(s.validate(&json!({"x":"ok"})).is_ok());
        assert!(s.validate(&json!({})).is_err());
    }

    #[test]
    fn schema_wrong_type() {
        let s = JsonSchema::string("x");
        assert!(s.validate(&json!(42)).is_err());
    }

    #[tokio::test]
    async fn invalid_input() {
        let mut r = ToolRouter::new();
        r.register(Arc::new(Echo::new()));
        assert!(r.dispatch("echo", json!({})).await.is_err());
    }

    #[tokio::test]
    async fn composition_cycle() {
        let r = ToolRouter::new();
        let p = CompositionPlan {
            steps: vec![
                CompositionStep {
                    id: "a".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec!["b".into()],
                },
                CompositionStep {
                    id: "b".into(),
                    tool: "x".into(),
                    input: json!({}),
                    depends_on: vec!["a".into()],
                },
            ],
        };
        assert!(p.execute(&r).await.is_err());
    }
}
