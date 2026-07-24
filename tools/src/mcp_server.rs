//! MCP (Model Context Protocol) Server Mode.
//!
//! Exposes QbitOS as an MCP-compatible tool server. Other AI agents (Claude,
//! Cursor, Continue, etc.) can call QbitOS capabilities — memory, skills,
//! cognitive cycles — through the standard MCP tool-call protocol.
//!
//! # Transport
//!
//! The server speaks JSON-RPC 2.0 over:
//!   - stdio (for local-agent integration — Claude Desktop, etc.)
//!   - TCP (for remote-agent integration)
//!
//! # Protocol
//!
//! Implements the MCP 2026-07-28 spec's `tools/list` and `tools/call` methods.
//! Each QbitOS capability is exposed as one tool with a JSON Schema for input
//! validation and structured output.
//!
//! # Example stdio session
//!
//! ```text
//! → {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2026-07-28"}}
//! ← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2026-07-28","capabilities":{"tools":{}}}}
//! → {"jsonrpc":"2.0","id":2,"method":"tools/list"}
//! ← {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"memory_search",...}]}}
//! → {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"sky","limit":5}}}
//! ← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"..."}]}}
//! ```

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]

use crate::mcp::{JsonSchema, McpToolError, ToolAnnotation};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::Arc;
use thiserror::Error;

/// Errors emitted by the MCP server.
#[derive(Debug, Error)]
pub enum McpServerError {
    /// JSON-RPC parse error.
    #[error("parse error: {0}")]
    Parse(String),
    /// Method not found.
    #[error("method not found: {0}")]
    MethodNotFound(String),
    /// Tool not found.
    #[error("tool not found: {0}")]
    ToolNotFound(String),
    /// Tool execution failed.
    #[error("tool execution failed: {0}")]
    Execution(String),
    /// I/O error on the transport.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// JSON-RPC 2.0 request.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    fn ok(id: Value, result: Value) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
    }
    fn err(id: Value, code: i32, message: String) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: None, error: Some(JsonRpcError { code, message, data: None }) }
    }
}

/// MCP tool descriptor (server-side, what we expose to clients).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: JsonSchema,
    #[serde(default)]
    pub annotations: Vec<ToolAnnotation>,
}

/// MCP tool content item (returned in `tools/call` responses).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpContent {
    Text { text: String },
    Image { data: String, mime_type: String },
    Resource { uri: String, mime_type: Option<String> },
}

/// MCP tool call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpCallResult {
    pub content: Vec<McpContent>,
    #[serde(default)]
    pub is_error: bool,
}

/// A handler function for an MCP tool call.
pub type ToolHandler = Arc<dyn Fn(Value) -> Result<McpCallResult, McpServerError> + Send + Sync>;

/// The MCP server. Holds a registry of tools that external agents can call.
pub struct McpServer {
    server_info: ServerInfo,
    tools: HashMap<String, McpToolDescriptor>,
    handlers: HashMap<String, ToolHandler>,
}

/// Server identity sent in the `initialize` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new(ServerInfo {
            name: "qbitos".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        })
    }
}

impl McpServer {
    /// Create a new MCP server with the given identity.
    #[must_use]
    pub fn new(server_info: ServerInfo) -> Self {
        Self {
            server_info,
            tools: HashMap::new(),
            handlers: HashMap::new(),
        }
    }

    /// Register a tool with its handler.
    pub fn register(&mut self, descriptor: McpToolDescriptor, handler: ToolHandler) {
        self.handlers.insert(descriptor.name.clone(), handler);
        self.tools.insert(descriptor.name.clone(), descriptor);
    }

    /// Handle a single JSON-RPC request and produce a response.
    pub fn handle_request(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => {
                let result = json!({
                    "protocolVersion": "2026-07-28",
                    "capabilities": {
                        "tools": { "listChanged": false }
                    },
                    "serverInfo": self.server_info,
                });
                JsonRpcResponse::ok(req.id.clone(), result)
            }
            "tools/list" => {
                let tools: Vec<&McpToolDescriptor> = self.tools.values().collect();
                JsonRpcResponse::ok(req.id.clone(), json!({ "tools": tools }))
            }
            "tools/call" => {
                let name = req.params["name"].as_str().unwrap_or("");
                let args = req.params["arguments"].clone();
                match self.handlers.get(name) {
                    Some(handler) => match handler(args) {
                        Ok(result) => JsonRpcResponse::ok(req.id.clone(), serde_json::to_value(result).unwrap_or_default()),
                        Err(e) => JsonRpcResponse::err(req.id.clone(), -32603, e.to_string()),
                    },
                    None => JsonRpcResponse::err(req.id.clone(), -32602, format!("tool not found: {name}")),
                }
            }
            "ping" => JsonRpcResponse::ok(req.id.clone(), json!({})),
            _ => JsonRpcResponse::err(req.id.clone(), -32601, format!("method not found: {}", req.method)),
        }
    }

    /// Run the server over stdio (for Claude Desktop / local agent integration).
    ///
    /// Reads JSON-RPC requests line-by-line from stdin, writes responses to
    /// stdout. Blocks until stdin is closed.
    pub fn serve_stdio(self) -> Result<(), McpServerError> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let server = Arc::new(self);

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let req: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let resp = JsonRpcResponse::err(Value::Null, -32700, format!("parse error: {e}"));
                    writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap())?;
                    stdout.flush()?;
                    continue;
                }
            };
            let resp = server.handle_request(&req);
            writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap())?;
            stdout.flush()?;
        }
        Ok(())
    }

    /// Run the server over TCP (for remote-agent integration).
    ///
    /// Each accepted connection is handled in its own tokio task. The protocol
    /// is one JSON-RPC message per line (newline-delimited JSON).
    pub async fn serve_tcp(self, addr: &str) -> Result<(), McpServerError> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let server = Arc::new(self);
        tracing::info!("MCP server listening on {addr}");

        loop {
            let (stream, _peer) = listener.accept().await?;
            let server = Arc::clone(&server);
            tokio::spawn(async move {
                let (read_half, mut write_half) = stream.into_split();
                let mut reader = BufReader::new(read_half).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let req: JsonRpcRequest = match serde_json::from_str(&line) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    let resp = server.handle_request(&req);
                    let resp_str = serde_json::to_string(&resp).unwrap_or_default();
                    if write_half.write_all(resp_str.as_bytes()).await.is_err() {
                        break;
                    }
                    if write_half.write_all(b"\n").await.is_err() {
                        break;
                    }
                }
            });
        }
    }
}

/// Build the default set of QbitOS MCP tools (memory + cognitive + skills).
///
/// The handlers return closures that call into the provided `CognitiveService`.
pub fn default_qbitos_tools() -> Vec<(McpToolDescriptor, ToolHandler)> {
    vec![
        // ─── memory_search ─────────────────────────────────────────────────
        (
            McpToolDescriptor {
                name: "memory_search".into(),
                description: "Search the QbitOS memory fabric for relevant memories. Returns ranked results by semantic similarity.".into(),
                input_schema: JsonSchema::object()
                    .with_property("query", JsonSchema::string("The search query"))
                    .with_property("limit", JsonSchema::string("Maximum results to return (default 5)"))
                    .with_required("query"),
                annotations: vec![ToolAnnotation::ReadOnly],
            },
            Arc::new(|_args: Value| {
                // In a real deployment this would call CognitiveService::memory().search().
                // For the standalone MCP server we return a placeholder instructing the
                // operator to wire up a CognitiveService instance.
                Ok(McpCallResult {
                    content: vec![McpContent::Text { text: "memory_search: connect a CognitiveService to enable".into() }],
                    is_error: false,
                })
            }),
        ),
        // ─── memory_store ──────────────────────────────────────────────────
        (
            McpToolDescriptor {
                name: "memory_store".into(),
                description: "Store a new memory in the QbitOS memory fabric.".into(),
                input_schema: JsonSchema::object()
                    .with_property("content", JsonSchema::string("The memory content to store"))
                    .with_property("memory_type", JsonSchema::string("Memory layer: sensory|working|episodic|semantic|procedural|meta"))
                    .with_required("content"),
                annotations: vec![],
            },
            Arc::new(|_args: Value| {
                Ok(McpCallResult {
                    content: vec![McpContent::Text { text: "memory_store: connect a CognitiveService to enable".into() }],
                    is_error: false,
                })
            }),
        ),
        // ─── cognitive_cycle ───────────────────────────────────────────────
        (
            McpToolDescriptor {
                name: "cognitive_cycle".into(),
                description: "Run a full QbitOS cognitive cycle (Observe → Improve) on the given prompt. Returns the agent's response.".into(),
                input_schema: JsonSchema::object()
                    .with_property("prompt", JsonSchema::string("The input prompt"))
                    .with_property("context", JsonSchema::object())
                    .with_required("prompt"),
                annotations: vec![ToolAnnotation::ReadOnly],
            },
            Arc::new(|_args: Value| {
                Ok(McpCallResult {
                    content: vec![McpContent::Text { text: "cognitive_cycle: connect a CognitiveService to enable".into() }],
                    is_error: false,
                })
            }),
        ),
        // ─── list_skills ───────────────────────────────────────────────────
        (
            McpToolDescriptor {
                name: "list_skills".into(),
                description: "List all registered QbitOS skills with their confidence scores and execution counts.".into(),
                input_schema: JsonSchema::object(),
                annotations: vec![ToolAnnotation::ReadOnly],
            },
            Arc::new(|_args: Value| {
                Ok(McpCallResult {
                    content: vec![McpContent::Text { text: "list_skills: connect a SkillRegistry to enable".into() }],
                    is_error: false,
                })
            }),
        ),
        // ─── execute_skill ─────────────────────────────────────────────────
        (
            McpToolDescriptor {
                name: "execute_skill".into(),
                description: "Execute a registered QbitOS skill by its ID with the given input.".into(),
                input_schema: JsonSchema::object()
                    .with_property("skill_id", JsonSchema::string("The skill ID"))
                    .with_property("input", JsonSchema::object())
                    .with_required("skill_id"),
                annotations: vec![],
            },
            Arc::new(|_args: Value| {
                Ok(McpCallResult {
                    content: vec![McpContent::Text { text: "execute_skill: connect a SkillExecutor to enable".into() }],
                    is_error: false,
                })
            }),
        ),
        // ─── language_detect ───────────────────────────────────────────────
        (
            McpToolDescriptor {
                name: "language_detect".into(),
                description: "Detect the language of a text. Returns an ISO 639-1 code.".into(),
                input_schema: JsonSchema::object()
                    .with_property("text", JsonSchema::string("The text to analyze"))
                    .with_required("text"),
                annotations: vec![ToolAnnotation::ReadOnly],
            },
            Arc::new(|args: Value| {
                let text = args["text"].as_str().unwrap_or("");
                let lang = qbitos_multilingual::detect_language(text);
                let payload = serde_json::to_string(&json!({"language": lang.to_string()}))
                    .unwrap_or_else(|_| r#"{"language":"unknown"}"#.into());
                Ok(McpCallResult {
                    content: vec![McpContent::Text { text: payload }],
                    is_error: false,
                })
            }),
        ),
    ]
}

/// Convert a `serde_json::Error` into an `McpServerError` (used by `?` in handlers).
impl From<serde_json::Error> for McpServerError {
    fn from(e: serde_json::Error) -> Self {
        Self::Parse(e.to_string())
    }
}

/// Convert an `McpToolError` into an `McpServerError`.
impl From<McpToolError> for McpServerError {
    fn from(e: McpToolError) -> Self {
        Self::Execution(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_initialize() {
        let server = McpServer::default();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Value::from(1),
            method: "initialize".into(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2026-07-28");
        assert_eq!(result["serverInfo"]["name"], "qbitos");
    }

    #[test]
    fn handle_tools_list() {
        let mut server = McpServer::default();
        for (desc, handler) in default_qbitos_tools() {
            server.register(desc, handler);
        }
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Value::from(2),
            method: "tools/list".into(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        assert!(tools.len() >= 5);
        assert!(tools.iter().any(|t| t["name"] == "memory_search"));
        assert!(tools.iter().any(|t| t["name"] == "cognitive_cycle"));
    }

    #[test]
    fn handle_tools_call_unknown_returns_error() {
        let server = McpServer::default();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Value::from(3),
            method: "tools/call".into(),
            params: json!({"name": "nonexistent", "arguments": {}}),
        };
        let resp = server.handle_request(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[test]
    fn handle_unknown_method() {
        let server = McpServer::default();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Value::from(4),
            method: "foo/bar".into(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn handle_ping() {
        let server = McpServer::default();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Value::from(5),
            method: "ping".into(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
    }

    #[test]
    fn language_detect_tool_works() {
        let mut server = McpServer::default();
        for (desc, handler) in default_qbitos_tools() {
            server.register(desc, handler);
        }
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Value::from(6),
            method: "tools/call".into(),
            params: json!({"name": "language_detect", "arguments": {"text": "Hello world"}}),
        };
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
    }
}
