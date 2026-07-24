//! Tool VM: registry + dispatch.
//!
//! The tool VM is the kernel's interface to the outside world. It maintains
//! a registry of tools, dispatches calls to them, and enforces the
//! plan → permission → execute → validate → memory-update pipeline.

use ahash::AHashMap;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

use crate::permissions::PermissionStore;

/// Unique id for a tool invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolId(pub Uuid);

impl ToolId {
    /// Mint a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for ToolId {
    fn default() -> Self {
        Self::new()
    }
}

/// The kind of a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Filesystem tool (read, write, list, etc.).
    Filesystem,
    /// Terminal tool (exec commands).
    Terminal,
    /// HTTP tool (GET, POST, etc.).
    Http,
    /// Database tool (query, mutate).
    Database,
    /// Browser automation tool.
    Browser,
    /// Compiler tool.
    Compiler,
    /// Simulation tool.
    Simulation,
    /// Custom tool (loaded as WASM).
    Custom,
}

/// A tool call: the structured request to invoke a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique id.
    pub id: ToolId,
    /// Tool name (e.g. "filesystem.read", "terminal.exec").
    pub tool: String,
    /// Structured parameters.
    pub parameters: serde_json::Value,
    /// Agent making the call (for permission check).
    pub agent: Uuid,
    /// Whether to validate the output before returning.
    pub validate: bool,
}

impl ToolCall {
    /// Create a new tool call.
    #[must_use]
    pub fn new(tool: impl Into<String>, parameters: serde_json::Value, agent: Uuid) -> Self {
        Self {
            id: ToolId::new(),
            tool: tool.into(),
            parameters,
            agent,
            validate: true,
        }
    }
}

/// A tool's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Tool call id.
    pub call_id: ToolId,
    /// Whether the call succeeded.
    pub success: bool,
    /// Structured output.
    pub output: serde_json::Value,
    /// Error message, if failed.
    pub error: Option<String>,
    /// Wall-clock duration in microseconds.
    pub duration_us: u64,
}

/// Errors emitted by the tool VM.
#[derive(Debug, Error)]
pub enum ToolVmError {
    /// The tool was not found.
    #[error("tool {0} not found")]
    NotFound(String),
    /// The agent does not have permission.
    #[error("permission denied for agent {agent} action {action} target {target}")]
    PermissionDenied {
        /// Agent id.
        agent: Uuid,
        /// Action attempted.
        action: String,
        /// Target of the action.
        target: String,
    },
    /// The tool itself failed.
    #[error("tool {tool} failed: {error}")]
    ToolFailed {
        /// Tool name.
        tool: String,
        /// Error message.
        error: String,
    },
    /// Validation of the output failed.
    #[error("output validation failed: {0}")]
    ValidationFailed(String),
}

/// Trait that every tool must implement.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's name (e.g. "filesystem.read").
    fn name(&self) -> &str;

    /// The tool's kind.
    fn kind(&self) -> ToolKind;

    /// The action this tool performs (used for permission check).
    fn action(&self) -> &str;

    /// The scope extracted from parameters (used for permission check).
    fn scope(&self, parameters: &serde_json::Value) -> String;

    /// The JSON schema for the parameters (optional).
    fn schema(&self) -> Option<serde_json::Value> { None }

    /// Execute the tool.
    ///
    /// # Errors
    /// Returns an error if the tool fails.
    async fn execute(&self, parameters: &serde_json::Value) -> Result<serde_json::Value, String>;
}

/// A registry of tools.
pub struct ToolRegistry {
    tools: RwLock<AHashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { tools: RwLock::new(AHashMap::new()) }
    }

    /// Register a tool.
    pub fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.write().insert(name, tool);
    }

    /// Look up a tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.read().get(name).cloned()
    }

    /// List all registered tools.
    #[must_use]
    pub fn list(&self) -> Vec<String> {
        self.tools.read().keys().cloned().collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The tool VM: dispatches tool calls with permission checks.
pub struct ToolVm {
    registry: Arc<ToolRegistry>,
    permissions: Arc<PermissionStore>,
    /// Per-tool failure tracking for circuit breaker: (count, last_fail_timestamp)
    failure_counts: RwLock<AHashMap<String, (u32, u64)>>,
}

impl ToolVm {
    /// Create a new tool VM.
    #[must_use]
    pub fn new(registry: Arc<ToolRegistry>, permissions: Arc<PermissionStore>) -> Self {
        Self { registry, permissions, failure_counts: RwLock::new(AHashMap::new()) }
    }

    /// Dispatch a tool call.
    ///
    /// Pipeline:
    /// 1. Look up the tool.
    /// 2. Check permissions.
    /// 3. Execute the tool.
    /// 4. Validate the output (if `call.validate`).
    ///
    /// # Errors
    /// Returns [`ToolVmError`] if any step fails.
    pub async fn dispatch(&self, call: ToolCall) -> Result<ToolOutput, ToolVmError> {
        let tool = self.registry.get(&call.tool).ok_or_else(|| ToolVmError::NotFound(call.tool.clone()))?;
        let scope = tool.scope(&call.parameters);
        let action = tool.action();

        // Permission check
        if !self.permissions.check(call.agent, action, &scope) {
            return Err(ToolVmError::PermissionDenied {
                agent: call.agent,
                action: action.to_string(),
                target: scope,
            });
        }

        // Circuit breaker: check if this tool has been failing repeatedly.
        // After 5 consecutive failures, reject calls for 30 seconds.
        {
            let failures = self.failure_counts.read();
            if let Some(&(count, last_fail)) = failures.get(&call.tool) {
                if count >= 5 {
                    let elapsed = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                        .saturating_sub(last_fail);
                    if elapsed < 30 {
                        return Err(ToolVmError::ToolFailed {
                            tool: call.tool.clone(),
                            error: format!("circuit breaker open ({} consecutive failures, retry in {}s)", count, 30 - elapsed),
                        });
                    }
                }
            }
        }

        // Execute
        let start = std::time::Instant::now();
        let result = tool.execute(&call.parameters).await;
        let duration_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);

        match result {
            Ok(output) => {
                // Success: reset failure counter
                self.failure_counts.write().remove(&call.tool);
                // Validate: ensure the output is serializable
                if call.validate && serde_json::to_string(&output).is_err() {
                    return Err(ToolVmError::ValidationFailed("output is not serializable".into()));
                }
                Ok(ToolOutput {
                    call_id: call.id,
                    success: true,
                    output,
                    error: None,
                    duration_us,
                })
            }
            Err(e) => {
                // Failure: increment failure counter
                let mut failures = self.failure_counts.write();
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let entry = failures.entry(call.tool.clone()).or_insert((0, now));
                entry.0 += 1;
                entry.1 = now;
                Err(ToolVmError::ToolFailed { tool: call.tool.clone(), error: e })
            }
        }
    }

    /// Get a reference to the tool registry.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Get a reference to the permission store.
    #[must_use]
    pub fn permissions(&self) -> &PermissionStore {
        &self.permissions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permissions::Permission;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str { "echo" }
        fn kind(&self) -> ToolKind { ToolKind::Custom }
        fn action(&self) -> &str { "echo" }
        fn scope(&self, _: &serde_json::Value) -> String { "*".into() }
        async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
            Ok(params.clone())
        }
    }

    #[tokio::test]
    async fn dispatch_with_permission() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(EchoTool));
        let perms = Arc::new(PermissionStore::new());
        let agent = Uuid::now_v7();
        perms.grant(agent, Permission::new("echo", "*"));
        let vm = ToolVm::new(registry, perms);
        let call = ToolCall::new("echo", serde_json::json!({"hello": "world"}), agent);
        let out = vm.dispatch(call).await.unwrap();
        assert!(out.success);
        assert_eq!(out.output, serde_json::json!({"hello": "world"}));
    }

    #[tokio::test]
    async fn dispatch_denied_without_permission() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(EchoTool));
        let perms = Arc::new(PermissionStore::new());
        let agent = Uuid::now_v7();
        let vm = ToolVm::new(registry, perms);
        let call = ToolCall::new("echo", serde_json::json!({}), agent);
        let err = vm.dispatch(call).await.unwrap_err();
        assert!(matches!(err, ToolVmError::PermissionDenied { .. }));
    }
}
