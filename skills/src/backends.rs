//! Tool and inference backends that the skill executor calls into.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Errors emitted by tool backends.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The tool was not found.
    #[error("tool `{0}` not found")]
    NotFound(String),
    /// The tool execution failed.
    #[error("execution error: {0}")]
    Execution(String),
}

/// A backend that executes tool calls.
#[async_trait]
pub trait ToolBackend: Send + Sync {
    /// Execute a tool call with the given name and arguments.
    async fn call(&self, tool: &str, args: &serde_json::Value) -> Result<serde_json::Value, BackendError>;
}

/// A backend that performs inference (LLM calls).
#[async_trait]
pub trait InferBackend: Send + Sync {
    /// Generate a response for the given prompt.
    async fn infer(&self, prompt: &str, context: &serde_json::Value) -> Result<String, BackendError>;
}

/// A no-op tool backend that returns empty results (for testing).
#[derive(Default)]
pub struct NoopToolBackend {
    call_count: AtomicU64,
}

impl NoopToolBackend {
    /// Create a new noop tool backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of calls made to this backend.
    #[must_use]
    pub fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ToolBackend for NoopToolBackend {
    async fn call(&self, tool: &str, _args: &serde_json::Value) -> Result<serde_json::Value, BackendError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(serde_json::json!({"tool": tool, "result": "noop"}))
    }
}

/// A no-op infer backend that returns canned responses (for testing).
#[derive(Default)]
pub struct NoopInferBackend {
    call_count: AtomicU64,
}

impl NoopInferBackend {
    /// Create a new noop infer backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of calls made to this backend.
    #[must_use]
    pub fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl InferBackend for NoopInferBackend {
    async fn infer(&self, prompt: &str, _context: &serde_json::Value) -> Result<String, BackendError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(format!("[noop-infer] {prompt}"))
    }
}

/// An in-memory tool backend for testing (closures).
pub struct InMemoryToolBackend {
    handler: Box<dyn Fn(&str, &serde_json::Value) -> Result<serde_json::Value, BackendError> + Send + Sync>,
}

impl InMemoryToolBackend {
    /// Create a new in-memory tool backend with the given handler.
    #[must_use]
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(&str, &serde_json::Value) -> Result<serde_json::Value, BackendError> + Send + Sync + 'static,
    {
        Self {
            handler: Box::new(handler),
        }
    }
}

impl std::fmt::Debug for InMemoryToolBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryToolBackend").finish()
    }
}

#[async_trait]
impl ToolBackend for InMemoryToolBackend {
    async fn call(&self, tool: &str, args: &serde_json::Value) -> Result<serde_json::Value, BackendError> {
        (self.handler)(tool, args)
    }
}

/// Shared handles.
pub type SharedToolBackend = Arc<dyn ToolBackend>;
pub type SharedInferBackend = Arc<dyn InferBackend>;

/// Configuration for a backend (serialized form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend type name.
    pub backend_type: String,
    /// Backend-specific config.
    pub config: serde_json::Value,
}
