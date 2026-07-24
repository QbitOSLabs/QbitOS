//! QbitOS Tools
//!
//! The tool virtual machine: a secure, sandboxed execution environment for
//! tools that the cognitive kernel can invoke. Every tool call goes through:
//!
//! ```text
//! Plan → Permission Check → Execute → Validate → Memory Update
//! ```
//!
//! Built-in tools (filesystem, terminal, http, compiler, etc.) are
//! implemented in Rust. Custom tools can be loaded as WASM modules via
//! [`wasm::WasmToolBackend`] when the `wasm` feature is enabled.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod executor;
pub mod mcp;
pub mod mcp_server;
pub mod permissions;
pub mod sandbox;
pub mod sandbox_config;
pub mod tool_vm;
pub mod tools;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use executor::{ExecutionOutcome, ToolExecutor};
pub use mcp::{
    CompositionPlan, CompositionStep, JsonSchema, McpTool, McpToolError, ToolAnnotation,
    ToolDescriptor, ToolRouter,
};
pub use mcp_server::{
    default_qbitos_tools, McpCallResult, McpContent, McpServer, McpServerError,
    McpToolDescriptor as ServerToolDescriptor, ServerInfo, ToolHandler,
};
pub use permissions::{Permission, PermissionSet, PermissionStore};
pub use sandbox::Sandbox;
pub use tool_vm::{ToolCall, ToolId, ToolKind, ToolOutput, ToolRegistry, ToolVm, ToolVmError};
pub use tools::{
    advanced::{CalculatorTool, FileAnalyzerTool, SentimentAnalyzerTool, EntityExtractorTool, SummarizerTool, TimeTool},
    filesystem::FilesystemTool,
    http::HttpTool,
    terminal::TerminalTool,
};

#[cfg(feature = "wasm")]
pub use wasm::{WasmError, WasmTool, WasmToolRegistry};

#[cfg(feature = "wasm")]
pub use sandbox_config::{HardenedSandbox, MemoryLimiter, SandboxLimits};

/// Re-export of the composite tool module.
pub use tool_module::ToolModule;

mod tool_module;
