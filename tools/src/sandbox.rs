//! Sandbox.
//!
//! Configuration for sandboxing tool execution. When the `wasm` feature is
//! enabled, custom tools run inside a Wasmtime sandbox. Built-in Rust tools
//! are trusted and run in-process.

use serde::{Deserialize, Serialize};

/// Sandbox configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sandbox {
    /// Whether to enable WASM sandboxing for custom tools.
    pub enable_wasm: bool,
    /// Maximum memory (in bytes) a WASM module can use.
    pub max_memory_bytes: u64,
    /// Maximum stack size (in bytes).
    pub max_stack_bytes: u64,
    /// Maximum execution time (in milliseconds).
    pub max_execution_ms: u64,
    /// Whether to allow network access from WASM modules.
    pub allow_network: bool,
    /// Whether to allow filesystem access from WASM modules.
    pub allow_filesystem: bool,
    /// Whether to allow environment variable access from WASM modules.
    pub allow_env: bool,
}

impl Default for Sandbox {
    fn default() -> Self {
        Self {
            enable_wasm: true,
            max_memory_bytes: 64 * 1024 * 1024,
            max_stack_bytes: 1024 * 1024,
            max_execution_ms: 5_000,
            allow_network: false,
            allow_filesystem: false,
            allow_env: false,
        }
    }
}

impl Sandbox {
    /// Create a strict sandbox (no network, no filesystem, no env).
    #[must_use]
    pub fn strict() -> Self {
        Self::default()
    }

    /// Create a permissive sandbox (allows everything).
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            enable_wasm: true,
            max_memory_bytes: 256 * 1024 * 1024,
            max_stack_bytes: 8 * 1024 * 1024,
            max_execution_ms: 30_000,
            allow_network: true,
            allow_filesystem: true,
            allow_env: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_strict() {
        let s = Sandbox::default();
        assert!(s.enable_wasm);
        assert!(!s.allow_network);
        assert!(!s.allow_filesystem);
        assert!(!s.allow_env);
        assert_eq!(s.max_memory_bytes, 64 * 1024 * 1024);
        assert_eq!(s.max_execution_ms, 5_000);
    }

    #[test]
    fn strict_disallows_everything() {
        let s = Sandbox::strict();
        assert!(!s.allow_network);
        assert!(!s.allow_filesystem);
        assert!(!s.allow_env);
    }

    #[test]
    fn permissive_allows_everything() {
        let s = Sandbox::permissive();
        assert!(s.allow_network);
        assert!(s.allow_filesystem);
        assert!(s.allow_env);
        assert!(s.max_memory_bytes > 64 * 1024 * 1024);
        assert!(s.max_execution_ms > 5_000);
    }

    #[test]
    fn serialization_roundtrip() {
        let s = Sandbox::permissive();
        let json = serde_json::to_string(&s).unwrap();
        let back: Sandbox = serde_json::from_str(&json).unwrap();
        assert_eq!(back.allow_network, s.allow_network);
        assert_eq!(back.max_memory_bytes, s.max_memory_bytes);
    }
}
