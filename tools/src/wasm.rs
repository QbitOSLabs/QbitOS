//! WASM tool backend.
//!
//! Loads custom tools as WASM modules and executes them in a Wasmtime
//! sandbox. The WASM module must export a function `execute(params: i32) -> i32`
//! that takes a pointer to a JSON-encoded parameter string and returns a
//! pointer to a JSON-encoded result string. Memory is shared via the
//! module's exported `memory`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use wasmtime::{Engine, Linker, Module, Store};

use crate::sandbox::Sandbox;
use crate::tool_vm::{Tool, ToolKind};

/// Errors emitted by the WASM backend.
#[derive(Debug, Error)]
pub enum WasmError {
    /// Failed to compile the WASM module.
    #[error("failed to compile WASM module: {0}")]
    Compile(String),
    /// Failed to instantiate the module.
    #[error("failed to instantiate WASM module: {0}")]
    Instantiate(String),
    /// The module is missing a required export.
    #[error("missing export: {0}")]
    MissingExport(String),
    /// The module failed during execution.
    #[error("execution failed: {0}")]
    Execution(String),
    /// The module exceeded its resource limits.
    #[error("resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
}

/// A WASM-loaded tool.
pub struct WasmTool {
    name: String,
    engine: Engine,
    module: Module,
    sandbox: Sandbox,
}

impl WasmTool {
    /// Load a WASM module from a file.
    ///
    /// # Errors
    /// Returns [`WasmError`] if the module cannot be compiled.
    pub fn from_file(name: impl Into<String>, path: &PathBuf, sandbox: Sandbox) -> Result<Self, WasmError> {
        let engine = Engine::default();
        let module = Module::from_file(&engine, path).map_err(|e| WasmError::Compile(e.to_string()))?;
        Ok(Self { name: name.into(), engine, module, sandbox })
    }

    /// Load a WASM module from bytes.
    ///
    /// # Errors
    /// Returns [`WasmError`] if the module cannot be compiled.
    pub fn from_bytes(name: impl Into<String>, bytes: &[u8], sandbox: Sandbox) -> Result<Self, WasmError> {
        let engine = Engine::default();
        let module = Module::new(&engine, bytes).map_err(|e| WasmError::Compile(e.to_string()))?;
        Ok(Self { name: name.into(), engine, module, sandbox })
    }
}

#[async_trait]
impl Tool for WasmTool {
    fn name(&self) -> &str { &self.name }
    fn kind(&self) -> ToolKind { ToolKind::Custom }
    fn action(&self) -> &str { "wasm" }
    fn scope(&self, _: &serde_json::Value) -> String { "*".into() }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let result = self.execute_inner(params).await;
        result.map_err(|e| e.to_string())
    }
}

impl WasmTool {
    /// Inner execution returning a typed [`WasmError`].
    async fn execute_inner(&self, params: &serde_json::Value) -> Result<serde_json::Value, WasmError> {
        let mut store = Store::new(&self.engine, ());
        let linker = Linker::new(&self.engine);
        let instance = linker.instantiate(&mut store, &self.module)
            .map_err(|e| WasmError::Instantiate(e.to_string()))?;

        let memory = instance.get_memory(&mut store, "memory")
            .ok_or_else(|| WasmError::MissingExport("memory".into()))?;
        let execute_fn = instance.get_typed_func::<i32, i32>(&mut store, "execute")
            .map_err(|e| WasmError::MissingExport(format!("execute: {e}")))?;

        // Serialize params and write them into WASM memory
        let params_bytes = serde_json::to_vec(params).map_err(|e| WasmError::Execution(e.to_string()))?;
        let params_len = params_bytes.len();
        let memory_size = memory.data_size(&store);
        if params_len + 4 > memory_size {
            return Err(WasmError::ResourceLimitExceeded("insufficient WASM memory for params".into()));
        }
        memory.data_mut(&mut store)[..params_len].copy_from_slice(&params_bytes);

        let len_bytes = (params_len as u32).to_le_bytes();
        memory.data_mut(&mut store)[params_len..params_len + 4].copy_from_slice(&len_bytes);

        let result_ptr = execute_fn.call(&mut store, 0).map_err(|e| WasmError::Execution(e.to_string()))?;

        let data = memory.data(&store);
        if result_ptr as usize + 4 > data.len() {
            return Err(WasmError::Execution("result pointer out of bounds".into()));
        }
        let result_len = u32::from_le_bytes([
            data[result_ptr as usize],
            data[result_ptr as usize + 1],
            data[result_ptr as usize + 2],
            data[result_ptr as usize + 3],
        ]) as usize;
        if result_ptr as usize + 4 + result_len > data.len() {
            return Err(WasmError::Execution("result body out of bounds".into()));
        }
        let result_bytes = &data[result_ptr as usize + 4..result_ptr as usize + 4 + result_len];
        let result: serde_json::Value = serde_json::from_slice(result_bytes)
            .map_err(|e| WasmError::Execution(e.to_string()))?;
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WasmToolManifest {
    name: String,
    version: String,
    description: String,
    parameters_schema: Option<serde_json::Value>,
}

/// A registry of WASM tools, keyed by name.
pub struct WasmToolRegistry {
    tools: parking_lot::RwLock<ahash::AHashMap<String, Arc<WasmTool>>>,
    sandbox: Sandbox,
}

impl WasmToolRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new(sandbox: Sandbox) -> Self {
        Self {
            tools: parking_lot::RwLock::new(ahash::AHashMap::new()),
            sandbox,
        }
    }

    /// Load a WASM tool from a file.
    ///
    /// # Errors
    /// Returns [`WasmError`] if the module cannot be compiled.
    pub fn load_from_file(&self, name: impl Into<String>, path: &PathBuf) -> Result<(), WasmError> {
        let tool = WasmTool::from_file(name, path, self.sandbox.clone())?;
        let name = tool.name.clone();
        self.tools.write().insert(name, Arc::new(tool));
        Ok(())
    }

    /// Load a WASM tool from bytes.
    ///
    /// # Errors
    /// Returns [`WasmError`] if the module cannot be compiled.
    pub fn load_from_bytes(&self, name: impl Into<String>, bytes: &[u8]) -> Result<(), WasmError> {
        let tool = WasmTool::from_bytes(name, bytes, self.sandbox.clone())?;
        let name = tool.name.clone();
        self.tools.write().insert(name, Arc::new(tool));
        Ok(())
    }

    /// Look up a WASM tool by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<WasmTool>> {
        self.tools.read().get(name).cloned()
    }

    /// List all loaded WASM tools.
    #[must_use]
    pub fn list(&self) -> Vec<String> {
        self.tools.read().keys().cloned().collect()
    }
}
