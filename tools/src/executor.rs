//! Tool executor.
//!
//! High-level orchestration: takes a [`ToolCall`], dispatches it via the
//! [`ToolVm`], records the outcome, and (optionally) writes the result to
//! memory.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use qbitos_memory::MemoryFabric;

use crate::permissions::Permission;
use crate::tool_vm::{ToolCall, ToolVm, ToolVmError};

/// The outcome of an executor run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionOutcome {
    /// The tool call id.
    pub call_id: crate::tool_vm::ToolId,
    /// Whether the call succeeded.
    pub success: bool,
    /// The tool output (if successful).
    pub output: Option<serde_json::Value>,
    /// The error message (if failed).
    pub error: Option<String>,
    /// Wall-clock duration in microseconds.
    pub duration_us: u64,
    /// Whether the outcome was recorded to memory.
    pub recorded_to_memory: bool,
}

/// The tool executor.
pub struct ToolExecutor {
    vm: Arc<ToolVm>,
    memory: Option<Arc<MemoryFabric>>,
}

impl ToolExecutor {
    /// Create a new executor.
    #[must_use]
    pub fn new(vm: Arc<ToolVm>) -> Self {
        Self { vm, memory: None }
    }

    /// Attach a memory fabric so outcomes are recorded.
    pub fn with_memory(mut self, memory: Arc<MemoryFabric>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Execute a tool call.
    ///
    /// # Errors
    /// Propagates [`ToolVmError`].
    pub async fn execute(&self, call: ToolCall) -> Result<ExecutionOutcome, ToolVmError> {
        let start = Instant::now();
        let call_id = call.id;
        let result = self.vm.dispatch(call).await;
        let duration_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);

        let mut outcome = match result {
            Ok(output) => ExecutionOutcome {
                call_id,
                success: true,
                output: Some(output.output.clone()),
                error: None,
                duration_us,
                recorded_to_memory: false,
            },
            Err(e) => ExecutionOutcome {
                call_id,
                success: false,
                output: None,
                error: Some(e.to_string()),
                duration_us,
                recorded_to_memory: false,
            },
        };

        // Record to memory if attached
        if let Some(memory) = &self.memory {
            let content = serde_json::json!({
                "call_id": call_id,
                "success": outcome.success,
                "output": outcome.output,
                "error": outcome.error,
                "duration_us": outcome.duration_us,
            });
            let ep_outcome = if outcome.success {
                qbitos_memory::episodic::Outcome::Success
            } else {
                qbitos_memory::episodic::Outcome::Failure
            };
            memory.record_episode(content, ep_outcome);
            outcome.recorded_to_memory = true;
        }

        Ok(outcome)
    }

    /// Grant a permission to an agent.
    pub fn grant(&self, agent: Uuid, perm: Permission) {
        self.vm.permissions().grant(agent, perm);
    }
}
