//! Terminal tool.
//!
//! Executes shell commands in a controlled subprocess. The tool enforces a
//! timeout and captures stdout, stderr, and exit code.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

use crate::tool_vm::{Tool, ToolKind};

/// The terminal tool.
pub struct TerminalTool {
    /// Default timeout for commands.
    default_timeout: Duration,
    /// Whether to allow shell metacharacters (default: false — requires explicit `shell: true`).
    allow_shell: bool,
}

impl TerminalTool {
    /// Create a new terminal tool with the given default timeout.
    #[must_use]
    pub fn new(default_timeout: Duration) -> Self {
        Self { default_timeout, allow_shell: false }
    }

    /// Allow shell metacharacters. When enabled, commands run via `sh -c`.
    pub fn allow_shell(mut self) -> Self {
        self.allow_shell = true;
        self
    }
}

impl Default for TerminalTool {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecParams {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    env: std::collections::HashMap<String, String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
    #[serde(default)]
    shell: bool,
}

#[async_trait]
impl Tool for TerminalTool {
    fn name(&self) -> &str { "terminal" }
    fn kind(&self) -> ToolKind { ToolKind::Terminal }
    fn action(&self) -> &str { "exec" }
    fn scope(&self, params: &serde_json::Value) -> String {
        params.get("command").and_then(|c| c.as_str()).unwrap_or("*").to_string()
    }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let p: ExecParams = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        if p.shell && !self.allow_shell {
            return Err("shell execution is not allowed".into());
        }

        let mut cmd = if p.shell {
            let mut c = Command::new("sh");
            c.arg("-c").arg(&p.command);
            c
        } else {
            let mut c = Command::new(&p.command);
            for arg in &p.args {
                c.arg(arg);
            }
            c
        };

        if let Some(cwd) = &p.cwd {
            cmd.current_dir(cwd);
        }
        for (k, v) in &p.env {
            cmd.env(k, v);
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let timeout_duration = p.timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(self.default_timeout);

        let output = timeout(timeout_duration, cmd.output())
            .await
            .map_err(|_| format!("command timed out after {:?}", timeout_duration))?
            .map_err(|e| e.to_string())?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let code = output.status.code().unwrap_or(-1);

        Ok(serde_json::json!({
            "command": p.command,
            "exit_code": code,
            "stdout": stdout,
            "stderr": stderr,
            "success": code == 0,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn exec_echo() {
        let tool = TerminalTool::default().allow_shell();
        let params = serde_json::json!({"command": "echo hello", "shell": true});
        let result = tool.execute(&params).await.unwrap();
        assert_eq!(result["stdout"].as_str().unwrap().trim(), "hello");
        assert_eq!(result["exit_code"], 0);
    }

    #[tokio::test]
    async fn shell_disallowed_by_default() {
        let tool = TerminalTool::default();
        let params = serde_json::json!({"command": "echo hello", "shell": true});
        let result = tool.execute(&params).await;
        assert!(result.is_err());
    }
}
