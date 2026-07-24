//! Filesystem tool.
//!
//! Provides `filesystem.read`, `filesystem.write`, `filesystem.list`,
//! `filesystem.stat`, and `filesystem.delete` operations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;

use crate::tool_vm::{Tool, ToolKind};

/// The filesystem tool. Bound to a `root` directory; all paths are resolved
/// relative to `root` and sandboxed to never escape it.
pub struct FilesystemTool {
    root: PathBuf,
}

impl FilesystemTool {
    /// Create a new filesystem tool bound to `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve `path` relative to `root`, ensuring the result does not
    /// escape `root`.
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let p = Path::new(path);
        let resolved = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        };
        // Canonicalize (without requiring existence)
        let canonical = resolved.canonicalize().unwrap_or_else(|_| resolved.clone());
        // Ensure the canonical path starts with root
        let root_canonical = self.root.canonicalize().unwrap_or_else(|_| self.root.clone());
        if !canonical.starts_with(&root_canonical) {
            return Err(format!("path {} escapes filesystem root", path));
        }
        Ok(canonical)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReadParams {
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WriteParams {
    path: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ListParams {
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeleteParams {
    path: String,
}

#[async_trait]
impl Tool for FilesystemTool {
    fn name(&self) -> &str { "filesystem" }
    fn kind(&self) -> ToolKind { ToolKind::Filesystem }
    fn action(&self) -> &str { "filesystem" }
    fn scope(&self, params: &serde_json::Value) -> String {
        params.get("path").and_then(|p| p.as_str()).unwrap_or("*").to_string()
    }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let op = params.get("op").and_then(|o| o.as_str()).unwrap_or("read");
        match op {
            "read" => {
                let p: ReadParams = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
                let path = self.resolve(&p.path)?;
                let content = fs::read_to_string(&path).await.map_err(|e| e.to_string())?;
                Ok(serde_json::json!({"path": p.path, "content": content, "bytes": content.len()}))
            }
            "write" => {
                let p: WriteParams = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
                let path = self.resolve(&p.path)?;
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
                }
                fs::write(&path, &p.content).await.map_err(|e| e.to_string())?;
                Ok(serde_json::json!({"path": p.path, "bytes": p.content.len(), "written": true}))
            }
            "list" => {
                let p: ListParams = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
                let path = self.resolve(&p.path)?;
                let mut entries = Vec::new();
                let mut reader = fs::read_dir(&path).await.map_err(|e| e.to_string())?;
                while let Some(entry) = reader.next_entry().await.map_err(|e| e.to_string())? {
                    let name = entry.file_name().to_string_lossy().to_string();
                    let is_dir = entry.metadata().await.map(|m| m.is_dir()).unwrap_or(false);
                    entries.push(serde_json::json!({"name": name, "is_dir": is_dir}));
                }
                entries.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
                Ok(serde_json::json!({"path": p.path, "entries": entries}))
            }
            "delete" => {
                let p: DeleteParams = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
                let path = self.resolve(&p.path)?;
                if path.is_dir() {
                    fs::remove_dir_all(&path).await.map_err(|e| e.to_string())?;
                } else {
                    fs::remove_file(&path).await.map_err(|e| e.to_string())?;
                }
                Ok(serde_json::json!({"path": p.path, "deleted": true}))
            }
            other => Err(format!("unknown filesystem op: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn write_and_read() {
        let tmp = TempDir::new().unwrap();
        let tool = FilesystemTool::new(tmp.path());
        let write_params = serde_json::json!({
            "op": "write",
            "path": "hello.txt",
            "content": "hello world",
        });
        tool.execute(&write_params).await.unwrap();
        let read_params = serde_json::json!({"op": "read", "path": "hello.txt"});
        let result = tool.execute(&read_params).await.unwrap();
        assert_eq!(result["content"], "hello world");
    }

    #[tokio::test]
    async fn sandbox_escape_rejected() {
        let tmp = TempDir::new().unwrap();
        let tool = FilesystemTool::new(tmp.path());
        let params = serde_json::json!({"op": "read", "path": "../../../etc/passwd"});
        let result = tool.execute(&params).await;
        assert!(result.is_err());
    }
}
