//! HTTP tool.
//!
//! Performs GET, POST, PUT, DELETE requests using `reqwest`. Returns the
//! status code, headers, and body as JSON.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::tool_vm::{Tool, ToolKind};

/// The HTTP tool.
pub struct HttpTool {
    client: reqwest::Client,
    default_timeout: Duration,
}

impl HttpTool {
    /// Create a new HTTP tool.
    #[must_use]
    pub fn new(default_timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(default_timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, default_timeout }
    }
}

impl Default for HttpTool {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HttpParams {
    method: String,
    url: String,
    #[serde(default)]
    headers: HashMap<String, String>,
    #[serde(default)]
    body: Option<serde_json::Value>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[async_trait]
impl Tool for HttpTool {
    fn name(&self) -> &str { "http" }
    fn kind(&self) -> ToolKind { ToolKind::Http }
    fn action(&self) -> &str { "network" }
    fn scope(&self, params: &serde_json::Value) -> String {
        params.get("url").and_then(|u| u.as_str()).unwrap_or("*").to_string()
    }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let p: HttpParams = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let method = reqwest::Method::from_bytes(p.method.to_uppercase().as_bytes())
            .map_err(|e| e.to_string())?;

        let timeout_duration = p.timeout_ms.map(Duration::from_millis).unwrap_or(self.default_timeout);
        let mut req = self.client.request(method, &p.url).timeout(timeout_duration);
        for (k, v) in &p.headers {
            req = req.header(k, v);
        }
        if let Some(body) = p.body {
            req = req.json(&body);
        }

        let response = req.send().await.map_err(|e| e.to_string())?;
        let status = response.status().as_u16();
        let headers: HashMap<String, String> = response.headers().iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        let body_text = response.text().await.map_err(|e| e.to_string())?;
        // Try to parse as JSON; fall back to string
        let body_json: serde_json::Value = serde_json::from_str(&body_text).unwrap_or(serde_json::Value::String(body_text));

        Ok(serde_json::json!({
            "url": p.url,
            "status": status,
            "headers": headers,
            "body": body_json,
            "success": (200..300).contains(&status),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn invalid_url_returns_error() {
        let tool = HttpTool::default();
        let params = serde_json::json!({
            "method": "GET",
            "url": "http://localhost:1/nonexistent",
            "timeout_ms": 100,
        });
        let result = tool.execute(&params).await;
        assert!(result.is_err());
    }
}
