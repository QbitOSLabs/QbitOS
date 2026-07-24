//! JSON-RPC 2.0 server.

use crate::error::{ServiceError, ServiceResult};
use crate::orchestrator::CognitiveService;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    /// JSON-RPC version, must be "2.0".
    pub jsonrpc: String,
    /// Method name.
    pub method: String,
    /// Parameters (positional or named).
    #[serde(default)]
    pub params: Option<Value>,
    /// Request ID.
    #[serde(default)]
    pub id: Option<Value>,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    /// JSON-RPC version.
    pub jsonrpc: String,
    /// Result, if successful.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error, if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    /// Request ID (mirrored from request).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    /// Error code.
    pub code: i32,
    /// Error message.
    pub message: String,
    /// Optional additional data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    /// Parse error.
    #[must_use]
    pub fn parse_error() -> Self {
        Self {
            code: -32700,
            message: "Parse error".into(),
            data: None,
        }
    }

    /// Invalid request.
    #[must_use]
    pub fn invalid_request() -> Self {
        Self {
            code: -32600,
            message: "Invalid Request".into(),
            data: None,
        }
    }

    /// Method not found.
    #[must_use]
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("Method not found: {method}"),
            data: None,
        }
    }

    /// Invalid params.
    #[must_use]
    pub fn invalid_params(msg: &str) -> Self {
        Self {
            code: -32602,
            message: format!("Invalid params: {msg}"),
            data: None,
        }
    }

    /// Internal error.
    #[must_use]
    pub fn internal(msg: &str) -> Self {
        Self {
            code: -32603,
            message: format!("Internal error: {msg}"),
            data: None,
        }
    }
}

impl From<ServiceError> for RpcError {
    fn from(e: ServiceError) -> Self {
        Self {
            code: e.rpc_code(),
            message: e.to_string(),
            data: None,
        }
    }
}

/// A JSON-RPC method handler.
type RpcHandler = Arc<
    dyn Fn(
            CognitiveService,
            Option<Value>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ServiceResult<Value>> + Send>>
        + Send
        + Sync,
>;

/// JSON-RPC server.
pub struct RpcServer {
    service: CognitiveService,
    handlers: Arc<RwLock<HashMap<String, RpcHandler>>>,
}

impl RpcServer {
    /// Create a new RPC server backed by the given cognitive service.
    #[must_use]
    pub fn new(service: CognitiveService) -> Self {
        let srv = Self {
            service,
            handlers: Arc::new(RwLock::new(HashMap::new())),
        };
        srv.register_defaults();
        srv
    }

    /// Register a custom RPC method.
    pub fn register<F, Fut>(&self, method: &str, handler: F)
    where
        F: Fn(CognitiveService, Option<Value>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ServiceResult<Value>> + Send + 'static,
    {
        let boxed: RpcHandler = Arc::new(move |svc, params| Box::pin(handler(svc, params)));
        self.handlers
            .write()
            .unwrap()
            .insert(method.to_string(), boxed);
    }

    fn register_defaults(&self) {
        let svc = self.service.clone();
        let mut handlers_guard = self.handlers.write().unwrap();
        macro_rules! reg {
            ($name:expr, $body:expr) => {
                let handler: Arc<
                    dyn Fn(
                            CognitiveService,
                            Option<Value>,
                        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ServiceResult<Value>> + Send>>
                        + Send
                        + Sync,
                > = Arc::new($body);
                handlers_guard.insert($name.to_string(), handler);
            };
        }
        let svc_ping = svc.clone();
        reg!("ping", move |_svc, _params| {
            let _ = &svc_ping;
            Box::pin(async move { Ok(serde_json::json!({ "pong": true })) })
        });
        let svc_stats = svc.clone();
        reg!("stats", move |_svc, _params| {
            let s = svc_stats.clone();
            Box::pin(async move {
                let stats = s.stats();
                Ok(serde_json::to_value(&stats).unwrap_or(Value::Null))
            })
        });
        let svc_health = svc.clone();
        reg!("health", move |_svc, _params| {
            let s = svc_health.clone();
            Box::pin(async move {
                Ok(serde_json::json!({
                    "instance_id": s.instance_id(),
                    "instance_name": s.instance_name(),
                    "healthy": !s.is_shutting_down(),
                    "subsystems": s.healthy_subsystems(),
                }))
            })
        });
        let svc_cycle = svc.clone();
        reg!("cognitive.cycle", move |_svc, params| {
            let s = svc_cycle.clone();
            Box::pin(async move {
                let input = params
                    .as_ref()
                    .and_then(|p| p.get("input"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let cycle_id = uuid::Uuid::new_v4();
                s.publish(crate::events::ServiceEvent::CognitiveCycleCompleted {
                    cycle_id,
                    phase: "execute".into(),
                    duration_ms: 1,
                    summary: format!("Processed: {input}"),
                });
                Ok(serde_json::json!({
                    "cycle_id": cycle_id,
                    "output": format!("Processed: {input}"),
                }))
            })
        });
        let svc_mem = svc.clone();
        reg!("memory.search", move |_svc, params| {
            let s = svc_mem.clone();
            Box::pin(async move {
                let query = params
                    .as_ref()
                    .and_then(|p| p.get("query"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let _ = s;
                Ok(serde_json::json!({
                    "results": [{
                        "id": format!("mem-{}", uuid::Uuid::new_v4()),
                        "content": format!("Result for: {query}"),
                        "score": 0.91,
                    }]
                }))
            })
        });
        let svc_skills = svc.clone();
        reg!("skills.list", move |_svc, _params| {
            let s = svc_skills.clone();
            Box::pin(async move {
                let skills = s.skills().registry.list();
                let arr: Vec<Value> = skills
                    .iter()
                    .map(|sk| {
                        serde_json::json!({
                            "id": sk.id,
                            "name": sk.name,
                            "description": sk.description,
                            "confidence": sk.confidence(),
                            "success_rate": sk.success_rate(),
                        })
                    })
                    .collect();
                Ok(serde_json::json!({ "skills": arr }))
            })
        });
        let svc_train = svc.clone();
        reg!("training.step", move |_svc, params| {
            let s = svc_train.clone();
            Box::pin(async move {
                let steps = params
                    .as_ref()
                    .and_then(|p| p.get("steps"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                let mut total_reward = 0.0;
                for i in 0..steps {
                    let reward = 0.5 + 0.4 * (i as f64 / steps.max(1) as f64);
                    total_reward += reward;
                    s.replay().add(qbitos_training::Experience::new(
                        serde_json::json!({"step": i}),
                        serde_json::json!({}),
                        reward,
                        serde_json::json!({"step": i + 1}),
                    ));
                    s.curriculum().record_attempt(reward > 0.5);
                    s.curriculum().maybe_advance();
                }
                Ok(serde_json::json!({
                    "steps_executed": steps,
                    "mean_reward": if steps == 0 { 0.0 } else { total_reward / steps as f64 },
                }))
            })
        });
    }

    /// Handle a single RPC request and produce a response.
    pub async fn handle(&self, req: RpcRequest) -> RpcResponse {
        if req.jsonrpc != "2.0" {
            return RpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(RpcError::invalid_request()),
                id: req.id,
            };
        }
        let handler_opt = {
            let handlers = self.handlers.read().unwrap();
            handlers.get(&req.method).cloned()
        };
        let Some(handler) = handler_opt else {
            return RpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(RpcError::method_not_found(&req.method)),
                id: req.id,
            };
        };
        match handler(self.service.clone(), req.params.clone()).await {
            Ok(value) => RpcResponse {
                jsonrpc: "2.0".into(),
                result: Some(value),
                error: None,
                id: req.id,
            },
            Err(e) => RpcResponse {
                jsonrpc: "2.0".into(),
                result: None,
                error: Some(RpcError::from(e)),
                id: req.id,
            },
        }
    }

    /// Handle a batch of RPC requests.
    pub async fn handle_batch(&self, batch: Vec<RpcRequest>) -> Vec<RpcResponse> {
        let mut out = Vec::with_capacity(batch.len());
        for req in batch {
            out.push(self.handle(req).await);
        }
        out
    }

    /// Parse a single JSON-RPC request from a JSON string and handle it.
    pub async fn handle_raw(&self, raw: &str) -> ServiceResult<String> {
        if raw.trim_start().starts_with('[') {
            let batch: Vec<RpcRequest> = serde_json::from_str(raw)
                .map_err(|_| ServiceError::BadRequest("invalid batch request".into()))?;
            let responses = self.handle_batch(batch).await;
            Ok(serde_json::to_string(&responses)?)
        } else {
            let req: RpcRequest = serde_json::from_str(raw)
                .map_err(|_| ServiceError::BadRequest("invalid request".into()))?;
            let resp = self.handle(req).await;
            Ok(serde_json::to_string(&resp)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ServiceConfig;
    use super::*;

    #[tokio::test]
    async fn ping_pong() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let rpc = RpcServer::new(svc);
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            method: "ping".into(),
            params: None,
            id: Some(Value::from(1)),
        };
        let resp = rpc.handle(req).await;
        assert!(resp.error.is_none());
        assert_eq!(resp.id, Some(Value::from(1)));
        assert_eq!(resp.result.unwrap()["pong"], true);
    }

    #[tokio::test]
    async fn unknown_method_returns_error() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let rpc = RpcServer::new(svc);
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            method: "nonexistent.method".into(),
            params: None,
            id: Some(Value::from(2)),
        };
        let resp = rpc.handle(req).await;
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
    }

    #[tokio::test]
    async fn batch_handling() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let rpc = RpcServer::new(svc);
        let batch = vec![
            RpcRequest {
                jsonrpc: "2.0".into(),
                method: "ping".into(),
                params: None,
                id: Some(Value::from(1)),
            },
            RpcRequest {
                jsonrpc: "2.0".into(),
                method: "stats".into(),
                params: None,
                id: Some(Value::from(2)),
            },
        ];
        let responses = rpc.handle_batch(batch).await;
        assert_eq!(responses.len(), 2);
        assert!(responses.iter().all(|r| r.error.is_none()));
    }

    #[tokio::test]
    async fn invalid_jsonrpc_version() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let rpc = RpcServer::new(svc);
        let req = RpcRequest {
            jsonrpc: "1.0".into(),
            method: "ping".into(),
            params: None,
            id: Some(Value::from(1)),
        };
        let resp = rpc.handle(req).await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32600);
    }

    #[tokio::test]
    async fn handle_raw_single_request() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let rpc = RpcServer::new(svc);
        let raw = r#"{"jsonrpc":"2.0","method":"ping","id":1}"#;
        let out = rpc.handle_raw(raw).await.unwrap();
        assert!(out.contains("pong"));
    }

    #[tokio::test]
    async fn handle_raw_batch_request() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let rpc = RpcServer::new(svc);
        let raw = r#"[{"jsonrpc":"2.0","method":"ping","id":1},{"jsonrpc":"2.0","method":"ping","id":2}]"#;
        let out = rpc.handle_raw(raw).await.unwrap();
        assert!(out.starts_with('['));
    }

    #[tokio::test]
    async fn cognitive_cycle_rpc_works() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let rpc = RpcServer::new(svc);
        let req = RpcRequest {
            jsonrpc: "2.0".into(),
            method: "cognitive.cycle".into(),
            params: Some(serde_json::json!({"input": "hello"})),
            id: Some(Value::from(1)),
        };
        let resp = rpc.handle(req).await;
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert!(result["output"].as_str().unwrap().contains("hello"));
    }
}
