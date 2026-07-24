//! QbitOS Services — exposes the cognitive operating system to external clients.
//!
//! Provides:
//!
//! 1. **HTTP/REST API** — RESTful endpoints for cognitive operations, memory,
//!    skills, and training. Built on Axum.
//! 2. **WebSocket real-time** — bidirectional streaming for live cognitive
//!    events and agent thoughts.
//! 3. **JSON-RPC** — method-dispatch protocol for programmatic clients.
//! 4. **OpenAPI spec generation** — auto-generated OpenAPI 3.1 document.
//! 5. **Service orchestration** — wires subsystems into a single
//!    `CognitiveService` handle and manages their lifecycle.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod error;
pub mod orchestrator;
pub mod events;
pub mod event_store;
pub mod http;
pub mod ws;
pub mod rpc;
pub mod openapi;
pub mod middleware;
pub mod auth;
pub mod llm_gateway;
pub mod streaming;

/// gRPC server (feature-gated).
#[cfg(feature = "grpc")]
pub mod grpc;

pub use error::{ServiceError, ServiceResult};
pub use llm_gateway::{GatewayConfig, GatewayError, GatewayLogEntry, LlmBackend, LlmGateway, LlmRequest, LlmResponse, Provider, ProviderConfig, SharedLlmGateway, StubLlmBackend};
pub use orchestrator::{CognitiveService, ServiceConfig, ServiceHandle, ServiceStats};
pub use events::{EventBus, EventSubscription, ServiceEvent, EventId};
pub use event_store::{PersistentEventStore, EventStoreConfig, EventStoreStats};
pub use http::{HttpServer, router::build_app};
pub use ws::{WebSocketServer, WsMessage, WsSession};
pub use rpc::{RpcServer, RpcRequest, RpcResponse, RpcError};
pub use openapi::{OpenApiSpec, generate_openapi};
pub use auth::{AuthService, AuthConfig, AuthPrincipal, AuthError};

#[cfg(feature = "grpc")]
pub use grpc::GrpcServer;

/// Semantic version of the service API.
pub const API_VERSION: &str = "0.1.0";
