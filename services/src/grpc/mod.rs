//! QbitOS gRPC server.
//!
//! High-performance internal RPC over HTTP/2 using tonic + prost. The gRPC
//! API mirrors the HTTP REST surface but uses Protocol Buffers for wire
//! format — useful for microservice-to-microservice communication where
//! JSON parsing overhead is significant.
//!
//! This module is feature-gated behind `grpc` because tonic + prost pull in
//! additional compile-time dependencies. Enable with:
//!
//! ```toml
//! qbitos-services = { features = ["grpc"] }
//! ```
//!
//! # Protocol
//!
//! The Protocol Buffers schema lives in `services/proto/qbitos.proto`. The
//! build script (`services/build.rs`) invokes `tonic-build` to generate Rust
//! types from the schema at compile time.
//!
//! # Server lifecycle
//!
//! ```no_run
//! use qbitos_services::{CognitiveService, ServiceConfig};
//! use qbitos_services::grpc::GrpcServer;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let svc = CognitiveService::new(ServiceConfig::default());
//!     let grpc = GrpcServer::new(svc);
//!     grpc.serve("0.0.0.0:9090").await?;
//!     Ok(())
//! }
//! ```

#![cfg(feature = "grpc")]

use crate::orchestrator::CognitiveService;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::{transport::Server, Request, Response, Status, Streaming};

/// Generated protobuf types (built by build.rs).
pub mod qbitos {
    tonic::include_proto!("qbitos.v1");
}

use qbitos::cognitive_service_server::{
    CognitiveService as CognitiveServiceTrait, CognitiveServiceServer,
};
use qbitos::*;

/// gRPC server wrapping a `CognitiveService` handle.
pub struct GrpcServer {
    svc: CognitiveService,
}

impl GrpcServer {
    /// Construct a new gRPC server.
    #[must_use]
    pub fn new(svc: CognitiveService) -> Self {
        Self { svc }
    }

    /// Start the gRPC server on the given address. Blocks until shutdown.
    ///
    /// # Errors
    /// Returns an error if the server fails to bind or accept connections.
    pub async fn serve(self, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let socket_addr: SocketAddr = addr.parse()?;
        let inner = Arc::new(self);
        let svc = CognitiveServiceServer::from_arc(Arc::clone(&inner));
        Server::builder()
            .add_service(svc)
            .serve(socket_addr)
            .await?;
        Ok(())
    }
}

#[tonic::async_trait]
impl CognitiveServiceTrait for GrpcServer {
    async fn health(
        &self,
        _req: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let stats = self.svc.stats();
        let subsystems: Vec<String> = self
            .svc
            .healthy_subsystems()
            .into_iter()
            .map(String::from)
            .collect();
        let resp = HealthResponse {
            instance_id: self.svc.instance_id(),
            instance_name: self.svc.instance_name(),
            healthy: !self.svc.is_shutting_down(),
            api_version: crate::API_VERSION.into(),
            uptime_secs: stats.uptime_secs as i64,
            subsystems,
        };
        Ok(Response::new(resp))
    }

    async fn version(
        &self,
        _req: Request<VersionRequest>,
    ) -> Result<Response<VersionResponse>, Status> {
        let resp = VersionResponse {
            version: crate::API_VERSION.into(),
            commit: env!("CARGO_PKG_VERSION").into(),
            build_date: chrono::Utc::now().to_rfc3339(),
            rustc_version: "1.85.0".into(),
        };
        Ok(Response::new(resp))
    }

    async fn stats(
        &self,
        _req: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        let s = self.svc.stats();
        let resp = StatsResponse {
            uptime_secs: s.uptime_secs as i64,
            total_requests: s.total_requests,
            active_ws_sessions: s.active_ws_sessions,
            events_emitted: s.events_emitted,
            cpu_utilization: s.cpu_utilization,
            memory_used_bytes: s.memory_used_bytes,
            memory_total_bytes: s.memory_total_bytes,
            running_tasks: s.running_tasks as u32,
            cycles_completed: s.cycles_completed,
        };
        Ok(Response::new(resp))
    }

    async fn cognitive_cycle(
        &self,
        req: Request<CognitiveCycleRequest>,
    ) -> Result<Response<CognitiveCycleResponse>, Status> {
        let inner = req.into_inner();
        let result = self
            .svc
            .cognitive_cycle_json(&inner.prompt, &inner.context_json)
            .map_err(|e| Status::internal(e.to_string()))?;

        let resp = CognitiveCycleResponse {
            response: result.response.clone(),
            result_json: serde_json::to_string(&result).unwrap_or_default(),
            duration_ms: result.duration_ms,
            stages_completed: vec![
                "observe".into(),
                "understand".into(),
                "create_goal".into(),
                "plan".into(),
                "execute".into(),
                "verify".into(),
                "learn".into(),
                "improve".into(),
            ],
            safety_violation: false,
        };
        Ok(Response::new(resp))
    }

    type StreamEventsStream =
        Pin<Box<dyn Stream<Item = Result<ServiceEvent, Status>> + Send + 'static>>;

    async fn stream_events(
        &self,
        req: Request<StreamEventsRequest>,
    ) -> Result<Response<Self::StreamEventsStream>, Status> {
        let inner = req.into_inner();
        let mut sub = self.svc.event_bus().subscribe();
        let max_events = if inner.max_events == 0 {
            usize::MAX
        } else {
            inner.max_events as usize
        };

        let (tx, rx) = mpsc::channel::<Result<ServiceEvent, Status>>(128);
        tokio::spawn(async move {
            let mut emitted = 0usize;
            while let Some(env) = sub.recv().await {
                if emitted >= max_events {
                    break;
                }
                let event = ServiceEvent {
                    seq: env.id.0,
                    timestamp_ms: env.timestamp.timestamp_millis(),
                    event_type: format!("{:?}", env.event),
                    payload_json: serde_json::to_string(&env.event).unwrap_or_default(),
                };
                if tx.send(Ok(event)).await.is_err() {
                    break;
                }
                emitted += 1;
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn store_memory(
        &self,
        req: Request<StoreMemoryRequest>,
    ) -> Result<Response<StoreMemoryResponse>, Status> {
        let inner = req.into_inner();
        let id = self
            .svc
            .memory()
            .store(&inner.content, &inner.memory_type)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(StoreMemoryResponse {
            id: id.to_string(),
            success: true,
        }))
    }

    async fn search_memory(
        &self,
        req: Request<SearchMemoryRequest>,
    ) -> Result<Response<SearchMemoryResponse>, Status> {
        let inner = req.into_inner();
        let results = self
            .svc
            .memory()
            .search(&inner.query, inner.limit as usize)
            .map_err(|e| Status::internal(e.to_string()))?;
        let pb_results = results
            .into_iter()
            .map(|r| MemoryResult {
                id: r.id.to_string(),
                content: r.content,
                score: r.score,
                metadata_json: serde_json::to_string(&r.metadata).unwrap_or_default(),
                created_at_ms: r.created_at.timestamp_millis(),
            })
            .collect();
        Ok(Response::new(SearchMemoryResponse { results: pb_results }))
    }

    async fn delete_memory(
        &self,
        req: Request<DeleteMemoryRequest>,
    ) -> Result<Response<DeleteMemoryResponse>, Status> {
        let inner = req.into_inner();
        let id: uuid::Uuid = inner
            .id
            .parse()
            .map_err(|e| Status::invalid_argument(format!("invalid UUID: {e}")))?;
        let deleted = self
            .svc
            .memory()
            .delete(id)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(DeleteMemoryResponse { deleted }))
    }

    async fn list_skills(
        &self,
        _req: Request<ListSkillsRequest>,
    ) -> Result<Response<ListSkillsResponse>, Status> {
        let skills = self.svc.skills().list();
        let pb_skills = skills
            .into_iter()
            .map(|s| SkillInfo {
                id: s.id.unwrap_or(0),
                name: s.name,
                description: s.description,
                confidence: s.confidence() as f64,
                execution_count: s.execution_count,
                success_count: s.success_count,
            })
            .collect();
        Ok(Response::new(ListSkillsResponse { skills: pb_skills }))
    }

    async fn execute_skill(
        &self,
        req: Request<ExecuteSkillRequest>,
    ) -> Result<Response<ExecuteSkillResponse>, Status> {
        let inner = req.into_inner();
        let input: serde_json::Value =
            serde_json::from_str(&inner.input_json).unwrap_or(serde_json::Value::Null);
        let report = self
            .svc
            .skills_executor()
            .execute(inner.skill_id, input)
            .await;
        Ok(Response::new(ExecuteSkillResponse {
            success: report.status.is_success(),
            output_json: serde_json::to_string(&report.context).unwrap_or_default(),
            duration_ms: report.duration_ms,
            error: if report.status.is_success() {
                String::new()
            } else {
                format!("{:?}", report.status)
            },
        }))
    }

    async fn training_step(
        &self,
        req: Request<TrainingStepRequest>,
    ) -> Result<Response<TrainingStepResponse>, Status> {
        let inner = req.into_inner();
        let result = self
            .svc
            .training_loop()
            .run(inner.steps as usize)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(TrainingStepResponse {
            steps_executed: inner.steps,
            mean_reward: 0.0,
            mean_loss: 0.0,
            duration_ms: 0,
        }))
    }

    async fn get_curriculum(
        &self,
        _req: Request<GetCurriculumRequest>,
    ) -> Result<Response<GetCurriculumResponse>, Status> {
        let curr = self.svc.curriculum();
        let resp = GetCurriculumResponse {
            current_stage: format!("{:?}", curr.current_stage()),
            current_level: curr.current_level() as u32,
            success_rate: curr.success_rate(),
            samples_at_level: curr.samples_at_level(),
            stages: vec![
                "trivial".into(),
                "easy".into(),
                "medium".into(),
                "hard".into(),
                "expert".into(),
            ],
        };
        Ok(Response::new(resp))
    }
}

// Suppress unused-import warning when only some stream types are needed.
#[allow(unused_imports)]
use Streaming as _Streaming;

/// Re-export the tonic transport server for convenience.
pub use tonic::transport::Server as TonicServer;
