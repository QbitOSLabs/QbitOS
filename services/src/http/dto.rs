//! Data Transfer Objects (DTOs) for the HTTP API.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Standard API response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Whether the request succeeded.
    pub ok: bool,
    /// The payload, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    /// Error message, if `ok` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Error code, if `ok` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<i32>,
    /// Server-side request ID.
    pub request_id: Uuid,
    /// Server-side timestamp.
    pub timestamp: DateTime<Utc>,
}

impl<T> ApiResponse<T> {
    /// Build a successful response.
    #[must_use]
    pub fn ok(data: T) -> Self {
        Self {
            ok: true,
            data: Some(data),
            error: None,
            error_code: None,
            request_id: Uuid::new_v4(),
            timestamp: Utc::now(),
        }
    }

    /// Build an error response.
    #[must_use]
    pub fn err(message: impl Into<String>, code: i32) -> Self {
        Self {
            ok: false,
            data: None,
            error: Some(message.into()),
            error_code: Some(code),
            request_id: Uuid::new_v4(),
            timestamp: Utc::now(),
        }
    }
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Instance ID.
    pub instance_id: Uuid,
    /// Instance name.
    pub instance_name: String,
    /// Whether the service is healthy.
    pub healthy: bool,
    /// API version.
    pub api_version: String,
    /// Uptime in seconds.
    pub uptime_seconds: u64,
    /// List of healthy subsystems.
    pub subsystems: Vec<String>,
    /// Service statistics.
    pub stats: Value,
}

/// Request to run a cognitive cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveCycleRequest {
    /// Input observation.
    pub input: String,
    /// Optional modality hint.
    #[serde(default)]
    pub modality: Option<String>,
    /// Whether to emit intermediate thoughts.
    #[serde(default)]
    pub stream_thoughts: bool,
    /// Maximum number of cycles to run.
    #[serde(default = "default_max_cycles")]
    pub max_cycles: usize,
}

fn default_max_cycles() -> usize {
    1
}

/// Response from a cognitive cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveCycleResponse {
    /// The cycle ID.
    pub cycle_id: Uuid,
    /// Final output / action.
    pub output: String,
    /// Phases executed.
    pub phases: Vec<String>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Confidence score (0..1).
    pub confidence: f64,
}

/// Request to store a memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreMemoryRequest {
    /// The memory content (text).
    pub content: String,
    /// Memory layer (L0..L5).
    #[serde(default = "default_layer")]
    pub layer: String,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: Value,
}

fn default_layer() -> String {
    "L1_working".into()
}

/// Memory search request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMemoryRequest {
    /// The query text.
    pub query: String,
    /// Maximum number of results.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Minimum similarity threshold.
    #[serde(default)]
    pub min_score: Option<f64>,
    /// Filter by layer.
    #[serde(default)]
    pub layer: Option<String>,
}

fn default_k() -> usize {
    10
}

/// Memory search result item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    /// The memory ID.
    pub id: String,
    /// The content.
    pub content: String,
    /// Similarity score (0..1).
    pub score: f64,
    /// Memory layer.
    pub layer: String,
    /// Optional metadata.
    pub metadata: Value,
}

/// Request to execute a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteSkillRequest {
    /// The skill ID or name.
    pub skill_id: String,
    /// Input for the skill.
    pub input: Value,
    /// Optional timeout in milliseconds.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

/// Skill execution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillExecutionResult {
    /// The skill ID.
    pub skill_id: String,
    /// Whether the execution succeeded.
    pub success: bool,
    /// Output of the skill.
    pub output: Value,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Training step request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainStepRequest {
    /// Number of training steps.
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Curriculum stage.
    #[serde(default)]
    pub stage: Option<String>,
}

fn default_steps() -> usize {
    1
}

/// Training report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainReportResponse {
    /// Steps executed.
    pub steps_executed: usize,
    /// Final loss / metric.
    pub metric: f64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Whether any improvements were applied.
    pub improvements_applied: usize,
}

/// Event subscription request (HTTP polling).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeRequest {
    /// Optional filter by event kind.
    #[serde(default)]
    pub filter: Option<String>,
}

/// Subscription response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeResponse {
    /// The subscription ID.
    pub subscription_id: u64,
    /// Initial cursor position.
    pub from_event_id: u64,
}

/// Poll request for events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollEventsRequest {
    /// The subscription ID.
    pub subscription_id: u64,
    /// Maximum events to return.
    #[serde(default = "default_poll_batch")]
    pub max: usize,
}

fn default_poll_batch() -> usize {
    100
}
