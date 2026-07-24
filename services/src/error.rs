//! Error types for the services crate.

use thiserror::Error;

/// Result alias used throughout the services crate.
pub type ServiceResult<T> = Result<T, ServiceError>;

/// All errors that can be produced by the service layer.
#[derive(Debug, Error)]
pub enum ServiceError {
    /// HTTP / Axum-level error.
    #[error("http error: {0}")]
    Http(#[from] axum::Error),

    /// Hyper transport error.
    #[error("hyper error: {0}")]
    Hyper(#[from] hyper::Error),

    /// JSON serialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid request payload or parameters.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// The requested resource was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Caller is not authorized to perform this action.
    #[error("unauthorized: {0}")]
    Unauthorized(String),

    /// Request conflicts with current state.
    #[error("conflict: {0}")]
    Conflict(String),

    /// Rate limit exceeded.
    #[error("rate limited: retry after {retry_after_ms}ms")]
    RateLimited {
        /// Milliseconds to wait before retrying.
        retry_after_ms: u64 },

    /// An upstream QbitOS subsystem returned an error.
    #[error("subsystem '{subsystem}' error: {message}")]
    Subsystem {
        /// Name of the subsystem that failed.
        subsystem: &'static str,
        /// Error message.
        message: String },

    /// The cognitive service is not yet initialized.
    #[error("service not initialized")]
    NotInitialized,

    /// Internal server error (catch-all).
    #[error("internal error: {0}")]
    Internal(String),

    /// Service is shutting down.
    #[error("service shutting down")]
    ShuttingDown,

    /// WebSocket protocol error.
    #[error("websocket error: {0}")]
    WebSocket(String),

    /// Timeout while waiting for a subsystem response.
    #[error("timeout after {0}ms")]
    Timeout(u64),

    /// A required capability is missing on this server.
    #[error("unsupported capability: {0}")]
    UnsupportedCapability(&'static str),
}

impl ServiceError {
    /// Map this error to an HTTP status code.
    #[must_use]
    pub fn status_code(&self) -> u16 {
        match self {
            Self::BadRequest(_) => 400,
            Self::Unauthorized(_) => 401,
            Self::NotFound(_) => 404,
            Self::Conflict(_) => 409,
            Self::RateLimited { .. } => 429,
            Self::Timeout(_) => 504,
            Self::ShuttingDown => 503,
            Self::UnsupportedCapability(_) => 501,
            Self::NotInitialized => 503,
            Self::Subsystem { .. } | Self::Internal(_) => 500,
            Self::Http(_) | Self::Hyper(_) | Self::Json(_) | Self::WebSocket(_) => 500,
        }
    }

    /// Convert into a JSON-RPC error code (per JSON-RPC 2.0 spec).
    #[must_use]
    pub fn rpc_code(&self) -> i32 {
        match self {
            Self::BadRequest(_) => -32602,
            Self::NotFound(_) => -32001,
            Self::Unauthorized(_) => -32002,
            Self::Conflict(_) => -32003,
            Self::RateLimited { .. } => -32004,
            Self::Timeout(_) => -32005,
            Self::ShuttingDown => -32006,
            Self::UnsupportedCapability(_) => -32007,
            Self::NotInitialized => -32008,
            Self::Subsystem { .. } | Self::Internal(_) => -32603,
            Self::Http(_) | Self::Hyper(_) | Self::Json(_) | Self::WebSocket(_) => -32603,
        }
    }
}

impl From<std::io::Error> for ServiceError {
    fn from(e: std::io::Error) -> Self {
        Self::Internal(format!("io: {e}"))
    }
}

impl From<tokio::task::JoinError> for ServiceError {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::Internal(format!("join: {e}"))
    }
}

impl From<anyhow::Error> for ServiceError {
    fn from(e: anyhow::Error) -> Self {
        Self::Internal(format!("{e:#}"))
    }
}
