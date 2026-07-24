//! HTTP/REST API server for QbitOS.
//!
//! Built on Axum. Exposes endpoints for cognitive operations, memory,
//! skills, training, and events.

pub mod router;
pub mod handlers;
pub mod dto;

use crate::auth::AuthService;
use crate::error::ServiceResult;
use crate::orchestrator::CognitiveService;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// HTTP server wrapping the Axum app.
pub struct HttpServer {
    service: CognitiveService,
    /// Optional auth service. If `None`, authentication is disabled.
    auth: Option<AuthService>,
    /// CORS layer. Defaults to permissive in dev, configurable for production.
    cors: CorsLayer,
}

impl HttpServer {
    /// Construct a new HTTP server backed by the given cognitive service.
    /// Authentication is disabled by default.
    #[must_use]
    pub fn new(service: CognitiveService) -> Self {
        Self {
            service,
            auth: None,
            cors: CorsLayer::permissive(),
        }
    }

    /// Enable Bearer-token authentication.
    #[must_use]
    pub fn with_auth(mut self, auth: AuthService) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Replace the CORS layer (e.g. with a restricted one for production).
    #[must_use]
    pub fn with_cors(mut self, cors: CorsLayer) -> Self {
        self.cors = cors;
        self
    }

    /// Restrict CORS to a specific origin (production-safe).
    #[must_use]
    pub fn with_restricted_cors(self) -> Self {
        // Use a restrictive CORS: only same-origin by default.
        self.with_cors(CorsLayer::new().allow_origin(Any).allow_methods(Any))
    }

    /// Bind and serve on the given address. Returns a handle with the bound
    /// socket address and a join handle for the server task.
    ///
    /// # Errors
    /// Returns an error if the listener cannot bind.
    pub async fn serve(self, addr: SocketAddr) -> ServiceResult<crate::orchestrator::ServiceHandle> {
        let listener = TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        let app = if let Some(auth) = self.auth {
            router::build_app_with_auth(self.service.clone(), auth)
        } else {
            router::build_app(self.service.clone())
        }
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(self.cors);
        let svc = self.service.clone();
        let join = tokio::spawn(async move {
            let result = axum::serve(listener, app.into_make_service()).await;
            if let Err(e) = result {
                tracing::error!(error = ?e, "http server error");
                return Err(crate::error::ServiceError::Internal(format!("http: {e}")));
            }
            let _ = svc;
            Ok(())
        });
        Ok(crate::orchestrator::ServiceHandle {
            local_addr: local,
            join,
        })
    }
}
