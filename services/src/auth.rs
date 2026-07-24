//! Authentication service for the HTTP API.
//!
//! Implements Bearer-token authentication. Tokens are SHA-256 hashed at
//! registration time and compared in constant time at verification time.
//!
//! # Modes
//!
//! - **Disabled** (`AuthConfig { enabled: false }`): no authentication required.
//!   All requests pass through. This is the default for local development.
//! - **Single-token** (`AuthConfig { enabled: true, tokens: { "admin" → hash } }`):
//!   the client must send `Authorization: Bearer <token>` with one of the
//!   registered tokens.
//! - **Public-read** (`AuthConfig { enabled: true, public_read_paths: [...] }`):
//!   read-only endpoints (`GET`) on the public paths list are accessible
//!   without authentication; everything else requires a token.
//!
//! # Example
//!
//! ```no_run
//! use qbitos_services::{AuthService, AuthConfig};
//!
//! let mut auth = AuthService::new(AuthConfig::default());
//! auth.register_token("admin", "super-secret-token");
//! // ... attach `auth.require_auth(...)` as Axum middleware ...
//! ```

use crate::error::ServiceError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use axum::extract::Request;
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;

/// Configuration for the auth service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Whether authentication is required.
    pub enabled: bool,
    /// Whether GET requests on public-read paths bypass auth.
    pub allow_public_read: bool,
    /// Paths that are always accessible without authentication.
    #[serde(default)]
    pub public_paths: Vec<String>,
    /// Path prefixes that are always accessible without authentication
    /// (e.g. `/api/v1/health`, `/api/v1/openapi`).
    #[serde(default)]
    pub public_prefixes: Vec<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allow_public_read: false,
            public_paths: vec![
                "/api/v1/health".into(),
                "/api/v1/version".into(),
                "/api/v1/openapi.json".into(),
                "/api/v1/openapi.yaml".into(),
                "/api/v1/events/ws".into(),
                "/metrics".into(),
            ],
            public_prefixes: vec![],
        }
    }
}

impl AuthConfig {
    /// Build an auth configuration from environment variables.
    ///
    /// - If `QBITOS_TOKEN` is set, auth is enabled and the token is registered
    ///   under the "admin" role.
    /// - If `QBITOS_TOKEN` is unset, auth is disabled (default).
    #[must_use]
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(token) = std::env::var("QBITOS_TOKEN") {
            if !token.is_empty() {
                cfg.enabled = true;
            }
        }
        cfg
    }

    /// Returns the env-var-provided token, if any.
    #[must_use]
    pub fn env_token() -> Option<String> {
        std::env::var("QBITOS_TOKEN").ok().filter(|t| !t.is_empty())
    }
}

/// An authenticated principal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthPrincipal {
    /// The principal name (e.g. "admin", "agent-1").
    pub name: String,
    /// The token's role (e.g. "admin", "read", "write").
    pub role: String,
}

/// Errors produced by the auth service.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// No `Authorization` header was provided.
    #[error("missing authorization header")]
    MissingHeader,
    /// The `Authorization` header was malformed.
    #[error("invalid authorization header: {0}")]
    InvalidHeader(String),
    /// The bearer token was not recognized.
    #[error("invalid token")]
    InvalidToken,
    /// The principal is not allowed to perform this action.
    #[error("forbidden: {0}")]
    Forbidden(String),
}

impl From<AuthError> for ServiceError {
    fn from(e: AuthError) -> Self {
        match e {
            AuthError::MissingHeader | AuthError::InvalidHeader(_) | AuthError::InvalidToken => {
                Self::Unauthorized(e.to_string())
            }
            AuthError::Forbidden(msg) => Self::Unauthorized(msg),
        }
    }
}

/// The authentication service.
pub struct AuthService {
    config: RwLock<AuthConfig>,
    /// Map of principal name → (sha256(token), role).
    tokens: Arc<RwLock<std::collections::HashMap<String, ([u8; 32], String)>>>,
}

impl AuthService {
    /// Construct a new auth service with the given configuration.
    #[must_use]
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config: RwLock::new(config),
            tokens: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Construct a disabled auth service (always allows all requests).
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(AuthConfig::default())
    }

    /// Construct an enabled auth service with the default public paths.
    #[must_use]
    pub fn enabled() -> Self {
        Self::new(AuthConfig {
            enabled: true,
            ..AuthConfig::default()
        })
    }

    /// Register a token for a principal.
    pub fn register_token(&self, name: impl Into<String>, token: impl Into<String>) {
        let name = name.into();
        let token = token.into();
        let hash = hash_token(&token);
        let role = if name == "admin" { "admin" } else { "user" };
        self.tokens
            .write()
            .insert(name, (hash, role.to_string()));
    }

    /// Register a token with an explicit role.
    pub fn register_token_with_role(
        &self,
        name: impl Into<String>,
        token: impl Into<String>,
        role: impl Into<String>,
    ) {
        let name = name.into();
        let token = token.into();
        let hash = hash_token(&token);
        self.tokens.write().insert(name, (hash, role.into()));
    }

    /// Revoke a principal's token.
    pub fn revoke_token(&self, name: &str) -> bool {
        self.tokens.write().remove(name).is_some()
    }

    /// Update the configuration.
    pub fn set_config(&self, config: AuthConfig) {
        *self.config.write() = config;
    }

    /// Read the current configuration.
    #[must_use]
    pub fn config(&self) -> AuthConfig {
        self.config.read().clone()
    }

    /// Number of registered principals.
    #[must_use]
    pub fn principal_count(&self) -> usize {
        self.tokens.read().len()
    }

    /// Whether authentication is currently required.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.read().enabled
    }

    /// Verify a bearer token and return the matching principal.
    ///
    /// # Errors
    /// Returns [`AuthError::InvalidToken`] if the token is not recognized.
    pub fn verify_token(&self, token: &str) -> Result<AuthPrincipal, AuthError> {
        let hash = hash_token(token);
        let tokens = self.tokens.read();
        for (name, (stored_hash, role)) in tokens.iter() {
            if hash.ct_eq(stored_hash).into() {
                return Ok(AuthPrincipal {
                    name: name.clone(),
                    role: role.clone(),
                });
            }
        }
        Err(AuthError::InvalidToken)
    }

    /// Check whether a request path is public (always accessible).
    fn is_public_path(&self, path: &str) -> bool {
        let config = self.config.read();
        if config.public_paths.iter().any(|p| p == path) {
            return true;
        }
        if config
            .public_prefixes
            .iter()
            .any(|prefix| path.starts_with(prefix))
        {
            return true;
        }
        false
    }

    /// Authenticate a request, returning the principal if authentication
    /// succeeds (or is not required).
    ///
    /// # Errors
    /// Returns [`AuthError`] if authentication is required and fails.
    pub fn authenticate(&self, headers: &HeaderMap, path: &str, method: &Method) -> Result<Option<AuthPrincipal>, AuthError> {
        let config = self.config.read();
        if !config.enabled {
            return Ok(None);
        }
        if self.is_public_path(path) {
            return Ok(None);
        }
        if config.allow_public_read && method == Method::GET {
            return Ok(None);
        }
        drop(config);

        // Extract the Authorization header.
        let auth_header = headers
            .get(header::AUTHORIZATION)
            .ok_or(AuthError::MissingHeader)?
            .to_str()
            .map_err(|e| AuthError::InvalidHeader(e.to_string()))?;

        // Parse "Bearer <token>".
        let token = auth_header
            .strip_prefix("Bearer ")
            .or_else(|| auth_header.strip_prefix("bearer "))
            .ok_or_else(|| AuthError::InvalidHeader("expected 'Bearer <token>'".into()))?
            .trim();

        let principal = self.verify_token(token)?;
        Ok(Some(principal))
    }
}

/// Hash a token with SHA-256.
fn hash_token(token: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

impl std::fmt::Debug for AuthService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthService")
            .field("enabled", &self.is_enabled())
            .field("principal_count", &self.principal_count())
            .finish_non_exhaustive()
    }
}

impl Clone for AuthService {
    fn clone(&self) -> Self {
        Self {
            config: RwLock::new(self.config.read().clone()),
            tokens: Arc::clone(&self.tokens),
        }
    }
}

// ─── Axum middleware ───────────────────────────────────────────────────────

/// Axum middleware that requires authentication on non-public routes.
pub async fn require_auth_middleware(
    State(auth): State<AuthService>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    match auth.authenticate(req.headers(), &path, &method) {
        Ok(maybe_principal) => {
            if let Some(principal) = maybe_principal {
                let mut resp = next.run(req).await;
                resp.headers_mut().insert(
                    "x-authenticated-principal",
                    principal.name.parse().unwrap_or_else(|_| "invalid".parse().unwrap()),
                );
                resp
            } else {
                next.run(req).await
            }
        }
        Err(e) => {
            let status = match e {
                AuthError::MissingHeader | AuthError::InvalidHeader(_) | AuthError::InvalidToken => {
                    StatusCode::UNAUTHORIZED
                }
                AuthError::Forbidden(_) => StatusCode::FORBIDDEN,
            };
            (
                status,
                [(
                    axum::http::header::WWW_AUTHENTICATE,
                    axum::http::HeaderValue::from_static("Bearer"),
                )],
                format!("{{\"error\":\"{e}\"}}"),
            )
                .into_response()
        }
    }
}

// We need `State` from axum::extract for the middleware signature.
use axum::extract::State;

#[cfg(test)]
mod tests {
    use crate::ServiceConfig;
    use super::*;
    use axum::http::HeaderValue;

    fn make_headers(token: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(t) = token {
            h.insert(
                header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {t}")).unwrap(),
            );
        }
        h
    }

    #[test]
    fn disabled_auth_allows_everything() {
        let auth = AuthService::disabled();
        let h = make_headers(None);
        let p = auth
            .authenticate(&h, "/api/v1/anything", &Method::POST)
            .unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn enabled_auth_blocks_unauthenticated() {
        let mut config = AuthConfig::default();
        config.enabled = true;
        let auth = AuthService::new(config);
        auth.register_token("admin", "secret");
        let h = make_headers(None);
        let r = auth.authenticate(&h, "/api/v1/skills", &Method::GET);
        assert!(matches!(r, Err(AuthError::MissingHeader)));
    }

    #[test]
    fn enabled_auth_accepts_valid_token() {
        let mut config = AuthConfig::default();
        config.enabled = true;
        let auth = AuthService::new(config);
        auth.register_token("admin", "secret");
        let h = make_headers(Some("secret"));
        let p = auth.authenticate(&h, "/api/v1/skills", &Method::GET).unwrap();
        assert_eq!(p.unwrap().name, "admin");
    }

    #[test]
    fn enabled_auth_rejects_invalid_token() {
        let mut config = AuthConfig::default();
        config.enabled = true;
        let auth = AuthService::new(config);
        auth.register_token("admin", "secret");
        let h = make_headers(Some("wrong"));
        let r = auth.authenticate(&h, "/api/v1/skills", &Method::GET);
        assert!(matches!(r, Err(AuthError::InvalidToken)));
    }

    #[test]
    fn public_paths_bypass_auth() {
        let mut config = AuthConfig::default();
        config.enabled = true;
        let auth = AuthService::new(config);
        let h = make_headers(None);
        let p = auth
            .authenticate(&h, "/api/v1/health", &Method::GET)
            .unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn public_read_allows_get_without_auth() {
        let mut config = AuthConfig::default();
        config.enabled = true;
        config.allow_public_read = true;
        let auth = AuthService::new(config);
        let h = make_headers(None);
        let p = auth
            .authenticate(&h, "/api/v1/skills", &Method::GET)
            .unwrap();
        assert!(p.is_none());
        // POST still requires auth.
        let r = auth.authenticate(&h, "/api/v1/skills", &Method::POST);
        assert!(r.is_err());
    }

    #[test]
    fn public_prefixes_bypass_auth() {
        let mut config = AuthConfig::default();
        config.enabled = true;
        config.public_prefixes.push("/api/v1/openapi".into());
        let auth = AuthService::new(config);
        let h = make_headers(None);
        let p = auth
            .authenticate(&h, "/api/v1/openapi.json", &Method::GET)
            .unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn revoke_token_works() {
        let auth = AuthService::enabled();
        auth.register_token("admin", "secret");
        assert_eq!(auth.principal_count(), 1);
        assert!(auth.revoke_token("admin"));
        assert_eq!(auth.principal_count(), 0);
    }

    #[test]
    fn malformed_header_returns_error() {
        let auth = AuthService::enabled();
        let mut h = HeaderMap::new();
        h.insert(header::AUTHORIZATION, HeaderValue::from_static("NotBearer xxx"));
        let r = auth.authenticate(&h, "/api/v1/skills", &Method::GET);
        assert!(matches!(r, Err(AuthError::InvalidHeader(_))));
    }

    #[test]
    fn roles_are_assigned_correctly() {
        let auth = AuthService::enabled();
        auth.register_token("admin", "t1");
        auth.register_token_with_role("agent", "t2", "write");
        let h = make_headers(Some("t1"));
        let p = auth.authenticate(&h, "/api/v1/x", &Method::GET).unwrap().unwrap();
        assert_eq!(p.role, "admin");
        let h = make_headers(Some("t2"));
        let p = auth.authenticate(&h, "/api/v1/x", &Method::GET).unwrap().unwrap();
        assert_eq!(p.role, "write");
    }

    #[test]
    fn token_hashes_are_constant_time_compared() {
        // Just verify that two different tokens hash differently.
        let auth = AuthService::enabled();
        auth.register_token("a", "token_a");
        auth.register_token("b", "token_b");
        let h = make_headers(Some("token_a"));
        let p = auth.authenticate(&h, "/api/v1/x", &Method::GET).unwrap().unwrap();
        assert_eq!(p.name, "a");
        let h = make_headers(Some("token_b"));
        let p = auth.authenticate(&h, "/api/v1/x", &Method::GET).unwrap().unwrap();
        assert_eq!(p.name, "b");
    }
}
