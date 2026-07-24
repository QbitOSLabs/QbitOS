//! Axum router construction.

use crate::auth::AuthService;
use crate::http::handlers::{self, ServiceState};
use crate::orchestrator::CognitiveService;
use axum::middleware as axum_mw;
use axum::routing::{delete, get, post};
use axum::Router;

/// Build the full Axum application with all QbitOS routes.
///
/// Authentication is disabled by default. Use `build_app_with_auth` to
/// require Bearer tokens on non-public routes.
#[must_use]
pub fn build_app(svc: CognitiveService) -> Router {
    build_app_inner(svc, None)
}

/// Build the full Axum application with authentication middleware.
#[must_use]
pub fn build_app_with_auth(svc: CognitiveService, auth: AuthService) -> Router {
    build_app_inner(svc, Some(auth))
}

fn build_app_inner(svc: CognitiveService, auth: Option<AuthService>) -> Router {
    let state = ServiceState::new(svc);
    let routes = Router::new()
        // ─── System ────────────────────────────────────────────────
        .route("/api/v1/health", get(handlers::health))
        .route("/api/v1/version", get(handlers::version))
        .route("/api/v1/stats", get(handlers::stats))
        // ─── Cognitive ─────────────────────────────────────────────
        .route("/api/v1/cognitive/cycle", post(handlers::cognitive_cycle))
        .route("/api/v1/cognitive/stream", post(crate::streaming::cognitive_stream))
        .route("/api/v1/generate/stream", post(crate::streaming::generate_stream))
        // ─── Memory ────────────────────────────────────────────────
        .route("/api/v1/memory", post(handlers::store_memory))
        .route("/api/v1/memory/search", post(handlers::search_memory))
        .route("/api/v1/memory/{id}", delete(handlers::delete_memory))
        // ─── Skills ────────────────────────────────────────────────
        .route("/api/v1/skills", get(handlers::list_skills))
        .route("/api/v1/skills/execute", post(handlers::execute_skill))
        // ─── Training ──────────────────────────────────────────────
        .route("/api/v1/training/step", post(handlers::train_step))
        .route("/api/v1/curriculum", get(handlers::get_curriculum))
        .route("/api/v1/replay/stats", get(handlers::replay_stats))
        .route("/api/v1/meta/strategies", get(handlers::meta_strategies))
        // ─── Events ────────────────────────────────────────────────
        .route("/api/v1/events/subscribe", post(handlers::subscribe_events))
        .route("/api/v1/events/poll", post(handlers::poll_events))
        // ─── Event store ───────────────────────────────────────────
        .route("/api/v1/events/replay", post(handlers::replay_events))
        .route("/api/v1/events/lookup/{id}", get(handlers::lookup_event))
        // ─── WebSocket upgrade ─────────────────────────────────────
        .route("/api/v1/events/ws", get(crate::ws::ws_handler))
        // ─── SSE event stream (unidirectional, HTTP/1.1) ────────────
        .route("/api/v1/events/sse", get(crate::streaming::events_sse))
        // ─── OpenAPI spec ──────────────────────────────────────────
        .route("/api/v1/openapi.json", get(handlers::openapi_json))
        // ─── Prometheus metrics ────────────────────────────────────
        .route("/metrics", get(handlers::metrics))
        // ─── Multilingual ──────────────────────────────────────────
        .route("/api/v1/language/detect", post(handlers::language_detect))
        .route("/api/v1/language/tokenize", post(handlers::language_tokenize))
        .route("/api/v1/language/stem", post(handlers::language_stem))
        // ─── Algorithms ────────────────────────────────────────────
        .route("/api/v1/algo/levenshtein", post(handlers::algo_levenshtein))
        .route("/api/v1/algo/topk", post(handlers::algo_topk))
        .route("/api/v1/algo/bloom", post(handlers::algo_bloom))
        // ─── Fallback ──────────────────────────────────────────────
        .fallback(handlers::fallback)
        .with_state(state.clone());

    let routes = if let Some(auth) = auth {
        routes.layer(axum_mw::from_fn_with_state(
            auth,
            crate::auth::require_auth_middleware,
        ))
    } else {
        routes
    };

    routes
}
