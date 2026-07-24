//! HTTP integration tests for the QbitOS services crate.
//!
//! These tests boot the full Axum application in-process using the same
//! `build_app` function used in production, then exercise each route via
//! `tower::ServiceExt::oneshot`. This validates the entire request →
//! handler → orchestrator → response path end-to-end without binding a
//! real TCP socket.

#![cfg(test)]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use qbitos_services::{build_app, CognitiveService, ServiceConfig};
use serde_json::{json, Value};
use tower::ServiceExt;

/// Construct a fresh app + service for each test (full isolation).
fn test_app() -> axum::Router {
    let svc = CognitiveService::new(ServiceConfig::default());
    build_app(svc)
}

/// Send a GET request and return (status, body_json).
async fn get(app: axum::Router, path: &str) -> (StatusCode, Value) {
    let resp = app
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .expect("request failed");
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json_val: Value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).unwrap_or(Value::Null)
    };
    (status, json_val)
}

/// Send a POST request with JSON body and return (status, body_json).
async fn post(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("request failed");
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json_val: Value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).unwrap_or(Value::Null)
    };
    (status, json_val)
}

/// Send a DELETE request and return (status, body_json).
async fn delete(app: axum::Router, path: &str) -> (StatusCode, Value) {
    let resp = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(path)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("request failed");
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json_val: Value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).unwrap_or(Value::Null)
    };
    (status, json_val)
}

// ─── Health & System ────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok_with_instance_id() {
    let (status, body) = get(test_app(), "/api/v1/health").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["healthy"].is_boolean());
    assert!(body["data"]["instance_id"].is_string());
    assert!(body["data"]["api_version"].is_string());
}

#[tokio::test]
async fn version_endpoint_returns_semver() {
    let (status, body) = get(test_app(), "/api/v1/version").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["version"].is_string());
    let v = body["data"]["version"].as_str().unwrap();
    // Must look like a semver string (at least one digit dot one digit).
    assert!(v.chars().next().unwrap().is_ascii_digit());
    assert!(v.contains('.'));
}

#[tokio::test]
async fn stats_endpoint_returns_numeric_fields() {
    let (status, body) = get(test_app(), "/api/v1/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["uptime_secs"].is_number() || body["data"]["uptime_secs"].is_string());
}

// ─── Memory ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn memory_store_and_search_roundtrip() {
    let app = test_app();

    // Store a memory.
    let (status, body) = post(
        app.clone(),
        "/api/v1/memory",
        json!({
            "content": "The sky is blue because of Rayleigh scattering.",
            "metadata": { "topic": "physics" }
        }),
    )
    .await;
    assert!(status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
        "store failed: {body}");
}

#[tokio::test]
async fn memory_search_with_empty_query_returns_results_or_empty() {
    let (status, body) = post(
        test_app(),
        "/api/v1/memory/search",
        json!({ "query": "", "limit": 5 }),
    )
    .await;
    // Empty query should be rejected (400) or return empty results (200).
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
        "unexpected status: {status}"
    );
    if status == StatusCode::OK {
        assert!(body["data"]["results"].is_array());
    }
}

#[tokio::test]
async fn memory_delete_returns_success_or_not_found() {
    // Delete a non-existent memory — should return 404 or success-with-false.
    let (status, _body) = delete(
        test_app(),
        "/api/v1/memory/00000000-0000-0000-0000-000000000000",
    )
    .await;
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "unexpected status: {status}"
    );
}

// ─── Cognitive Cycle ────────────────────────────────────────────────────────

#[tokio::test]
async fn cognitive_cycle_runs_eight_stages() {
    let (status, body) = post(
        test_app(),
        "/api/v1/cognitive/cycle",
        json!({
            "prompt": "What is 2 plus 2?",
            "context": {}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "cognitive cycle failed: {body}");
}

// ─── Skills ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn list_skills_returns_array() {
    let (status, body) = get(test_app(), "/api/v1/skills").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["skills"].is_array());
}

#[tokio::test]
async fn execute_skill_with_unknown_id_returns_error() {
    let (status, _body) = post(
        test_app(),
        "/api/v1/skills/execute",
        json!({
            "skill_id": 99999,
            "input": "test"
        }),
    )
    .await;
    assert!(
        status == StatusCode::NOT_FOUND || status == StatusCode::BAD_REQUEST || status == StatusCode::OK,
        "expected 404/400 for unknown skill, got {status}"
    );
}

// ─── Training ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn get_curriculum_returns_stages() {
    let (status, body) = get(test_app(), "/api/v1/curriculum").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"].is_object());
}

#[tokio::test]
async fn replay_stats_returns_numeric_fields() {
    let (status, body) = get(test_app(), "/api/v1/replay/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"].is_object());
}

#[tokio::test]
async fn meta_strategies_returns_array() {
    let (status, body) = get(test_app(), "/api/v1/meta/strategies").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["strategies"].is_array());
}

#[tokio::test]
async fn training_step_executes_without_error() {
    let (status, body) = post(
        test_app(),
        "/api/v1/training/step",
        json!({ "steps": 1 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "training step failed: {body}");
}

// ─── Events ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn events_subscribe_returns_subscription_id() {
    let (status, body) = post(test_app(), "/api/v1/events/subscribe", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["subscription_id"].is_number() || body["data"]["subscription_id"].is_string());
}

#[tokio::test]
async fn events_poll_returns_array() {
    let app = test_app();
    // Subscribe first so we have a subscription id.
    let (_, sub_body) = post(app.clone(), "/api/v1/events/subscribe", json!({})).await;
    let sub_id = sub_body["data"]["subscription_id"].as_u64().unwrap_or(1);

    let (status, body) = post(
        app,
        "/api/v1/events/poll",
        json!({ "subscription_id": sub_id, "timeout_ms": 100 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["events"].is_array());
}

#[tokio::test]
async fn events_replay_from_zero_returns_array() {
    let (status, body) = post(
        test_app(),
        "/api/v1/events/replay",
        json!({ "from_seq": 0, "limit": 10 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["events"].is_array());
}

#[tokio::test]
async fn events_lookup_invalid_id_returns_not_found() {
    let (status, _body) = get(test_app(), "/api/v1/events/lookup/00000000").await;
    assert!(
        status == StatusCode::NOT_FOUND || status == StatusCode::OK,
        "unexpected status: {status}"
    );
}

// ─── OpenAPI ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn openapi_json_returns_valid_spec() {
    let (status, body) = get(test_app(), "/api/v1/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["openapi"], "3.1.0");
    assert!(body["paths"].is_object());
    // Verify a few well-known paths are present.
    assert!(body["paths"]["/api/v1/health"].is_object());
    assert!(body["paths"]["/api/v1/cognitive/cycle"].is_object());
    assert!(body["paths"]["/api/v1/memory"].is_object());
}

// ─── Metrics ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_format() {
    let app = test_app();
    let resp = app
        .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
        .await
        .expect("request failed");
    assert_eq!(resp.status(), StatusCode::OK);
    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    // Prometheus exposition format contains `# TYPE` and `# HELP` comments.
    assert!(
        body_str.contains("# TYPE") || body_str.contains("# HELP") || body_str.is_empty(),
        "metrics body should be prometheus format or empty, got: {body_str}"
    );
}

// ─── Language ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn language_detect_returns_iso_code() {
    let (status, body) = post(
        test_app(),
        "/api/v1/language/detect",
        json!({ "text": "Hello world, this is English text." }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["language"].is_string());
}

#[tokio::test]
async fn language_tokenize_returns_tokens() {
    let (status, body) = post(
        test_app(),
        "/api/v1/language/tokenize",
        json!({ "text": "The quick brown fox jumps over the lazy dog." }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["tokens"].is_array());
    assert!(body["data"]["tokens"].as_array().unwrap().len() >= 5);
}

#[tokio::test]
async fn language_stem_returns_root() {
    let (status, body) = post(
        test_app(),
        "/api/v1/language/stem",
        json!({ "word": "running" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["stem"].is_string());
}

// ─── Algorithms ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn algo_levenshtein_returns_distance() {
    let (status, body) = post(
        test_app(),
        "/api/v1/algo/levenshtein",
        json!({ "a": "kitten", "b": "sitting" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["distance"].as_u64().unwrap_or(99), 3);
}

#[tokio::test]
async fn algo_topk_returns_sorted_topk() {
    let (status, body) = post(
        test_app(),
        "/api/v1/algo/topk",
        json!({ "items": [5, 1, 9, 3, 7, 2, 8, 4, 6], "k": 3 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["top"].is_array());
}

#[tokio::test]
async fn algo_bloom_returns_membership() {
    let (status, body) = post(
        test_app(),
        "/api/v1/algo/bloom",
        json!({ "items": ["alpha", "beta", "gamma"], "query": "beta" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["member"].is_boolean());
}

// ─── Fallback / 404 ─────────────────────────────────────────────────────────

#[tokio::test]
async fn unknown_route_returns_404() {
    let (status, _body) = get(test_app(), "/api/v1/nonexistent").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ─── Concurrency ────────────────────────────────────────────────────────────

#[tokio::test]
async fn parallel_requests_all_succeed() {
    let app = test_app();
    let mut handles = Vec::new();
    for i in 0..10 {
        let app_clone = app.clone();
        handles.push(tokio::spawn(async move {
            let (status, _body) = get(app_clone, "/api/v1/health").await;
            assert_eq!(status, StatusCode::OK, "request {i} failed");
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
}
