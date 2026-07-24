//! Server-Sent Events (SSE) streaming endpoints.
//!
//! Provides chunked streaming responses for long-running LLM generations.
//! Each chunk is sent as an SSE event with a JSON payload describing the
//! token(s) produced, latency, and finish status.
//!
//! # Endpoints
//!
//! - `POST /api/v1/cognitive/stream` — SSE stream of cognitive cycle events
//! - `POST /api/v1/generate/stream`  — SSE stream of LLM tokens
//!
//! # Wire format
//!
//! Each event is one SSE message:
//! ```text
//! event: token
//! data: {"text":"Hello","tokens_generated":1,"duration_ms":12}
//!
//! event: done
//! data: {"finish_reason":"stop","total_tokens":42,"total_duration_ms":1234}
//! ```

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]

use crate::http::handlers::ServiceState;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::time::Duration;

/// `POST /api/v1/cognitive/stream` — stream cognitive-cycle progress as SSE.
///
/// Body: `{"prompt": "string", "context": {}}`
///
/// Emits one `event: stage` per cognitive stage (observe, understand, ...,
/// improve) and a final `event: done` with the aggregated result.
pub async fn cognitive_stream(
    State(state): State<ServiceState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let prompt = body["prompt"].as_str().unwrap_or("").to_string();
    let stages = vec![
        "observe",
        "understand",
        "create_goal",
        "plan",
        "execute",
        "verify",
        "learn",
        "improve",
    ];

    let svc = state.svc.clone();
    let kernel_present = svc.kernel().is_some();
    let stream = async_stream::stream! {
        // Emit one event per stage.
        for stage in &stages {
            let payload = json!({
                "stage": stage,
                "status": "started",
                "timestamp_ms": chrono::Utc::now().timestamp_millis(),
            });
            yield Ok::<Event, Infallible>(Event::default()
                .event("stage")
                .data(payload.to_string()));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Run the actual cognitive cycle (inline, mirroring the HTTP handler).
        let response_text = if kernel_present {
            // Use the attached cognitive kernel for a real response.
            let observation = json!({
                "input": prompt,
                "modality": "text",
            });
            if let Some(kernel) = svc.kernel() {
                let cycle = kernel.run_cycle(observation).await;
                cycle.result
                    .as_ref()
                    .and_then(|r| r.as_str().map(String::from))
                    .unwrap_or_else(|| format!("Processed: {prompt}"))
            } else {
                format!("Processed: {prompt}")
            }
        } else {
            // Stub path: deterministic response.
            if prompt.trim().is_empty() {
                "[no input]".to_string()
            } else {
                format!("Processed: {prompt}")
            }
        };

        yield Ok(Event::default()
            .event("done")
            .data(json!({
                "response": response_text,
                "stages_completed": stages,
                "kernel_used": kernel_present,
            }).to_string()));
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// `POST /api/v1/generate/stream` — stream LLM tokens as SSE.
///
/// Body: `{"prompt": "string", "max_tokens": 100}`
///
/// Emits `event: token` for each generated chunk and `event: done` at the end.
pub async fn generate_stream(
    State(state): State<ServiceState>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let prompt = body["prompt"].as_str().unwrap_or("").to_string();
    let max_tokens = body["max_tokens"].as_u64().unwrap_or(100) as usize;

    let svc = state.svc.clone();
    let kernel_present = svc.kernel().is_some();
    let stream = async_stream::stream! {
        let start = std::time::Instant::now();

        // Generate the full response.
        let response_text = if kernel_present {
            let observation = json!({"input": prompt, "modality": "text"});
            if let Some(kernel) = svc.kernel() {
                let cycle = kernel.run_cycle(observation).await;
                cycle.result
                    .as_ref()
                    .and_then(|r| r.as_str().map(String::from))
                    .unwrap_or_else(|| format!("Processed: {prompt}"))
            } else {
                String::new()
            }
        } else if prompt.trim().is_empty() {
            "[no input]".to_string()
        } else {
            format!("Processed: {prompt}")
        };

        // Chunk the response into word-level "tokens" for streaming display.
        let words: Vec<&str> = response_text.split_whitespace().collect();
        let chunk_size = 1.max(words.len() / max_tokens.max(1));
        let mut emitted = 0usize;
        for chunk in words.chunks(chunk_size) {
            let text = chunk.join(" ");
            let payload = json!({
                "text": text,
                "tokens_generated": chunk.len(),
                "duration_ms": start.elapsed().as_millis() as u64,
            });
            yield Ok::<Event, Infallible>(Event::default()
                .event("token")
                .data(payload.to_string()));
            emitted += chunk.len();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        yield Ok(Event::default()
            .event("done")
            .data(json!({
                "finish_reason": "stop",
                "total_tokens": emitted,
                "total_duration_ms": start.elapsed().as_millis() as u64,
            }).to_string()));
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// `GET /api/v1/events/sse` — SSE stream of live service events.
///
/// Unlike the WebSocket endpoint, this is unidirectional (server → client)
/// and works over plain HTTP/1.1 — useful for browser clients that don't
/// support WebSockets or for server-to-server event fan-out.
pub async fn events_sse(
    State(state): State<ServiceState>,
) -> impl IntoResponse {
    let mut sub = state.svc.event_bus().subscribe();
    let stream = async_stream::stream! {
        loop {
            match tokio::time::timeout(Duration::from_secs(30), sub.recv()).await {
                Ok(Some(env)) => {
                    let payload = json!({
                        "seq": env.id.0,
                        "timestamp_ms": env.timestamp.timestamp_millis(),
                        "event": format!("{:?}", env.event),
                        "data": env.event,
                    });
                    yield Ok::<Event, Infallible>(Event::default()
                        .event("event")
                        .data(payload.to_string()));
                }
                Ok(None) => break,
                Err(_) => {
                    // Timeout — emit a keepalive comment.
                    yield Ok(Event::default().comment("keepalive"));
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new()
        .interval(Duration::from_secs(15))
        .text("keepalive"))
}
