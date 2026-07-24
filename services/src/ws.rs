//! WebSocket real-time streaming.

use crate::events::ServiceEvent;
use crate::http::handlers::ServiceState;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A message that can be sent to the WebSocket client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsMessage {
    /// Message type (e.g. `event`, `ack`, `error`, `result`).
    pub kind: String,
    /// Payload.
    pub payload: Value,
}

/// A WebSocket session.
pub struct WsSession {
    /// The session ID.
    pub id: u64,
    /// The filter (if any) applied to events.
    pub filter: Option<String>,
}

impl WsSession {
    /// Create a new session.
    #[must_use]
    pub fn new(id: u64, filter: Option<String>) -> Self {
        Self { id, filter }
    }
}

/// Axum handler for `GET /api/v1/events/ws`.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<ServiceState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(socket: WebSocket, state: ServiceState) {
    state.svc.ws_session_started();
    let session_id = state
        .sub_counter
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (mut sender, mut receiver) = socket.split();

    // Send session-started message.
    let hello = WsMessage {
        kind: "session_started".into(),
        payload: serde_json::json!({
            "session_id": session_id,
            "instance_id": state.svc.instance_id(),
            "instance_name": state.svc.instance_name(),
        }),
    };
    if let Ok(text) = serde_json::to_string(&hello) {
        let _ = sender.send(Message::Text(text.into())).await;
    }

    // Subscribe to events.
    let mut event_sub = state.svc.event_bus().subscribe();

    // Use tokio::select! to multiplex events and incoming commands.
    loop {
        tokio::select! {
            // Incoming events from the bus.
            env = event_sub.recv() => {
                match env {
                    Some(env) => {
                        let msg = WsMessage {
                            kind: "event".into(),
                            payload: serde_json::to_value(&env).unwrap_or(Value::Null),
                        };
                        if let Ok(text) = serde_json::to_string(&msg) {
                            if sender.send(Message::Text(text.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                    None => break,
                }
            }
            // Incoming messages from the client.
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(cmd) = serde_json::from_str::<WsMessage>(&text) {
                            let resp = handle_command(&cmd, &state).await;
                            if let Ok(resp_text) = serde_json::to_string(&resp) {
                                if sender.send(Message::Text(resp_text.into())).await.is_err() {
                                    break;
                                }
                            }
                        } else {
                            let err = WsMessage {
                                kind: "error".into(),
                                payload: serde_json::json!({ "message": "invalid message format" }),
                            };
                            if let Ok(t) = serde_json::to_string(&err) {
                                let _ = sender.send(Message::Text(t.into())).await;
                            }
                        }
                    }
                    Some(Ok(Message::Ping(p))) => {
                        let _ = sender.send(Message::Pong(p)).await;
                    }
                    Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::warn!(error = ?e, "ws recv error");
                        break;
                    }
                    None => break,
                }
            }
        }
    }

    state.svc.ws_session_ended();
}

async fn handle_command(cmd: &WsMessage, state: &ServiceState) -> WsMessage {
    match cmd.kind.as_str() {
        "ping" => WsMessage {
            kind: "pong".into(),
            payload: serde_json::json!({ "echo": cmd.payload }),
        },
        "cycle" => {
            let input = cmd
                .payload
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let cycle_id = uuid::Uuid::new_v4();
            state.svc.publish(ServiceEvent::CognitiveCycleCompleted {
                cycle_id,
                phase: "execute".into(),
                duration_ms: 5,
                summary: format!("Processed: {input}"),
            });
            WsMessage {
                kind: "result".into(),
                payload: serde_json::json!({
                    "cycle_id": cycle_id,
                    "output": format!("Processed: {input}"),
                }),
            }
        }
        "stats" => {
            let stats = state.svc.stats();
            WsMessage {
                kind: "result".into(),
                payload: serde_json::to_value(&stats).unwrap_or(Value::Null),
            }
        }
        _ => WsMessage {
            kind: "error".into(),
            payload: serde_json::json!({
                "message": format!("unknown command: {}", cmd.kind),
            }),
        },
    }
}

/// WebSocket server (useful for tests and as a standalone).
pub struct WebSocketServer {
    /// The event bus to subscribe to.
    pub bus: crate::events::EventBus,
}

impl WebSocketServer {
    /// Construct a new WebSocket server.
    #[must_use]
    pub fn new(bus: crate::events::EventBus) -> Self {
        Self { bus }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::{CognitiveService, ServiceConfig};

    #[tokio::test]
    async fn ws_message_serialization() {
        let msg = WsMessage {
            kind: "ping".into(),
            payload: serde_json::json!({ "n": 1 }),
        };
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.contains("ping"));
        let back: WsMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back.kind, "ping");
    }

    #[test]
    fn session_id_assigned() {
        let s = WsSession::new(42, None);
        assert_eq!(s.id, 42);
    }

    #[tokio::test]
    async fn command_handler_ping() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let state = ServiceState::new(svc);
        let cmd = WsMessage {
            kind: "ping".into(),
            payload: serde_json::json!({"x": 1}),
        };
        let resp = handle_command(&cmd, &state).await;
        assert_eq!(resp.kind, "pong");
    }

    #[tokio::test]
    async fn command_handler_stats() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let state = ServiceState::new(svc);
        let cmd = WsMessage {
            kind: "stats".into(),
            payload: serde_json::json!({}),
        };
        let resp = handle_command(&cmd, &state).await;
        assert_eq!(resp.kind, "result");
    }

    #[tokio::test]
    async fn command_handler_unknown() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let state = ServiceState::new(svc);
        let cmd = WsMessage {
            kind: "unknown_cmd".into(),
            payload: serde_json::json!({}),
        };
        let resp = handle_command(&cmd, &state).await;
        assert_eq!(resp.kind, "error");
    }
}
