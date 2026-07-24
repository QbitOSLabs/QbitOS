//! Example: exercise the JSON-RPC 2.0 interface of the QbitOS service.
//!
//! Start the server first:
//! ```sh
//! qbitos serve --addr 127.0.0.1:8080
//! ```
//!
//! Then run this example:
//! ```sh
//! cargo run --example rpc_client
//! ```
//!
//! Note: this example hits the HTTP endpoint `/api/v1/rpc` (a JSON-RPC over
//! HTTP bridge). If your deployment uses a different transport, adapt the
//! URL accordingly. As a fallback, this example also shows how to call the
//! `RpcServer` directly in-process.

use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ─── In-process RPC demo ─────────────────────────────────────────────
    // This shows how to use the RpcServer directly without any network.
    println!("── In-process JSON-RPC demo ──");
    let svc = qbitos_services::CognitiveService::new(Default::default());
    let rpc = qbitos_services::RpcServer::new(svc.clone());

    // Single request: ping.
    let req = qbitos_services::RpcRequest {
        jsonrpc: "2.0".into(),
        method: "ping".into(),
        params: None,
        id: Some(serde_json::json!(1)),
    };
    let resp = rpc.handle(req).await;
    println!("→ ping");
    println!("← result: {:?}", resp.result);
    println!();

    // Single request: cognitive.cycle.
    let req = qbitos_services::RpcRequest {
        jsonrpc: "2.0".into(),
        method: "cognitive.cycle".into(),
        params: Some(serde_json::json!({"input": "Hello, RPC!"})),
        id: Some(serde_json::json!(2)),
    };
    let resp = rpc.handle(req).await;
    println!("→ cognitive.cycle");
    println!("← result: {:?}", resp.result);
    println!();

    // Batch request.
    let batch = vec![
        qbitos_services::RpcRequest {
            jsonrpc: "2.0".into(),
            method: "ping".into(),
            params: None,
            id: Some(serde_json::json!(1)),
        },
        qbitos_services::RpcRequest {
            jsonrpc: "2.0".into(),
            method: "stats".into(),
            params: None,
            id: Some(serde_json::json!(2)),
        },
        qbitos_services::RpcRequest {
            jsonrpc: "2.0".into(),
            method: "health".into(),
            params: None,
            id: Some(serde_json::json!(3)),
        },
    ];
    let responses = rpc.handle_batch(batch).await;
    println!("→ batch (3 requests)");
    println!("← {} responses", responses.len());
    for r in &responses {
        println!("  id={:?} ok={}", r.id, r.error.is_none());
    }
    println!();

    // Raw JSON-RPC string.
    let raw = r#"{"jsonrpc":"2.0","method":"ping","id":99}"#;
    let out = rpc.handle_raw(raw).await?;
    println!("→ raw: {raw}");
    println!("← raw: {out}");
    println!();

    // ─── Optional: HTTP-RPC demo (requires `qbitos serve` running) ───────
    println!("── HTTP JSON-RPC demo (optional) ──");
    let base = "http://127.0.0.1:8080";
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
    {
        Ok(c) => c,
        Err(_) => {
            println!("  (skipped: failed to build HTTP client)");
            return Ok(());
        }
    };

    // Try to call /api/v1/health as a smoke test for the running server.
    match client.get(format!("{base}/api/v1/health")).send().await {
        Ok(resp) => {
            println!("  server reachable (status {})", resp.status());
        }
        Err(_) => {
            println!("  (skipped: server not running at {base})");
            println!("  Start it with: qbitos serve --addr 127.0.0.1:8080");
        }
    }

    println!("\n✓ RPC example complete.");
    let _ = Arc::new(()); // keep Arc in scope for clarity
    Ok(())
}
