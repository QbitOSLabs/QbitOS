//! Example: connect to the QbitOS WebSocket event stream and print events.
//!
//! Start the server first:
//! ```sh
//! qbitos serve --addr 127.0.0.1:8080
//! ```
//!
//! Then run this example:
//! ```sh
//! cargo run --example ws_events
//! ```

use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::connect_async;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = "ws://127.0.0.1:8080/api/v1/events/ws";
    println!("→ Connecting to {url}");
    let (mut ws, _) = connect_async(url).await?;
    println!("  connected");

    // Send a ping.
    let ping = serde_json::json!({
        "kind": "ping",
        "payload": {"hello": "world"},
    });
    ws.send(Message::Text(ping.to_string().into())).await?;
    println!("→ Sent ping");

    // Listen for 10 messages.
    let mut count = 0;
    while let Some(msg) = ws.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                count += 1;
                let v: serde_json::Value = serde_json::from_str(&text)?;
                let kind = v["kind"].as_str().unwrap_or("?");
                println!("← #{count} kind={kind}");
                if count >= 10 {
                    break;
                }
            }
            Ok(Message::Ping(p)) => {
                let _ = ws.send(Message::Pong(p)).await;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("error: {e}");
                break;
            }
        }
    }

    let _ = ws.close(None).await;
    println!("\n✓ WebSocket example complete.");
    Ok(())
}
