//! Example: run a single cognitive cycle against the QbitOS service.
//!
//! Start the server first:
//! ```sh
//! qbitos serve --addr 127.0.0.1:8080
//! ```
//!
//! Then run this example:
//! ```sh
//! cargo run --example cognitive_cycle --features qbitos-services
//! ```

use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base = "http://127.0.0.1:8080";
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    // 1. Health check.
    println!("→ GET /api/v1/health");
    let resp: serde_json::Value = client
        .get(format!("{base}/api/v1/health"))
        .send()
        .await?
        .json()
        .await?;
    println!("  healthy: {}", resp["data"]["healthy"]);
    println!("  uptime:  {}s", resp["data"]["uptime_seconds"]);
    println!("  api_version: {}", resp["data"]["api_version"]);
    println!();

    // 2. Store a memory.
    println!("→ POST /api/v1/memory");
    let resp: serde_json::Value = client
        .post(format!("{base}/api/v1/memory"))
        .json(&serde_json::json!({
            "content": "The user prefers concise answers.",
            "layer": "L3_semantic",
        }))
        .send()
        .await?
        .json()
        .await?;
    println!("  memory_id: {}", resp["data"]["memory_id"]);
    println!();

    // 3. Search memories.
    println!("→ POST /api/v1/memory/search");
    let resp: serde_json::Value = client
        .post(format!("{base}/api/v1/memory/search"))
        .json(&serde_json::json!({
            "query": "user preferences",
            "k": 5,
        }))
        .send()
        .await?
        .json()
        .await?;
    let results = resp["data"].as_array().unwrap();
    println!("  results: {}", results.len());
    for r in results {
        println!("    - {} (score={:.2})", r["content"], r["score"]);
    }
    println!();

    // 4. Run a cognitive cycle.
    println!("→ POST /api/v1/cognitive/cycle");
    let resp: serde_json::Value = client
        .post(format!("{base}/api/v1/cognitive/cycle"))
        .json(&serde_json::json!({
            "input": "Hello, QbitOS!",
            "max_cycles": 1,
        }))
        .send()
        .await?
        .json()
        .await?;
    println!("  cycle_id: {}", resp["data"]["cycle_id"]);
    println!("  output:   {}", resp["data"]["output"]);
    println!("  duration: {}ms", resp["data"]["duration_ms"]);
    println!("  confidence: {:.2}", resp["data"]["confidence"]);
    println!();

    // 5. Run a training step.
    println!("→ POST /api/v1/training/step");
    let resp: serde_json::Value = client
        .post(format!("{base}/api/v1/training/step"))
        .json(&serde_json::json!({"steps": 5}))
        .send()
        .await?
        .json()
        .await?;
    println!("  steps_executed: {}", resp["data"]["steps_executed"]);
    println!("  metric:         {:.4}", resp["data"]["metric"]);
    println!("  duration:       {}ms", resp["data"]["duration_ms"]);
    println!();

    // 6. Get curriculum state.
    println!("→ GET /api/v1/curriculum");
    let resp: serde_json::Value = client
        .get(format!("{base}/api/v1/curriculum"))
        .send()
        .await?
        .json()
        .await?;
    println!("  current_stage: {}", resp["data"]["current_stage"]);
    println!("  total_stages:  {}", resp["data"]["total_stages"]);
    println!();

    // 7. Get service stats.
    println!("→ GET /api/v1/stats");
    let resp: serde_json::Value = client
        .get(format!("{base}/api/v1/stats"))
        .send()
        .await?
        .json()
        .await?;
    println!("  total_requests: {}", resp["data"]["total_requests"]);
    println!("  total_events:   {}", resp["data"]["total_events"]);

    println!("\n✓ Cognitive cycle example complete.");
    Ok(())
}
