//! QbitOS kernel binary — boots the runtime and runs a self-test cognitive cycle.

use std::sync::Arc;
use std::time::Duration;

use qbitos_kernel::{
    cognitive_kernel::{CognitiveCycle, CognitiveStageHandler, CognitiveState},
    config::KernelConfig,
    init_tracing,
    runtime::KernelRuntime,
};
use serde_json::json;

struct EchoHandler;

impl CognitiveStageHandler for EchoHandler {
    fn observe(&self, _state: &mut CognitiveState) -> serde_json::Value {
        json!({"status": "ok"})
    }
    fn understand(&self, obs: &serde_json::Value, _state: &mut CognitiveState) -> serde_json::Value {
        obs.clone()
    }
    fn create_goal(&self, u: &serde_json::Value, _state: &mut CognitiveState) -> Option<serde_json::Value> {
        Some(u.clone())
    }
    fn plan(&self, g: &serde_json::Value, _state: &mut CognitiveState) -> Option<serde_json::Value> {
        Some(g.clone())
    }
    fn execute(&self, p: &serde_json::Value, _state: &mut CognitiveState) -> serde_json::Value {
        p.clone()
    }
    fn verify(&self, _g: &serde_json::Value, _r: &serde_json::Value, _state: &mut CognitiveState) -> bool {
        true
    }
    fn learn(&self, _c: &CognitiveCycle, _state: &mut CognitiveState) -> serde_json::Value {
        json!({"learned": true})
    }
    fn improve(&self, _c: &CognitiveCycle, _state: &mut CognitiveState) -> serde_json::Value {
        json!({"improved": true})
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cfg = KernelConfig::default();
    let (tx, rx) = flume::bounded::<serde_json::Value>(64);
    let runtime = KernelRuntime::new(cfg, Arc::new(EchoHandler));
    let handle = runtime.start(rx);

    for i in 0..5 {
        tx.send_async(json!({
            "cycle": i,
            "prompt": format!("hello #{i}"),
        })).await?;
    }

    tokio::time::sleep(Duration::from_millis(500)).await;
    handle.shutdown();
    handle.join().await?;
    println!("QbitOS kernel self-test completed");
    Ok(())
}
