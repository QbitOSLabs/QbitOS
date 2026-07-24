//! Benchmark for the cognitive loop. Measures throughput of full
//! observe→understand→goal→plan→execute→verify→learn→improve cycles.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use qbitos_kernel::{
    cognitive_kernel::{CognitiveCycle, CognitiveKernel, CognitiveStageHandler, CognitiveState},
    event_system::EventBus,
    resource_manager::ResourceManager,
    scheduler::Scheduler,
};
use serde_json::json;

struct NoopHandler;

impl CognitiveStageHandler for NoopHandler {
    fn observe(&self, _s: &mut CognitiveState) -> serde_json::Value { json!({}) }
    fn understand(&self, o: &serde_json::Value, _s: &mut CognitiveState) -> serde_json::Value { o.clone() }
    fn create_goal(&self, u: &serde_json::Value, _s: &mut CognitiveState) -> Option<serde_json::Value> { Some(u.clone()) }
    fn plan(&self, g: &serde_json::Value, _s: &mut CognitiveState) -> Option<serde_json::Value> { Some(g.clone()) }
    fn execute(&self, p: &serde_json::Value, _s: &mut CognitiveState) -> serde_json::Value { p.clone() }
    fn verify(&self, _g: &serde_json::Value, _r: &serde_json::Value, _s: &mut CognitiveState) -> bool { true }
    fn learn(&self, _c: &CognitiveCycle, _s: &mut CognitiveState) -> serde_json::Value { json!({}) }
    fn improve(&self, _c: &CognitiveCycle, _s: &mut CognitiveState) -> serde_json::Value { json!({}) }
}

fn bench_cycle(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let bus = EventBus::new(1024);
    let scheduler = Scheduler::new(4);
    let resources = ResourceManager::new();
    let kernel = CognitiveKernel::new(bus, scheduler, resources, Arc::new(NoopHandler));

    c.bench_function("cognitive_cycle", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(kernel.run_cycle(json!({"input": "bench"})).await);
        });
    });
}

criterion_group!(benches, bench_cycle);
criterion_main!(benches);
