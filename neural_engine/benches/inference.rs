//! Benchmark for the neural inference engine.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use qbitos_neural_engine::{
    backends::{CandleBackend, InferenceEngine, InferenceRequest, ModelBackend},
    kv_cache::{KvCache, KvCacheConfig},
    sampling::{Sampler, SamplerConfig, SamplingStrategy},
};
use std::sync::Arc;

fn bench_generate(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let backend = Arc::new(CandleBackend::new("bench-model"));
    backend.load_blocking();
    let kv_cache = Arc::new(KvCache::new(KvCacheConfig::default()));
    let sampler = Arc::new(Sampler::new(SamplerConfig {
        strategy: SamplingStrategy::Greedy,
        ..SamplerConfig::default()
    }));
    let engine = InferenceEngine::new(backend, kv_cache, sampler);

    c.bench_function("generate_256_tokens", |b| {
        b.to_async(&rt).iter(|| async {
            let req = InferenceRequest {
                prompt: "Hello, world!".into(),
                max_tokens: 256,
                ..InferenceRequest::default()
            };
            black_box(engine.generate(&req).await.unwrap());
        });
    });
}

// Helper trait for sync load in benchmarks.
trait ModelBackendExt {
    fn load_blocking(&self);
}

impl ModelBackendExt for CandleBackend {
    fn load_blocking(&self) {
        let backend = self;
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async { backend.load().await.unwrap() });
    }
}

criterion_group!(benches, bench_generate);
criterion_main!(benches);
