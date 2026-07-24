# qbitos-neural-engine

Native cognitive agent + candle-based inference backends.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Provides inference for QbitOS. The default path uses the NativeAgent, which composes responses from memory recall, skill matching, knowledge-graph queries, and the reasoning engine — no external LLM required.

## Native Agent

The native agent implements a 2026-style chain-of-thought reasoning pipeline:

1. Decompose the prompt into sub-questions
2. For each sub-question, recall relevant memories + skills + facts
3. Apply self-consistency: generate multiple reasoning paths and select the one with highest agreement
4. Synthesize a final answer from the reasoning chain

```rust
use qbitos_neural_engine::{NativeAgent, InferenceRequest};
let agent = NativeAgent::new(cognitive_context);
let response = agent.compose_response(&request).await;
```

## Other Features

- KV cache for fast autoregressive generation
- Sampler (TopP, TopK, Temperature, greedy)
- Quantization (INT8, INT4)
- Self-reflection and constitutional AI
- Uncertainty quantification
- Optional external models (Qwen2, etc.) via candle-core
