# qbitos-memory

L0-L5 biological memory fabric with HNSW vector index and consolidation.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Implements a biological-memory-inspired hierarchy with five layers plus a meta-strategy layer:

| Layer | Name          | Purpose                          | Backend                    |
|-------|---------------|----------------------------------|----------------------------|
| L0    | Sensory       | Raw per-modality input buffers   | In-memory ring buffer      |
| L1    | Working       | Current task context             | In-memory map              |
| L2    | Episodic      | What happened when               | HNSW + BTreeMap            |
| L3    | Semantic      | Facts, concepts, relationships   | Petgraph DiGraph + HNSW    |
| L4    | Procedural    | Skills, workflows                | Skill genome               |
| L5    | Meta          | Learning strategies, confidence  | Postgres / Sled            |

## Backends

Five pluggable backends (all real implementations):

- **Sled** (default, pure Rust embedded)
- **PostgreSQL + pgvector**
- **Qdrant**
- **Redis** (cache layer)
- **RocksDB**

Enable via feature flags.

## Advanced Search

- Filtered ANN (approximate nearest neighbor)
- Product quantization for memory-efficient storage
- BM25 hybrid search (combines vector + lexical)
- MMR (Maximal Marginal Relevance) re-ranking
- Adaptive ef parameter (auto-tunes HNSW search depth)

## Usage

```rust
use qbitos_memory::MemoryFabric;
let memory = MemoryFabric::with_stub_embeddings(384);
memory.episodic.record(json!({"event": "hello"}), Outcome::Success);
let results = memory.episodic.recall_similar(&json!({"query": "hello"}), 10);
```

## Testing

15 tests covering all 6 memory layers, HNSW index, and consolidation.
