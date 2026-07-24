# qbitos-world-model

Knowledge graph, entity engine, causal reasoning, simulation.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Maintains the agent's model of the world.

## Knowledge Graph

A directed graph of entities and relationships:

```rust
use qbitos_world_model::WorldModel;
let wm = WorldModel::new();
wm.knowledge_graph.add("dog", "is_a", "mammal");
wm.knowledge_graph.add("cat", "is_a", "mammal");
let mammals = wm.knowledge_graph.query_subjects("mammal", "is_a");
```

## Entity Engine

- Alias resolution (NYC = New York City)
- Entity merging (deduplication)
- Confidence scoring

## Causal Reasoning

- Causal graph (DAG)
- Intervention queries (do-calculus)
- Counterfactual reasoning

## Simulation Engine

- Forward simulation: predict the state after a sequence of actions
- Used for planning and what-if analysis

## Testing

4 tests covering the knowledge graph.
