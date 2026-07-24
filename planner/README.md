# qbitos-planner

Goal engine, task graph (DAG), reasoning, and decision engine.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

The planner takes a goal derived by the cognitive kernel and produces a task DAG that achieves it.

## Modules

- **Goal engine** — Hierarchical goal decomposition. Top-level goals are broken into sub-goals until each is achievable by a single skill or tool.
- **Task graph** — DAG with dependency resolution. Tasks execute in topological order; independent tasks run in parallel.
- **Reasoning integration** — Calls into qbitos-reasoning for propositional logic, CSP solving, Bayesian inference, MDP planning, and game-tree search.
- **Decision engine** — Multi-criteria selection: picks the best skill for a task based on confidence, latency, and past success rate.

## Usage

```rust
use qbitos_planner::Planner;
let planner = Planner::new();
let plan = planner.plan(goal, &context);
```

## Testing

10 tests covering goal decomposition, DAG execution, and decision making.
