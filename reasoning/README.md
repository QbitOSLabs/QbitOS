# qbitos-reasoning

Propositional logic, CSP, Bayesian networks, MDP, game-tree search.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Five reasoning engines, each available as a standalone module.

## Propositional Logic

Resolution-based theorem prover:

```rust
use qbitos_reasoning::{LogicEngine, Proposition};
let engine = LogicEngine::new();
let prop = Proposition::Implies(
    Box::new(Proposition::atom("a")),
    Box::new(Proposition::atom("b")),
);
let result = engine.prove(&prop);
assert!(result.proven);
```

## CSP (Constraint Satisfaction Problem)

AC-3 arc consistency + forward checking + backtracking.

## Bayesian Networks

Probabilistic inference with conditional probability tables.

## MDP (Markov Decision Process)

Value iteration and policy iteration for stochastic planning.

## Game-Tree Search

Minimax with alpha-beta pruning for adversarial games.

## Testing

~20 tests covering all five engines.
