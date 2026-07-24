# qbitos-training

Curriculum, experience replay, self-play, self-improvement, meta-learning.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Self-learning, self-improvement, and curriculum engines for QbitOS.

## Modules

### Curriculum Learning

5-stage progression with per-stage promotion thresholds:

| Stage    | Difficulty | Promotion Threshold | Min Samples |
|----------|------------|---------------------|-------------|
| 1        | Trivial    | 90% success rate    | 8           |
| 2        | Easy       | 85%                 | 16          |
| 3        | Medium     | 75%                 | 32          |
| 4        | Hard       | 65%                 | 64          |
| 5        | Expert     | 50%                 | 128         |

### Experience Replay

Ring buffer with uniform + priority-weighted sampling (alpha exponent).

### Self-Play

Arbitrary games via 4 closures (policy, apply, is_terminal, evaluate). Tracks Win/Loss/Draw/Incomplete stats.

### Self-Improvement Cycle

Detects 4 weakness types, proposes fixes, tests, accepts if expected_improvement ≥ 5%.

### Meta-Learning

UCB1 + ε-greedy strategy selection with Wilson 95% confidence intervals.

## Usage

```rust
use qbitos_training::{TrainingLoop, TrainingConfig};
let loop_ = TrainingLoop::new(TrainingConfig::default());
let report = loop_.run(|step, stage| 0.8);
```

## Testing

29 tests covering all 6 sub-engines.
