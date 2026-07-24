# qbitos-learning

Multi-armed bandits, preference learning, feedback, Q-learning.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Online learning algorithms used by the cognitive kernel's Learn and Improve stages.

## Modules

- **Multi-armed bandits**: UCB1, ε-greedy for skill selection
- **Preference learning**: Bradley-Terry model for learning from pairwise feedback
- **Feedback collector**: Stores user feedback (explicit ratings, implicit signals)
- **Q-learning**: Tabular Q-learning for state-action value estimation

## Usage

```rust
use qbitos_learning::{Ucb1Bandit, FeedbackCollector};
let mut bandit = Ucb1Bandit::new(vec!["skill_a".into(), "skill_b".into()]);
let chosen = bandit.select();
bandit.update(&chosen, 0.8);
```

## Testing

~10 tests.
