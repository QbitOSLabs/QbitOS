# qbitos-evolution

Evaluator, improvement engine, regression guard.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Drives the controlled evolution of the agent's skills over time.

## Pipeline

1. **Evaluate** — Run all skills against a test suite, measure success rate and latency.
2. **Improve** — For underperforming skills, propose parameter mutations (rephrase prompt, add verification step, etc.).
3. **Test** — Apply the mutation and re-evaluate.
4. **Accept/Reject** — Accept if improvement exceeds threshold; reject if regression detected.

## Regression Guard

Every accepted change is recorded. If a future change breaks a previously-passing test, the change is automatically rejected and the previous version is restored.

## Usage

```rust
use qbitos_evolution::EvolutionModule;
let evolution = EvolutionModule::new();
let report = evolution.run_cycle(&app.skills, &test_suite).await?;
```
