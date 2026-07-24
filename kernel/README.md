# qbitos-kernel

Cognitive kernel, scheduler, event bus, and runtime for QbitOS.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

The cognitive kernel is the heart of QbitOS. It runs the 8-stage cognitive loop continuously, driven by a priority-aware preemptive scheduler.

```
Observe → Understand → Create Goal → Plan → Execute → Verify → Learn → Improve
```

Each stage emits events on the global event bus and writes to the appropriate memory layer.

## Modules

- **cognitive_kernel** — The kernel itself. Owns the event bus, scheduler, resource manager, and a stage handler.
- **runtime** — Boots the kernel in a background task and drives the loop from an observation channel.
- **scheduler** — Priority-aware preemptive scheduler (Realtime / Interactive / Normal / Background).
- **event_system** — Global event bus with topic-prefix + wildcard subscriptions.
- **resource_manager** — Per-agent resource budgets (CPU / memory / GPU / tokens / IO).
- **config** — Kernel configuration.

## Usage

```rust
use qbitos_kernel::{CognitiveKernel, EventBus, Scheduler, ResourceManager};

let bus = EventBus::new(4096);
let scheduler = Scheduler::new(8);
let resources = ResourceManager::new();
let kernel = CognitiveKernel::new(bus, scheduler, resources, handler);
let cycle = kernel.run_cycle(observation).await;
```

## Testing

```bash
cargo test -p qbitos-kernel
```

10 tests covering the cognitive loop, scheduler, and event bus.
