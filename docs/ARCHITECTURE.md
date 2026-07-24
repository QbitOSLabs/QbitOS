# QbitOS Architecture

> **A Rust 2024 native multimodal cognitive AI operating system.**
> *A small 3B AI model becomes powerful through a superior cognitive operating system.*

This document describes the system architecture of QbitOS at the level of
subsystems, data flow, and runtime topology. For per-crate API details see
the per-crate `README.md` files. For the HTTP/REST API see
[`API.md`](./API.md). For deployment see [`DEPLOYMENT.md`](./DEPLOYMENT.md).

---

## 1. Design Philosophy

QbitOS is **not** a chatbot wrapper, an API gateway, or an agent
orchestration framework. It is a from-scratch artificial cognitive
operating system whose intelligence is produced by its own native
runtime.

The core thesis:

> A small (3B-parameter) model becomes powerful when embedded in a
> superior cognitive operating system that provides memory, planning,
> tool use, skill composition, self-improvement, and a controlled
> evolution loop.

Every subsystem is written from scratch in Rust 2024 edition. No
external LLM API calls are required for the system to function — the
default inference path uses the **NativeAgent**, which composes
responses from memory recall, skill matching, knowledge-graph queries,
and the reasoning engine.

---

## 2. Workspace Layout

```
qbitos/
├── kernel/         Cognitive kernel, scheduler, event bus, resource manager
├── neural_engine/  Native cognitive agent + candle-based inference backends
├── vision/         Image processor, HOG/color/LBP features, OCR, object detection
├── speech/         STT (FFT+MFCC+VAD) + TTS (formant synthesis)
├── memory/         L0-L5 biological memory fabric with consolidation
├── world_model/    Knowledge graph, entity engine, causal reasoning, simulation
├── planner/        Goal engine, task graph (DAG), reasoning, decision engine
├── skills/         Skill genome: registry, generator, optimizer, executor
├── tools/          Tool VM: filesystem, terminal, http, calculator, summarizer, …
├── evolution/      Evaluator, improvement engine, regression guard
├── data/           Unified dataset loader: 32+ formats, streaming, S3/MinIO, HF Hub
├── reasoning/      Propositional logic, CSP, Bayesian networks, MDP, game-tree search
├── learning/       Multi-armed bandits, preference learning, feedback, Q-learning
├── safety/         Content filter, prompt-injection detector, audit log, rate limiter
├── training/       Curriculum, experience replay, self-play, self-improvement, meta-learning
├── services/       HTTP/REST (Axum), WebSocket, JSON-RPC 2.0, OpenAPI 3.1, auth
├── cli/            Top-level `qbitos` binary wiring all subsystems together
├── desktop/        Tauri v2 desktop app (React 19 + TS + Tailwind 4)
├── examples_pkg/   End-to-end demos (cognitive cycle, WS events, RPC, training loop)
└── docs/           This documentation tree
```

**20 crates total** · 190 Rust files · ~37 000 LOC · 196 tests passing.

---

## 3. The Cognitive Loop

The kernel runs an 8-stage cognitive cycle continuously, driven by the
scheduler. Each stage emits events on the global event bus and writes
to the appropriate memory layer.

```
        ┌────────────────────────────────────────────────────────────┐
        │                                                            │
        ▼                                                            │
   ┌─────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────┴──┐
   │ Observe │───▶│ Understand  │───▶│ Create Goals │───▶│   Plan     │
   └─────────┘    └─────────────┘    └──────────────┘    └────────────┘
                                                              │
                                                              ▼
   ┌─────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
   │ Improve │◀───│   Learn     │◀───│   Verify     │◀───│  Execute   │
   └─────────┘    └─────────────┘    └──────────────┘    └────────────┘
        │
        └─▶ (next cycle)
```

| Stage        | Responsibility                                  | Memory Layer |
|--------------|-------------------------------------------------|--------------|
| Observe      | Pull input from sensory/working memory          | L0, L1       |
| Understand   | Embed observation, retrieve episodic memories   | L2           |
| Create Goals | Derive a goal from the observation              | L1           |
| Plan         | Build a task DAG to achieve the goal            | —            |
| Execute      | Run tools / skills / reasoning steps            | L4           |
| Verify       | Check the result against success criteria       | —            |
| Learn        | Store the episode, update skill stats           | L2, L4       |
| Improve      | Run self-improvement cycle (detect → fix)       | L5           |

---

## 4. Memory Fabric (L0–L5)

A biological-memory-inspired hierarchy with five layers, plus a
meta-strategy layer:

| Layer | Name          | Purpose                          | Backend                              | Persistence |
|-------|---------------|----------------------------------|--------------------------------------|-------------|
| L0    | Sensory       | Raw per-modality input buffers   | In-memory ring buffer                | ephemeral   |
| L1    | Working       | Current task context             | In-memory map                        | ephemeral   |
| L2    | Episodic      | "What happened when"             | HNSW vector index + BTreeMap         | session     |
| L3    | Semantic      | Facts, concepts, relationships   | Petgraph DiGraph + HNSW              | permanent   |
| L4    | Procedural    | Skills, workflows                | Skill genome (qbitos-skills)         | permanent   |
| L5    | Meta          | Learning strategies, confidence  | Postgres / Sled                      | permanent   |

**Pluggable backends:** PostgreSQL + pgvector · Qdrant · Redis · RocksDB · Sled.
Default is Sled (pure Rust embedded). All five are real implementations;
choose per-deployment via feature flags.

**Advanced search:** filtered ANN, product quantization, BM25 hybrid
search, MMR re-ranking, adaptive `ef` parameter.

---

## 5. Subsystem Topology

```
                       ┌─────────────────────────────────────────────┐
                       │              qbitos CLI / App               │
                       │   (wires all subsystems into one binary)    │
                       └──┬───────────────────────────────────────┬──┘
                          │                                       │
            ┌─────────────▼──────────┐              ┌────────────▼────────┐
            │   Cognitive Kernel     │              │   Service Layer     │
            │  (8-stage loop,        │              │  (HTTP/WS/RPC)      │
            │   scheduler, bus)      │              │  Axum + Tower       │
            └──┬─────────────────────┘              └────────────┬────────┘
               │                                                  │
   ┌───────────┴────────────┬─────────────────────────────────────┼─────────────┐
   │                        │                                     │             │
   ▼                        ▼                                     ▼             ▼
┌─────────┐  ┌──────────────────────┐  ┌─────────────────┐  ┌──────────┐  ┌─────────┐
│ Memory  │  │  NeuralEngine        │  │   Planner       │  │  Skills  │  │  Tools  │
│ Fabric  │  │  (NativeAgent +      │  │  (Goal, DAG,    │  │ (genome, │  │  (VM,   │
│ (L0-L5) │  │   InferenceEngine)   │  │   Decision)     │  │  optim.) │  │  sandbox│
└────┬────┘  └──────────┬───────────┘  └────────┬────────┘  └────┬─────┘  └────┬────┘
     │                  │                       │                │             │
     │                  │                       │                │             │
     ▼                  ▼                       ▼                ▼             ▼
┌─────────┐  ┌──────────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│ World   │  │  Vision / Speech     │  │   Evolution     │  │     Training         │
│ Model   │  │  (multimodal I/O)    │  │  (evaluator,    │  │  (curriculum,        │
│ (KG,    │  │                      │  │   improvement,  │  │   replay, self-play, │
│  causal)│  │                      │  │   regression)   │  │   meta-learning)     │
└─────────┘  └──────────────────────┘  └─────────────────┘  └──────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Cross-cutting: Reasoning · Learning · Safety · Data (32+ dataset loaders)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Skill Module — Real Backend Wiring

The `SkillModule` exposes two trait-object backends:

- `ToolBackend` — invoked by `StepKind::Tool` steps.
- `InferBackend` — invoked by `StepKind::Infer` steps.

Three configurations:

### 6.1 Noop (default, fast build)

```rust
let skills = SkillModule::with_noop_backends(memory.episodic.clone());
```

Skill *registration* and *statistics* work, but skill *execution* returns
errors. Used for tests, REPL inspection, and rapid development builds.

### 6.2 Real backends (feature `real-backends`)

```rust
let tool_backend = Arc::new(ToolModuleBackend::new(tools));
let infer_backend = Arc::new(NativeAgentBackend::empty(embedder, 384));
let skills = Arc::new(SkillModule::new(
    memory.episodic.clone(),
    tool_backend,
    infer_backend.clone() as Arc<dyn InferBackend>,
));
// … later, after NativeAgent is constructed …
infer_backend.set_agent(native_agent);
```

**Late binding via `ArcSwapOption`** breaks the chicken-and-egg between
`NativeAgent` (needs `CognitiveContext` → `SkillModule`) and `SkillModule`
(needs `InferBackend` → `NativeAgent`). Lock-free concurrent reads with
atomic updates — safe to call `set_agent` while `infer` is in flight.

### 6.3 Custom backends

Implement `ToolBackend` and `InferBackend` yourself and pass them to
`SkillModule::new`. Useful for mocking in tests or bridging to
non-standard inference engines.

---

## 7. Service Layer

The `services` crate exposes the cognitive OS over four protocols:

| Protocol   | Endpoint                  | Use case                              |
|------------|---------------------------|---------------------------------------|
| HTTP/REST  | `/api/v1/*`               | Browser, curl, scripts                |
| WebSocket  | `/api/v1/events/ws`       | Live event streaming, command dispatch|
| JSON-RPC   | programmatic              | Programmatic clients (batch support)  |
| OpenAPI    | `/api/v1/openapi.json`    | Spec generation, client SDK generation|

**Orchestrator:** `CognitiveService` is the single handle — clone-cheap
(`Arc`-backed), owns the `MemoryFabric`, `SkillModule`, `Curriculum`,
`ExperienceReplay`, `MetaLearner`, `SelfImprovementCycle`, optional
`PersistentEventStore`, and optional `CognitiveKernel`.

**Authentication:** Bearer-token via `AuthService` (SHA-256 hashed,
constant-time compared). Public path / public prefix / public read
bypass modes. Enabled via `--auth-token <TOKEN>`.

**Persistent events:** `PersistentEventStore` backs the in-memory buffer
with the memory fabric's semantic layer — events survive across
restarts. `POST /api/v1/events/replay` restores history.

---

## 8. Training Loop

The `training` crate orchestrates five training sub-engines:

```
            ┌─────────────────────────────────────────────┐
            │              TrainingLoop                    │
            │  (orchestrates all sub-engines per step)    │
            └──┬──────────┬──────────┬──────────┬─────────┘
               │          │          │          │
   ┌───────────▼─┐  ┌─────▼─────┐  ┌▼────────┐ ┌▼───────────┐
   │ Curriculum  │  │ Experience│  │Self-Play│ │  Meta-     │
   │ 5 stages:   │  │  Replay   │  │ Engine  │ │  Learner   │
   │ Trivial →   │  │ (uniform +│  │ (game   │ │ (UCB1 +    │
   │ Easy → Med  │  │ prioritized)│  │ trees)  │ │ ε-greedy) │
   │ → Hard →    │  │           │  │         │ │            │
   │ Expert      │  │           │  │         │ │            │
   └─────────────┘  └───────────┘  └─────────┘ └────────────┘
               │          │          │          │
               └──────────┴──────────┴──────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Self-Improvement Cycle │
                  │  detect weakness →     │
                  │  propose fix → test →  │
                  │  accept/reject         │
                  └────────────────────────┘
```

| Stage    | Promotion Threshold | Min Samples |
|----------|---------------------|-------------|
| Trivial  | 90% success rate    | 8           |
| Easy     | 85%                 | 16          |
| Medium   | 75%                 | 32          |
| Hard     | 65%                 | 64          |
| Expert   | 50%                 | 128         |

---

## 9. Data Flow: A Cognitive Cycle

This is the end-to-end flow when a client `POST /api/v1/cognitive/cycle`
arrives at a server running with `--real-kernel`:

```
Client ──── POST /api/v1/cognitive/cycle ────▶ HTTP Router (Axum)
                                                       │
                                                       ▼
                                          AuthService (Bearer check)
                                                       │
                                                       ▼
                                          CognitiveService.record_request
                                                       │
                                                       ▼
                                          if kernel attached:
                                            kernel.run_cycle(observation)
                                                       │
                                  ┌────────────────────┴────────────────────┐
                                  │                                         │
                                  ▼                                         ▼
                          Observe → Understand                     Create Goals → Plan
                                  │                                         │
                                  ▼                                         ▼
                          Memory recall (L2)                       Planner builds DAG
                          Semantic lookup (L3)                              │
                                  │                                         │
                                  └──────────────┬──────────────────────────┘
                                                 ▼
                                              Execute
                                                 │
                                 ┌───────────────┼───────────────┐
                                 ▼               ▼               ▼
                              Tool step      Infer step      Verify step
                                 │               │               │
                                 ▼               ▼               ▼
                          ToolModuleBackend  NativeAgentBackend  (passes if
                          (real ToolVm)     (NativeAgent           output ok)
                                           compose_response)
                                 │               │               │
                                 └───────────────┴───────────────┘
                                                 │
                                                 ▼
                                       Learn → Improve
                                                 │
                                       ┌─────────┴─────────┐
                                       ▼                   ▼
                                  Store episode (L2)   SelfImprovementCycle
                                  Update skill stats   (detect weaknesses)
                                                       │
                                                       ▼
                                          Publish CognitiveCycleCompleted event
                                                       │
                                  ┌────────────────────┴────────────────────┐
                                  ▼                                         ▼
                          WebSocket subscribers                   PersistentEventStore
                          (live stream)                          (semantic memory)
                                                       │
                                                       ▼
                                          Return CognitiveCycleResponse
                                          (cycle_id, output, phases, confidence)
```

---

## 10. Build Configurations

| Configuration                       | Build time | Binary size | Use case                          |
|-------------------------------------|------------|-------------|-----------------------------------|
| `cargo build -p qbitos-cli`         | ~30s       | ~50 MB      | Dev / REPL / tests                |
| `--features postgres-backed` (default) | ~30s    | ~50 MB      | PostgreSQL primary backend        |
| `--features real-backends`          | ~5min      | ~193 MB     | Production with NativeAgent       |
| `--features all-backends`           | ~6min      | ~210 MB     | All DB backends enabled           |
| `--features real-backends,all-backends` | ~7min | ~230 MB     | Full production build             |

---

## 11. Security Model

- **Bearer-token auth** (optional, `--auth-token`): SHA-256 hashed,
  constant-time compared via `subtle::ConstantTimeEq`.
- **Public paths**: `/api/v1/health`, `/api/v1/version`,
  `/api/v1/openapi.json`, `/api/v1/openapi.yaml`, `/api/v1/events/ws`
  always bypass auth.
- **Public read mode**: GETs bypass auth (configurable).
- **CORS**: permissive (dev default) or restricted (`--restricted-cors`).
- **Rate limiting**: `qbitos_safety::RateLimiter` (token bucket per
  principal).
- **Audit logging**: `qbitos_safety::AuditLogger` records all
  authenticated requests.
- **Content filtering**: `qbitos_safety::ContentFilter` checks inputs.
- **Prompt-injection detection**: `qbitos_safety::InjectionDetector`.
- **Output validation**: `qbitos_safety::OutputValidator`.

---

## 12. Persistence Model

| State              | Stored where                          | Survives restart? |
|--------------------|---------------------------------------|-------------------|
| Memory L0/L1       | In-memory                             | No                |
| Memory L2 (episodic)| In-memory + optional durable backend | Yes (if backend)  |
| Memory L3 (semantic)| In-memory + optional durable backend | Yes (if backend)  |
| Memory L4 (skills) | In-memory (registry)                  | Yes (via state snapshot) |
| Memory L5 (meta)   | Postgres / Sled                       | Yes               |
| Skill stats        | In-memory                             | Yes (via state snapshot) |
| Knowledge graph    | In-memory                             | Yes (via state snapshot) |
| Event log          | In-memory buffer                      | Yes (if persistent event store) |
| Service config     | CLI args                              | No                |

The CLI's `state save` / `state load` commands serialize the entire
application state (skills, episodes, knowledge graph, datasets) to a
JSON file for cross-restart persistence.

---

## 13. See also

- [`API.md`](./API.md) — HTTP/REST + WebSocket + JSON-RPC reference
- [`DEPLOYMENT.md`](./DEPLOYMENT.md) — Production deployment guide
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — How to contribute
- [`CHANGELOG.md`](./CHANGELOG.md) — Release history
- [`../README.md`](../README.md) — Top-level overview
- [`../worklog.md`](../worklog.md) — Implementation history (per-task)
