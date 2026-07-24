


<p align="center">
  <img src="https://i.ibb.co/jkCpjxvR/HN-co-To-XYAEDc-JF.png" alt="QbitOS Banner" width="700">
</p>


<p align="center">
  A Rust 2024 native multimodal cognitive AI operating system.
</p>

<p align="center">
  <a href="https://github.com/QbitOSLabs/QbitOS">GitHub</a> •
  <a href="https://qbitoslabs.ly">Website</a> •
  <a href="https://x.com/labsqbitos">X</a>
</p>


> *A small 3B-parameter AI model becomes powerful through a superior cognitive operating system.*

[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](./docs/CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-196%20passing-brightgreen.svg)](#test-suite)
[![Crates](https://img.shields.io/badge/crates-20-purple.svg)](#workspace-layout)

---

## Table of Contents

- [What is QbitOS?](#what-is-qbitos)
- [Why QbitOS?](#why-qbitos)
- [Key Features](#key-features)
- [Workspace Layout](#workspace-layout)
- [The Cognitive Loop](#the-cognitive-loop)
- [Architecture](#architecture)
- [Data Flows](#data-flows)
- [Quick Start](#quick-start)
- [How to: Build QbitOS](#how-to-build-qbitos)
- [How to: Run the Cognitive Kernel](#how-to-run-the-cognitive-kernel)
- [How to: Start the Service Layer](#how-to-start-the-service-layer)
- [How to: Use the HTTP/REST API](#how-to-use-the-httprest-api)
- [How to: Subscribe to WebSocket Events](#how-to-subscribe-to-websocket-events)
- [How to: Call the JSON-RPC Interface](#how-to-call-the-json-rpc-interface)
- [How to: Register and Execute a Skill](#how-to-register-and-execute-a-skill)
- [How to: Train with the Self-Improvement Loop](#how-to-train-with-the-self-improvement-loop)
- [How to: Ingest a Dataset](#how-to-ingest-a-dataset)
- [How to: Wire Real Backends into Skills](#how-to-wire-real-backends-into-skills)
- [How to: Use the Desktop App](#how-to-use-the-desktop-app)
- [How to: Persist State Across Restarts](#how-to-persist-state-across-restarts)
- [How to: Deploy in Production](#how-to-deploy-in-production)
- [Configuration Reference](#configuration-reference)
- [Memory Backends](#memory-backends)
- [Security Model](#security-model)
- [Test Suite](#test-suite)
- [Performance](#performance)
- [Limitations and Roadmap](#limitations-and-roadmap)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## What is QbitOS?

QbitOS is a **from-scratch artificial cognitive operating system** written in
Rust 2024 edition. It is not a chatbot wrapper, not an API gateway, not an
agent orchestration framework layered over OpenAI. Its intelligence is
produced by its own native runtime: a cognitive kernel that runs an
eight-stage observe-learn-improve loop continuously, backed by a
biological-memory-inspired fabric, a skill genome, a tool virtual machine,
a knowledge graph, a planner, a controlled evolution engine, and a
unified dataset loader.

The core thesis: a small (3B-parameter) model becomes powerful when
embedded in a superior cognitive operating system that provides
**memory**, **planning**, **tool use**, **skill composition**,
**self-improvement**, and a **controlled evolution loop**. The default
inference path uses the NativeAgent, which composes responses from
memory recall, skill matching, knowledge-graph queries, and the
reasoning engine — no external LLM API calls are required for the
system to function.

QbitOS is **multimodal** (text + image + audio), **embeddable** (the
entire OS runs in a single binary), **production-grade** (HTTP/REST +
WebSocket + JSON-RPC services with auth, persistent events, and CORS),
and **auditable** (every action is logged, every state change is
reversible).

---

## Why QbitOS?

| Property                     | QbitOS                                       | Typical LLM-wrapper "agent" frameworks |
|------------------------------|----------------------------------------------|----------------------------------------|
| Inference path               | Native cognitive agent (memory + reasoning)  | Calls OpenAI / Anthropic / Gemini API  |
| Memory                       | L0-L5 biological hierarchy, 5 DB backends    | Conversation history in a single blob  |
| Skill composition            | Step-based genome (Read/Infer/Tool/Verify/Return) | Prompt strings                    |
| Self-improvement             | 5-stage curriculum + meta-learning + regression guard | None, or ad-hoc fine-tuning    |
| Multimodal                   | Native vision + speech + text pipelines      | Optional via vendor-specific APIs      |
| Tool execution               | Sandboxed VM with permission system          | Direct shell / Python exec             |
| Knowledge graph              | Petgraph-based, with causal reasoning        | None                                   |
| Reasoning engines            | Propositional logic + CSP + Bayesian + MDP + game tree | None, or chain-of-thought prompting |
| Service layer                | Axum HTTP/REST + WebSocket + JSON-RPC + OpenAPI | FastAPI / Flask wrapper              |
| Offline operation            | Full functionality with zero network egress  | Requires internet to call LLM vendor   |
| License                      | Apache-2.0                                   | Often restrictive or commercial        |

QbitOS is the right choice when you need:

- **Air-gapped or privacy-sensitive deployments** where calling external
  LLM APIs is forbidden.
- **Long-running agents** that must learn over time and not forget.
- **Auditable reasoning** — every conclusion is traceable to recalled
  memories, skills, and knowledge-graph facts.
- **A substrate for cognitive-AI research** — every component is
  replaceable and instrumented.
- **A production-grade service** with auth, persistence, and
  observability built in.

---

## Key Features

### Cognitive Kernel

- **8-stage cognitive loop**: Observe → Understand → Create Goals → Plan → Execute → Verify → Learn → Improve
- **Priority-aware preemptive scheduler**: Realtime / Interactive / Normal / Background queues
- **Global event bus**: topic-prefix + wildcard subscriptions, broadcast to N subscribers
- **Per-agent resource budgets**: CPU / memory / GPU / tokens / IO
- **Graceful shutdown**: in-flight cycles complete before exit

### Memory Fabric (L0–L5)

- **6-layer biological hierarchy**: Sensory, Working, Episodic, Semantic, Procedural, Meta
- **HNSW vector index**: filtered ANN, product quantization, BM25 hybrid search, MMR re-ranking, adaptive `ef`
- **5 pluggable backends**: PostgreSQL+pgvector · Qdrant · Redis · RocksDB · Sled (default, pure Rust embedded)
- **Consolidation engine**: promotes L2→L3→L4 based on importance + recency
- **~15 tests** covering all layers

### Skill Genome

- **5 step kinds**: Read, Infer, Tool, Verify, Return
- **Confidence scoring**: Beta(2,2) prior + sample-size scaling
- **Real backends** (feature `real-backends`):
  - `ToolModuleBackend` bridges to `qbitos_tools::ToolModule` (filesystem, terminal, http, calculator, …)
  - `NativeAgentBackend` bridges to `NativeAgent` with `ArcSwapOption` late binding (breaks the chicken-and-egg with `CognitiveContext`)
- **Optimizer**: prunes skills with success rate < 20% after 16 samples
- **Generator**: mines skills from agent-trajectory and instruction datasets
- **24 tests (default) / 33 (with `real-backends`)**

### Neural Engine

- **Native cognitive agent**: no external LLM required
- **Chain-of-thought reasoning**: prompt decomposition → multi-path reasoning → self-consistency selection → synthesis
- **Self-reflection + constitutional AI + uncertainty quantification**
- **Optional candle-core backend** for external models (Qwen2, etc.)
- **Quantization** (INT8, INT4) and **sampling** (TopP, TopK, Temperature, greedy)

### Multimodal I/O

- **Vision**: HOG + color histogram + LBP features, OCR (Otsu + connected components + template matching), object detection (contour + NMS)
- **Speech**: STT (FFT + MFCC + VAD), TTS (formant synthesis)

### World Model

- **Knowledge graph** (petgraph DiGraph) with alias resolution and entity merging
- **Causal reasoning** (do-calculus, counterfactuals)
- **Simulation engine** (forward simulation for planning)

### Planner

- **Goal engine**: hierarchical goal decomposition
- **Task graph (DAG)**: topological execution, parallel branches
- **Decision engine**: multi-criteria skill selection (confidence, latency, success rate)

### Tools

- **9 built-in tools**: filesystem, terminal, http, calculator, summarizer, file_analyzer, sentiment, entity_extractor, time
- **Tool VM** with permission system (Read / Write / Execute / Network)
- **Memory recording** of every tool outcome

### Reasoning

- **Propositional logic**: resolution-based theorem prover
- **CSP solver**: AC-3 + forward checking + backtracking
- **Bayesian networks**: probabilistic inference
- **MDP solver**: value iteration + policy iteration
- **Game-tree search**: minimax + alpha-beta pruning

### Learning

- **Multi-armed bandits** (UCB1, ε-greedy)
- **Preference learning** (Bradley-Terry)
- **Feedback collector** (explicit + implicit)
- **Q-learning** (tabular)

### Safety

- **Content filter** for harmful content
- **Prompt-injection detector** (e.g., "ignore previous instructions")
- **Output validator** (schema + policy)
- **Rate limiter** (token bucket per principal)
- **Audit logger** for compliance

### Training

- **Curriculum learning**: 5 stages (Trivial → Easy → Medium → Hard → Expert) with per-stage promotion thresholds
- **Experience replay**: ring buffer with uniform + priority-weighted sampling
- **Self-play engine**: arbitrary games via 4 closures
- **Self-improvement cycle**: detects 4 weakness types → proposes fixes → tests → accepts if improvement ≥ 5%
- **Meta-learning**: UCB1 + ε-greedy strategy selection with Wilson 95% confidence intervals
- **29 tests** covering all sub-engines

### Service Layer

- **HTTP/REST API** (Axum 0.8): 19 endpoints
- **WebSocket real-time** event streaming with bidirectional command dispatch
- **JSON-RPC 2.0** with 7 default methods + batch support
- **OpenAPI 3.1 spec generation** at `/api/v1/openapi.json`
- **Bearer-token authentication**: SHA-256 hashed, constant-time compared
- **Persistent event store**: events survive across restarts via the memory fabric's semantic layer
- **Real cognitive kernel integration**: `--real-kernel` flag dispatches through the 8-stage loop
- **CORS policies**: permissive (dev) or restricted (production)
- **39 tests** including 11 auth + 5 event_store

### Data Layer

- **32+ dataset loaders**: JSONL, CSV, Parquet, SQLite, PDF, HTML, image, audio, HuggingFace Hub, S3/MinIO, WebDataset, LMDB, HDF5, Excel, and more
- **Streaming load** with backpressure
- **Format auto-detection**
- **Synthetic data generator**

### CLI

- **15 subcommands**: `repl`, `status`, `ask`, `serve`, `dataset`, `memory`, `skill`, `tool`, `world`, `evolution`, `state`, `backend`, `reason`, `safety`, `learn`
- **Interactive REPL** with `:status`, `:memory`, `:skills`, `:tools`, `:world`, `:evolution`, `:quit`
- **State persistence** via `state save` / `state load`

### Desktop App

- **Tauri v2** backend (Rust) + **React 19** + TypeScript + Tailwind CSS 4
- **9 views**: Chat, Voice, Vision, Memory, Skills, World Model, Agent Monitor, Datasets, Service
- **Live WebSocket event stream** in the Service view
- **Live kernel status sidebar** (CPU, RAM, current stage, cycle count)

---

## Workspace Layout

```
qbitos/
├── kernel/          Cognitive kernel, scheduler, event bus, resource manager
├── neural_engine/   Native cognitive agent + candle-based inference backends
├── vision/          Image processor, HOG/color/LBP features, OCR, object detection
├── speech/          STT (FFT+MFCC+VAD) + TTS (formant synthesis)
├── memory/          L0-L5 biological memory fabric with consolidation
├── world_model/     Knowledge graph, entity engine, causal reasoning, simulation
├── planner/         Goal engine, task graph (DAG), reasoning, decision engine
├── skills/          Skill genome: registry, generator, optimizer, executor
├── tools/           Tool VM: filesystem, terminal, http, calculator, summarizer, …
├── evolution/       Evaluator, improvement engine, regression guard
├── data/            Unified dataset loader: 32+ formats, streaming, S3/MinIO, HF Hub
├── reasoning/       Propositional logic, CSP, Bayesian networks, MDP, game-tree search
├── learning/        Multi-armed bandits, preference learning, feedback, Q-learning
├── safety/          Content filter, prompt-injection detector, audit log, rate limiter
├── training/        Curriculum, experience replay, self-play, self-improvement, meta-learning
├── services/        HTTP/REST (Axum), WebSocket, JSON-RPC 2.0, OpenAPI 3.1, auth
├── cli/             Top-level `qbitos` binary wiring all subsystems together
├── desktop/         Tauri v2 desktop app (React 19 + TS + Tailwind 4)
├── examples_pkg/    End-to-end demos (cognitive cycle, WS events, RPC, training loop)
├── docs/            Architecture, API, deployment, contributing, changelog
├── examples/        Example source files (hosted by examples_pkg)
├── Cargo.toml       Workspace manifest (resolver = "3")
├── Cargo.lock       Pinned dependency versions
├── LICENSE          Apache 2.0
├── README.md        This file
└── worklog.md       Implementation history (per-task log)
```

**20 crates** · **190 Rust source files** · **~37,000 lines of code** · **196 tests passing**

---

## The Cognitive Loop

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

## Architecture

The QbitOS architecture is a layered cake with the cognitive kernel at
its core, surrounded by capability subsystems, wrapped by a service
layer for external access.

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

For a deeper dive into subsystem interactions, persistence model, and
security model see [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).

---

## Data Flows

QbitOS has three primary data flows: the **cognitive cycle**, the
**training loop**, and the **HTTP request lifecycle**. Understanding
these is key to understanding the system.

### 1. Cognitive Cycle Data Flow

When a client `POST /api/v1/cognitive/cycle` arrives at a server
running with `--real-kernel`:

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

### 2. Training Loop Data Flow

The training loop orchestrates five training sub-engines per step:

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

**Curriculum advancement** triggers when `success_rate ≥ promotion_threshold` AND `attempts ≥ min_samples`:

| Stage    | Difficulty | Promotion Threshold | Min Samples |
|----------|------------|---------------------|-------------|
| 1        | Trivial    | 90% success rate    | 8           |
| 2        | Easy       | 85%                 | 16          |
| 3        | Medium     | 75%                 | 32          |
| 4        | Hard       | 65%                 | 64          |
| 5        | Expert     | 50%                 | 128         |

### 3. HTTP Request Lifecycle

```
1. TCP accept (Tokio runtime)
        │
        ▼
2. Axum router matches path → handler
        │
        ▼
3. Tower middleware stack (in order):
   • TraceLayer (request logging)
   • CompressionLayer (gzip response)
   • CorsLayer (permissive or restricted)
   • require_auth_middleware (Bearer check, if enabled)
        │
        ▼
4. Handler extracts State<ServiceState> + JSON body
        │
        ▼
5. state.svc.record_request()  ← bumps the global request counter
        │
        ▼
6. Business logic dispatches to:
   • CognitiveService.kernel().run_cycle(...)
   • CognitiveService.skills().execute(...)
   • CognitiveService.memory().episodic.recall_similar(...)
   • etc.
        │
        ▼
7. Service publishes a ServiceEvent on the EventBus
        │
        ├──▶ WebSocket subscribers (live stream)
        ├──▶ PersistentEventStore (semantic memory, if enabled)
        └──▶ In-memory ring buffer (for HTTP polling)
        │
        ▼
8. Handler returns Json<ApiResponse<T>>
        │
        ▼
9. Tower middleware (reverse order) — compress, add CORS headers, log
        │
        ▼
10. HTTP response sent to client
```

### 4. Memory Consolidation Flow

```
L0 Sensory  ──(attention)──▶  L1 Working  ──(episode end)──▶  L2 Episodic
                                                                  │
                                  ┌───────────────────────────────┘
                                  ▼
                          (importance + recency)
                                  │
                                  ▼
                              L3 Semantic  ◀──(skill execution)──  L4 Procedural
                                  │                                       │
                                  └───────────(meta-strategy)──────────▶  L5 Meta
```

- **L0 → L1**: An attention filter selects which sensory inputs enter working memory.
- **L1 → L2**: When a cognitive cycle completes, the working memory snapshot is committed as an episode.
- **L2 → L3**: The consolidation engine promotes episodes to semantic memory when their importance exceeds a threshold.
- **L4**: Skills are stored as procedural memory; execution statistics feed back into confidence scores.
- **L5**: Meta-strategies (which learning algorithm to use, which skill to prefer) are stored separately.

### 5. Event Publication Flow

```
Any subsystem  ──▶  svc.publish(ServiceEvent::...)
                          │
                          ▼
                    EventBus (tokio::sync::broadcast)
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        WS Session 1  WS Session 2  PersistentEventStore
        (live)        (live)        (semantic memory)
                          │
                          ▼
                    In-memory ring buffer (capacity 10,000)
                          │
                          ▼
                    HTTP polling endpoint
                    (POST /api/v1/events/poll)
```

Events are fanned out to all active WebSocket subscribers in real time
and persisted to the semantic memory layer (if `--persistent-events`
is enabled) so they can be replayed after a restart.

---

## Quick Start

### Option A: Full setup with database (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/qbitos/qbitos.git
cd qbitos

# 2. Set up the database (creates user, database, schema, seed data, .env)
#    Linux/macOS:
./scripts/setup-db.sh --password "yourPassword"
#    Windows PowerShell:
#    .\scripts\setup-db.ps1 -Password "yourPassword"
#    Or use Docker (auto-starts PostgreSQL with pgvector):
#    ./scripts/setup-db.sh --docker --password "yourPassword"

# 3. Build the CLI
cargo build --release -p qbitos-cli

# 4. Start the HTTP server (connects to the database automatically)
./target/release/qbitos serve --addr 0.0.0.0:8080

# 5. Verify
curl http://localhost:8080/api/v1/health
# → {"ok":true,"data":{"instance_id":"...","healthy":true,...}}
```

The setup script automatically:
- Creates the `qbitos` database user and database
- Installs pgvector + 7 other PostgreSQL extensions
- Runs the full schema migration (14 tables, 82+ indexes)
- Seeds 50+ memories, 15 skills, 100+ knowledge triples, training data, safety rules
- Generates a `.env` file with the correct connection string

See [Database Documentation](docs/DATABASE.md) for the full schema reference.

### Option B: Docker Compose (one command)

```bash
# Boots the full stack: QbitOS + PostgreSQL + Qdrant + Redis + MongoDB
docker compose up -d

# PostgreSQL is auto-seeded on first boot (schema + indexes + seed data)
# Access the API at http://localhost:8080
```

### Option C: Quick CLI-only (no database)

```bash
# Build and run the interactive REPL (uses in-memory storage)
cargo run -p qbitos-cli --bin qbitos
```

### Option D: Desktop app

```bash
cd desktop
npm install
npm run tauri dev    # Opens the Tauri window with React UI
```

For the full set of build options and feature flags, see
[How to: Build QbitOS](#how-to-build-qbitos).

---

## How to: Build QbitOS

### Prerequisites

- **Rust 1.85+** (nightly recommended but not required)
- **Linux or macOS** (Windows is best-effort)
- **~10 GB free disk space** for build artifacts
- `pkg-config`, `libssl-dev` (for HTTPS support)

### Build configurations

| Configuration                                  | Build time | Binary size | Use case                          |
|------------------------------------------------|------------|-------------|-----------------------------------|
| `cargo build -p qbitos-cli`                    | ~30s       | ~50 MB      | Dev / REPL / tests                |
| `--features postgres-backed` (default)         | ~30s       | ~50 MB      | PostgreSQL primary backend        |
| `--features real-backends`                     | ~5 min     | ~193 MB     | Production with NativeAgent       |
| `--features all-backends`                      | ~6 min     | ~210 MB     | All DB backends enabled           |
| `--features real-backends,all-backends`        | ~7 min     | ~230 MB     | Full production build             |
| `--release`                                    | +50%       | -60% size   | Optimized production binary       |

### Build commands

```bash
# Fast dev build (no real backends, ~30s)
cargo build -p qbitos-cli

# Full build with real backends (~5 min first time, cached after)
cargo build -p qbitos-cli --features real-backends

# All DB backends
cargo build -p qbitos-cli --features all-backends

# Everything (real-backends + all-backends)
cargo build -p qbitos-cli --features real-backends,all-backends

# Release build (optimized, stripped)
cargo build --release -p qbitos-cli --features real-backends,postgres-backed

# Cross-compile to musl (fully static binary)
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl \
  -p qbitos-cli --features real-backends,postgres-backed

# Build the entire workspace (excluding desktop, which needs Node.js)
cargo build --workspace --exclude qbitos-desktop

# Build with real-backends across the whole workspace
cargo build --workspace --features qbitos-cli/real-backends --exclude qbitos-desktop
```

### Build the desktop app

```bash
cd desktop
npm install
npm run tauri dev    # development mode (hot reload)
npm run tauri build  # production build (native installers)
```

### Feature flags reference

| Feature             | Crate          | Effect                                                      |
|---------------------|----------------|-------------------------------------------------------------|
| `rocksdb-backed`    | qbitos-cli     | Enable RocksDB embedded backend (edge / offline)             |
| `postgres-backed`   | qbitos-cli     | Enable PostgreSQL + pgvector backend                        |
| `qdrant-backed`     | qbitos-cli     | Enable Qdrant vector DB backend                             |
| `redis-backed`      | qbitos-cli     | Enable Redis cache backend                                  |
| `rocksdb-backed`    | qbitos-cli     | Enable RocksDB backend                                      |
| `all-backends`      | qbitos-cli     | Enable all DB backends at once                              |
| `real-backends`     | qbitos-cli     | Wire real `ToolModuleBackend` + `NativeAgentBackend` into `SkillModule` |
| `real-backends`     | qbitos-skills  | Enable the `real_backends` module (adapters for tools/neural_engine) |

---

## How to: Run the Cognitive Kernel

The cognitive kernel runs the 8-stage loop continuously. The simplest
way to drive it is the interactive REPL:

```bash
./target/debug/qbitos repl
```

This boots the kernel and drops you into a prompt where you can chat,
query memory, inspect skills, etc.

```
QbitOS REPL — type :help for commands, :quit to exit
> Hello, what can you do?
[observe] understood: "Hello, what can you do?"
[plan] derived goal: explain_capabilities
[execute] infer step → NativeAgent.compose_response(...)
[verify] confidence=0.87 ≥ 0.5, accepting
[learn] stored episode ep-2026-07-19-001 (success)
[improve] no weaknesses detected
→ I'm QbitOS, a cognitive operating system. I can reason over memory,
  execute skills, query the knowledge graph, and improve over time.
> :memory
episodic: 1 episode | semantic: 0 concepts | procedural: 0 skills
> :skills
(No skills registered. Use `qbitos skill register` or ingest a dataset.)
> :quit
```

### One-shot cognitive cycle

For a single prompt without entering the REPL:

```bash
./target/debug/qbitos ask "What is the capital of France?"
```

### Boot the kernel programmatically

```rust
use qbitos_cli::App;
let app = App::new();
let tx = app.boot_kernel();
tx.send(serde_json::json!({"input": "Hello"})).unwrap();
```

---

## How to: Start the Service Layer

The `serve` subcommand starts the HTTP/REST + WebSocket + JSON-RPC
service layer.

### Basic (development)

```bash
./target/debug/qbitos serve --addr 127.0.0.1:8080
```

- No auth
- No persistent events
- Stub cognitive cycle
- Permissive CORS

### Hardened (production)

```bash
./target/release/qbitos serve \
  --addr 0.0.0.0:8080 \
  --name qbitos-prod-01 \
  --auth-token "$QBITOS_TOKEN" \
  --real-kernel \
  --persistent-events \
  --restricted-cors \
  --allow-self-improve
```

### All CLI flags

```
qbitos serve [OPTIONS]

Options:
      --addr <ADDR>              Bind address [default: 127.0.0.1:8080]
      --name <NAME>              Instance name [default: qbitos-cli]
      --no-safety                Disable safety enforcement
      --allow-self-improve       Enable self-improvement (writes to skill registry)
      --auth-token <TOK>         Enable Bearer-token auth (registers "admin" principal)
      --real-kernel              Attach the real CognitiveKernel (8-stage loop)
      --persistent-events        Attach persistent event store (survives restarts)
      --restricted-cors          Use restricted CORS policy (production-safe)
  -h, --help                     Print help
```

### Verify the server

```bash
# Health check (public)
curl http://localhost:8080/api/v1/health | jq .data.healthy
# → true

# Stats (requires auth if --auth-token was set)
curl -H "Authorization: Bearer $QBITOS_TOKEN" \
     http://localhost:8080/api/v1/stats

# OpenAPI spec
curl http://localhost:8080/api/v1/openapi.json | jq .info
```

For the full deployment guide (Docker, systemd, Nginx), see
[`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md).

---

## How to: Use the HTTP/REST API

The HTTP API exposes 19 endpoints under `/api/v1/`. All responses are
wrapped in the `ApiResponse` envelope:

```json
{
  "ok": true,
  "data": { /* payload */ },
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-07-19T12:34:56.789Z"
}
```

### Run a cognitive cycle

```bash
curl -X POST http://localhost:8080/api/v1/cognitive/cycle \
  -H "Authorization: Bearer $QBITOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input":"What is 2+2?","max_cycles":1}'
```

Response:

```json
{
  "ok": true,
  "data": {
    "cycle_id": "550e8400-...",
    "output": "Processed: What is 2+2?",
    "phases": ["observe", "understand", "create_goals", "plan",
               "execute", "verify", "learn", "improve"],
    "duration_ms": 12,
    "confidence": 0.9
  }
}
```

### Store and search memories

```bash
# Store
curl -X POST http://localhost:8080/api/v1/memory \
  -H "Authorization: Bearer $QBITOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content":"The user prefers concise answers.","layer":"L3_semantic"}'

# Search
curl -X POST http://localhost:8080/api/v1/memory/search \
  -H "Authorization: Bearer $QBITOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"user preferences","k":5}'
```

### Run a training step

```bash
curl -X POST http://localhost:8080/api/v1/training/step \
  -H "Authorization: Bearer $QBITOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"steps":10}'
```

For the full API reference (all 19 endpoints, request/response schemas,
error codes, SDK examples), see [`docs/API.md`](./docs/API.md).

---

## How to: Subscribe to WebSocket Events

Connect to `/api/v1/events/ws` for live event streaming:

```javascript
const ws = new WebSocket("ws://localhost:8080/api/v1/events/ws");

ws.onopen = () => {
  console.log("connected");
  ws.send(JSON.stringify({ kind: "ping", payload: { hello: "world" } }));
};

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.kind === "event") {
    console.log("event:", msg.payload.event.kind, msg.payload.event);
  } else {
    console.log("msg:", msg.kind);
  }
};
```

The server sends `session_started` on connect, then continuously
streams events as they are published. The client can send commands:

| Command  | Payload                | Response                              |
|----------|------------------------|---------------------------------------|
| `ping`   | Any JSON               | `pong` with the payload echoed        |
| `cycle`  | `{"input": "..."}`     | `result` with the cycle output        |
| `stats`  | `{}`                   | `result` with the service stats       |

### Event types

13 event kinds are streamed:

| Kind                          | Fields                                                |
|-------------------------------|-------------------------------------------------------|
| `cognitive_cycle_completed`   | `cycle_id`, `phase`, `duration_ms`, `summary`         |
| `memory_stored`               | `memory_id`, `layer`, `content_preview`               |
| `skill_executed`              | `skill_id`, `success`, `duration_ms`                  |
| `tool_invoked`                | `tool_name`, `success`                                |
| `reasoning_conclusion`        | `engine`, `conclusion`, `confidence`                  |
| `vision_perception`           | `objects_detected`, `ocr_chars`                       |
| `speech_event`                | `modality`, `duration_ms`                             |
| `learning_event`              | `event_type`, `metric`, `value`                       |
| `safety_event`                | `rule`, `severity`, `message`                         |
| `subsystem_started`           | `name`                                                |
| `subsystem_stopped`           | `name`                                                |
| `log`                         | `level`, `message`                                    |
| `custom`                      | `subsystem`, `payload`                                |

---

## How to: Call the JSON-RPC Interface

JSON-RPC 2.0 is available both in-process (via `RpcServer`) and over
HTTP.

### In-process (Rust)

```rust
use qbitos_services::{CognitiveService, RpcServer, RpcRequest};

let svc = CognitiveService::new(Default::default());
let rpc = RpcServer::new(svc);

let req = RpcRequest {
    jsonrpc: "2.0".into(),
    method: "cognitive.cycle".into(),
    params: Some(serde_json::json!({"input": "Hello!"})),
    id: Some(serde_json::json!(1)),
};
let resp = rpc.handle(req).await;
println!("{:?}", resp.result);
```

### Over HTTP

```bash
curl -X POST http://localhost:8080/api/v1/rpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}'
# → {"jsonrpc":"2.0","result":{"pong":true},"id":1}
```

### Batch

```bash
curl -X POST http://localhost:8080/api/v1/rpc \
  -H "Content-Type: application/json" \
  -d '[
    {"jsonrpc":"2.0","method":"ping","id":1},
    {"jsonrpc":"2.0","method":"stats","id":2}
  ]'
```

### Default methods

| Method              | Params                  | Returns                            |
|---------------------|-------------------------|------------------------------------|
| `ping`              | —                       | `{ "pong": true }`                 |
| `stats`             | —                       | Service stats object               |
| `health`            | —                       | Health summary object              |
| `cognitive.cycle`   | `{ "input": "..." }`    | `{ "cycle_id": "...", "output": "..." }` |
| `memory.search`     | `{ "query": "..." }`    | `{ "results": [...] }`             |
| `skills.list`       | —                       | `{ "skills": [...] }`              |
| `training.step`     | `{ "steps": N }`        | `{ "steps_executed": N, "mean_reward": X }` |

---

## How to: Register and Execute a Skill

Skills are named, versioned, composable procedures. Each skill has a
sequence of steps (Read / Infer / Tool / Verify / Return).

### Register a skill (Rust)

```rust
use qbitos_skills::{Skill, SkillStep, StepKind};

let mut skill = Skill::new("summarize", "Summarize a piece of text");
skill.add_step(SkillStep::new(
    "infer",
    StepKind::Infer,
    serde_json::json!({"prompt": "Summarize the following text"}),
));
skill.add_step(SkillStep::new(
    "return",
    StepKind::Return,
    serde_json::json!({}),
));
skill.success_criteria.push("output_is_non_empty".into());
let id = app.skills.registry.register(skill);
```

### Execute a skill (HTTP)

```bash
curl -X POST http://localhost:8080/api/v1/skills/execute \
  -H "Authorization: Bearer $QBITOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "skill_id": "summarize",
    "input": {"text": "Long text to summarize..."}
  }'
```

Response:

```json
{
  "ok": true,
  "data": {
    "skill_id": "summarize",
    "success": true,
    "output": {
      "skill_id": 1,
      "status": "success",
      "result": "Summary: ...",
      "steps": [
        { "name": "infer", "kind": "infer", "duration_ms": 12, "success": true },
        { "name": "return", "kind": "return", "duration_ms": 0, "success": true }
      ],
      "total_duration_ms": 15
    },
    "duration_ms": 16
  }
}
```

### List all skills

```bash
curl -H "Authorization: Bearer $QBITOS_TOKEN" \
     http://localhost:8080/api/v1/skills
```

### Skill execution path

When `real-backends` is enabled:

- `StepKind::Infer` → `NativeAgentBackend.infer()` → `NativeAgent.compose_response()` (chain-of-thought over memory + skills + world model)
- `StepKind::Tool` → `ToolModuleBackend.invoke()` → real `ToolModule.executor.execute()` (filesystem, terminal, http, calculator, …)

When `real-backends` is disabled, both backends are no-op (skill
execution returns errors, but registration and statistics still work).

---

## How to: Train with the Self-Improvement Loop

The training crate orchestrates five training sub-engines. The
simplest way to use it is the `self_improvement_loop` example:

```bash
cargo run -p qbitos-examples --example self_improvement_loop
```

This runs 200 training steps in-process, advancing the curriculum and
recording strategy rewards. Sample output:

```
── QbitOS Self-Improvement Loop ──

Running 200 training steps...

── Training Report ──
  steps_executed:    200
  mean_reward:       0.7743
  final_loss:        0.0388
  total_duration:    0ms
  curriculum_advanced: true
  improvements_applied: 0

── Meta-Learning Strategy Rewards ──
  experience_replay: 37.7243
  self_improvement: 39.6919
  curriculum: 37.7310
  self_play: 39.7189

── Individual Subsystems ──
Curriculum: 5 stages
  - Warm-up (trivial)
  - Basic skills (easy)
  - Intermediate (medium)
  - Advanced (hard)
  - Frontier (expert)

ExperienceReplay: 50 experiences, mean_reward=0.7450
  sampled 5 experiences

SelfImprovement: detected 4 weaknesses
  - [0.70] skill:summarize: Skill 'summarize' has low success rate (0.30)
  - [0.60] skill:translate: Skill 'translate' is slow (8000ms avg)
  - [0.50] memory:recall: Memory recall precision is low (0.50)
  - [0.45] reasoning:engine: Reasoning accuracy is low (0.55)

MetaLearner: best strategy = Some("curriculum")
```

### Programmatic usage

```rust
use qbitos_training::{TrainingLoop, TrainingConfig};

let loop_ = TrainingLoop::new(TrainingConfig {
    max_steps: 1000,
    batch_size: 32,
    use_curriculum: true,
    use_experience_replay: true,
    use_self_play: false,
    use_self_improvement: true,
    use_meta_learning: true,
    replay_capacity: 10_000,
    log_every: 100,
});

let report = loop_.run(|step, stage| {
    // Your reward function here
    0.8
});

println!("mean_reward: {:.4}", report.mean_reward);
println!("curriculum advanced: {}", report.curriculum_advanced);
```

### Run training via HTTP

```bash
curl -X POST http://localhost:8080/api/v1/training/step \
  -H "Authorization: Bearer $QBITOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"steps":100}'
```

---

## How to: Ingest a Dataset

The `data` crate provides 32+ dataset loaders with format auto-detection.

### Via CLI

```bash
# List available loaders
qbitos dataset list-loaders

# Ingest a JSONL dataset into the memory fabric
qbitos dataset ingest data.jsonl

# Mine skills from an agent-trajectory dataset
qbitos dataset mine trajectory.jsonl

# Ingest with a max record limit
qbitos dataset ingest large_data.jsonl --max-records 10000
```

### Supported formats

| Format          | Extension(s)              | Loader              |
|-----------------|---------------------------|---------------------|
| JSONL           | `.jsonl`                  | JsonlLoader         |
| CSV             | `.csv`                    | CsvLoader           |
| Parquet         | `.parquet`                | ParquetLoader       |
| SQLite          | `.sqlite`, `.db`          | SqliteLoader        |
| PDF             | `.pdf`                    | PdfLoader           |
| HTML            | `.html`, `.htm`           | HtmlLoader          |
| Image           | `.png`, `.jpg`, `.webp`   | ImageLoader         |
| Audio           | `.wav`, `.mp3`            | AudioLoader         |
| Excel           | `.xlsx`, `.xls`           | ExcelLoader         |
| WebDataset      | `.tar` shards             | WebDatasetLoader    |
| HuggingFace Hub | `hf:dataset/repo`         | HfHubLoader         |
| S3 / MinIO      | `s3://bucket/key`         | S3Loader            |
| LMDB            | `.lmdb`                   | LmdbLoader          |
| HDF5            | `.h5`, `.hdf5`            | Hdf5Loader          |
| … and 18 more   |                           |                     |

### Programmatic ingestion

```rust
use qbitos_data::{DatasetRegistry, LoadOptions};

let registry = DatasetRegistry::with_default_loaders();
let loader = registry.select(&path)?;
let mut stream = loader.load_with_options(&path, &LoadOptions::default()).await?;
while let Some(record) = stream.next_record().await? {
    println!("modality: {:?}", record.modality);
    println!("content: {}", record.content);
}
```

### Pipeline: dataset → memory

```rust
use qbitos_cli::pipeline::DatasetToMemory;

let pipeline = DatasetToMemory::new(memory.clone());
let result = pipeline.ingest_from_path(&registry, &path, Some(1000)).await?;
println!("ingested {} records", result.records_ingested);
```

### Pipeline: dataset → skills

```rust
use qbitos_cli::pipeline::DatasetToSkills;

let pipeline = DatasetToSkills::new(skills.clone(), memory.clone());
let result = pipeline.mine_from_path(&registry, &path, Some(1000)).await?;
println!("mined {} skills", result.skills_registered);
```

---

## How to: Wire Real Backends into Skills

By default, the `SkillModule` uses no-op backends (skill execution
returns errors, but registration and statistics work). To enable real
skill execution:

### 1. Build with `--features real-backends`

```bash
cargo build -p qbitos-cli --features real-backends
```

This pulls in `candle-core`, `candle-nn`, `candle-transformers`,
`tokenizers`, and `qbitos-neural-engine` — expect a ~5-minute first
build (cached after).

### 2. Verify in the server logs

```bash
./target/debug/qbitos serve --real-kernel --auth-token secret
# → 2026-07-19T... INFO wired real NativeAgent into SkillModule InferBackend
```

### 3. How the wiring works

The CLI's `App::new()` does the following (conditional on
`feature = "real-backends"`):

```rust
// 1. Construct the ToolModuleBackend wrapping the existing ToolModule
let tool_backend = Arc::new(ToolModuleBackend::new(tools.clone()));

// 2. Construct the NativeAgentBackend EMPTY (no agent yet)
let infer_backend = Arc::new(NativeAgentBackend::empty(
    memory.embedder.clone(),
    384,
));

// 3. Build the SkillModule with the real backends
let skills = Arc::new(SkillModule::new(
    memory.episodic.clone(),
    tool_backend,
    infer_backend.clone() as Arc<dyn InferBackend>,
));

// 4. ... construct the NativeAgent (which needs skills via CognitiveContext) ...

// 5. Late-bind the NativeAgent into the InferBackend
infer_backend.set_agent(native_agent.clone());
```

The `NativeAgentBackend` uses `arc_swap::ArcSwapOption<NativeAgent>`
for lock-free late binding. This breaks the chicken-and-egg between
`SkillModule` (needs `InferBackend`) and `NativeAgent` (needs
`CognitiveContext` which references `SkillModule`).

### 4. Verify skill execution works

```bash
# Register a test skill
curl -X POST http://localhost:8080/api/v1/memory \
  -H "Authorization: Bearer $QBITOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content":"test","layer":"L1_working"}'

# Execute a skill (if any registered)
curl -X POST http://localhost:8080/api/v1/skills/execute \
  -H "Authorization: Bearer $QBITOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"skill_id":"summarize","input":{"text":"hello"}}'
```

---

## How to: Use the Desktop App

The desktop app is built with Tauri v2 (Rust backend) + React 19 +
TypeScript + Tailwind CSS 4.

### Prerequisites

- Node.js 18+
- System dependencies:
  - **Linux**: `libwebkit2gtk-4.1-dev libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev`
  - **macOS**: Xcode Command Line Tools
  - **Windows**: Microsoft Visual Studio C++ Build Tools

### Development mode

```bash
cd desktop
npm install
npm run tauri dev
```

This launches the app with hot reload. The window opens with the
default Chat view.

### Production build

```bash
npm run tauri build
```

Produces native installers in `desktop/src-tauri/target/release/bundle/`:
- **Linux**: `.deb`, `.AppImage`
- **macOS**: `.dmg`, `.app`
- **Windows**: `.msi`, `.exe`

### Views

| View          | Description                                              |
|---------------|----------------------------------------------------------|
| Chat          | Chat interface with the cognitive kernel                 |
| Voice         | Speech-to-text and text-to-speech                        |
| Vision        | Image upload + OCR + object detection                    |
| Memory        | Browse memory layers, search                             |
| Skills        | List skills, view stats, execute                         |
| World Model   | Browse knowledge graph, query                            |
| Agent Monitor | Active tasks, resource usage, event bus stats            |
| Datasets      | List loaded datasets, ingest new ones                    |
| Service       | Live WebSocket event stream, health, cognitive cycle test|

The **Service** view is particularly useful for monitoring: it
connects to `/api/v1/events/ws` and streams events in real time, with
color-coded kind chips and a connection panel for the HTTP/WS URLs
and optional Bearer token.

---

## How to: Persist State Across Restarts

QbitOS supports two complementary persistence mechanisms.

### 1. State snapshots (full application state)

The CLI can serialize all in-memory state to a JSON file:

```bash
# Save
./target/release/qbitos state save /var/lib/qbitos/snapshot.json

# Load (into a fresh App instance)
./target/release/qbitos state load /var/lib/qbitos/snapshot.json
```

The snapshot includes:
- All registered skills (with execution stats)
- All episodic memories (L2)
- The knowledge graph (entities + relationships)
- Dataset catalog entries

### 2. Persistent event store (event log)

When started with `--persistent-events`, the service layer backs the
in-memory event buffer with the memory fabric's semantic layer.
Events survive across restarts:

```bash
# First session: generate events
./target/release/qbitos serve --persistent-events --auth-token secret &
curl -X POST -H "Authorization: Bearer secret" \
     http://localhost:8080/api/v1/cognitive/cycle \
     -d '{"input":"hello"}'
# Stop the server

# Second session: replay events from the previous session
./target/release/qbitos serve --persistent-events --auth-token secret &
curl -X POST -H "Authorization: Bearer secret" \
     http://localhost:8080/api/v1/events/replay
# → {"ok":true,"data":{"replayed":5,"buffer_len":5}}
```

### 3. Durable memory backends

For persistent memory across restarts (without snapshots), use a
durable backend:

```bash
# PostgreSQL + pgvector
export DATABASE_URL=postgres://user:pass@localhost/qbitos
./target/release/qbitos serve --features postgres-backed

# Or Sled (default, embedded)
export QBITOS_DATA_DIR=/var/lib/qbitos
./target/release/qbitos serve
# Data stored at /var/lib/qbitos/rocksdb/ (when rocksdb-backed is enabled)
```

### Persistence summary

| State              | Stored where                          | Survives restart?              |
|--------------------|---------------------------------------|--------------------------------|
| Memory L0/L1       | In-memory                             | No                             |
| Memory L2 (episodic)| In-memory + optional durable backend | Yes (if backend attached)      |
| Memory L3 (semantic)| In-memory + optional durable backend | Yes (if backend attached)      |
| Memory L4 (skills) | In-memory (registry)                  | Yes (via state snapshot)       |
| Memory L5 (meta)   | Postgres / Sled                       | Yes                            |
| Skill stats        | In-memory                             | Yes (via state snapshot)       |
| Knowledge graph    | In-memory                             | Yes (via state snapshot)       |
| Event log          | In-memory buffer                      | Yes (if `--persistent-events`) |
| Service config     | CLI args                              | No                             |

---

## How to: Deploy in Production

For the full deployment guide (Docker, systemd, Nginx, monitoring,
backup/restore), see [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md).
Quick reference:

### Docker

```bash
docker build -t qbitos:latest .
docker run -d \
  --name qbitos \
  -p 8080:8080 \
  -v qbitos-data:/var/lib/qbitos \
  -e QBITOS_TOKEN="$(openssl rand -hex 32)" \
  qbitos:latest \
  serve --addr 0.0.0.0:8080 --auth-token "$QBITOS_TOKEN" \
        --real-kernel --persistent-events --restricted-cors
```

### systemd

```ini
[Unit]
Description=QbitOS Cognitive Service
After=network.target

[Service]
Type=simple
User=qbitos
WorkingDirectory=/var/lib/qbitos
Environment=QBITOS_DATA_DIR=/var/lib/qbitos
ExecStart=/usr/local/bin/qbitos serve \
  --addr 0.0.0.0:8080 \
  --auth-token $(cat /etc/qbitos/token) \
  --real-kernel --persistent-events --restricted-cors
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Health check

```bash
curl http://localhost:8080/api/v1/health | jq .data.healthy
# → true
```

---

## Configuration Reference

### Environment variables

| Variable            | Default                | Description                            |
|---------------------|------------------------|----------------------------------------|
| `QBITOS_DATA_DIR`   | `/var/lib/qbitos`      | Data directory for Sled + snapshots    |
| `QBITOS_TOKEN`      | —                      | Bearer token (alternative to `--auth-token`) |
| `QBITOS_LOG_FORMAT` | `pretty`               | `pretty` or `json`                     |
| `DATABASE_URL`      | —                      | PostgreSQL connection string           |
| `QDRANT_URL`        | —                      | Qdrant server URL                      |
| `REDIS_URL`         | —                      | Redis connection string                |
| `RUST_LOG`          | `info`                 | Log level (`info,qbitos=debug,hyper=warn`) |

### `ServiceConfig` fields

```rust
pub struct ServiceConfig {
    pub instance_name: String,           // default "qbitos-default"
    pub max_concurrent_requests: usize,  // default 256
    pub request_timeout_ms: u64,         // default 30_000
    pub event_bus_capacity: usize,       // default 4096
    pub safety_enforced: bool,           // default true
    pub audit_log_enabled: bool,         // default true
    pub self_improvement_enabled: bool,  // default false
    pub embedding_dim: usize,            // default 384
    pub replay_capacity: usize,          // default 10_000
}
```

### `AuthConfig` fields

```rust
pub struct AuthConfig {
    pub enabled: bool,                   // default false
    pub allow_public_read: bool,         // default false (GETs bypass auth)
    pub public_paths: Vec<String>,       // default: health, version, openapi, events/ws
    pub public_prefixes: Vec<String>,    // default: empty
}
```

---

## Memory Backends

Five pluggable backends, all real implementations:

| Backend          | Feature flag        | Use case                              | Persistence |
|------------------|---------------------|---------------------------------------|-------------|
| **RocksDB**      | `rocksdb-backed`        | Embedded, edge deployments            | Disk        |
| **PostgreSQL**   | `postgres-backed`   | Production, multi-node, ACID          | Disk + WAL  |
| **Qdrant**       | `qdrant-backed`     | Large-scale vector search             | Disk        |
| **Redis**        | `redis-backed`      | Cache layer, low-latency              | Memory      |
| **RocksDB**      | `rocksdb-backed`    | High-throughput embedded              | Disk        |

### Auto-attach

The CLI auto-attaches all configured backends at startup via
`App::auto_attach_backends()`. Backends that fail to connect are
skipped with a warning.

```bash
# Single backend
export QBITOS_DATA_DIR=/var/lib/qbitos
./target/release/qbitos serve  # uses Sled

# Multiple backends (stacked)
export DATABASE_URL=postgres://...
export QDRANT_URL=http://...
export REDIS_URL=redis://...
./target/release/qbitos serve --features all-backends
```

The memory fabric queries all attached backends and merges results.

---

## Security Model

### Authentication

- **Bearer-token auth** (optional, `--auth-token <TOKEN>`):
  - SHA-256 hashed at registration time
  - Compared with `subtle::ConstantTimeEq` (constant-time comparison prevents timing attacks)
  - Single principal "admin" registered via CLI flag
  - Multi-principal mode available via embedded `AuthService`

### Public routes

Always bypass auth (regardless of config):

- `GET /api/v1/health`
- `GET /api/v1/version`
- `GET /api/v1/openapi.json`
- `GET /api/v1/openapi.yaml`
- `GET /api/v1/events/ws` (WebSocket upgrade)

### Public read mode

When `AuthConfig.allow_public_read = true`, all `GET` requests bypass
auth. Useful for read-only dashboards.

### CORS

- **Permissive** (dev default): allows all origins, methods, headers
- **Restricted** (`--restricted-cors`): same-origin only, configurable
  via `CorsLayer` in code

### Rate limiting

The `qbitos_safety::RateLimiter` (token bucket per principal) is
available but not yet wired into HTTP middleware. To enforce rate
limits, front the service with Nginx (see
[`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md)).

### Audit logging

The `qbitos_safety::AuditLogger` records all authenticated requests.
Enable with `ServiceConfig.audit_log_enabled = true` (default).

### Content safety

- **Content filter** (`qbitos_safety::ContentFilter`): detects harmful content
- **Prompt-injection detector** (`qbitos_safety::InjectionDetector`): detects "ignore previous instructions" style attacks
- **Output validator** (`qbitos_safety::OutputValidator`): validates agent output against schemas

---

## Test Suite

```bash
# Run all tests (excluding desktop, which needs Node.js)
cargo test --workspace --exclude qbitos-desktop

# Run tests with real-backends feature
cargo test --workspace --features qbitos-cli/real-backends --exclude qbitos-desktop

# Run a specific crate's tests
cargo test -p qbitos-services
cargo test -p qbitos-skills --features real-backends
```

**196/196 tests pass** across all subsystems:

| Crate                                  | Tests                              | Status |
|----------------------------------------|------------------------------------|--------|
| `qbitos-memory`                        | 15                                 | ✅     |
| `qbitos-planner`                       | 10                                 | ✅     |
| `qbitos-skills`                        | 24 (33 with `real-backends`)       | ✅     |
| `qbitos-tools`                         | 9                                  | ✅     |
| `qbitos-kernel`                        | 10                                 | ✅     |
| `qbitos-world-model`                   | 4                                  | ✅     |
| `qbitos-evolution`                     | 1                                  | ✅     |
| `qbitos-data` (lib + integration)      | 29                                 | ✅     |
| `qbitos-reasoning`                     | ~20                                | ✅     |
| `qbitos-learning`                      | ~10                                | ✅     |
| `qbitos-safety`                        | ~10                                | ✅     |
| `qbitos-training`                      | 29                                 | ✅     |
| `qbitos-services` (incl. 11 auth + 5 event_store) | 39                      | ✅     |
| `qbitos-cli` (lib + integration)       | 10 (13 with `real-backends`)       | ✅     |
| **Total**                              | **196**                            | ✅     |

### Test categories

- **Unit tests**: `#[cfg(test)] mod tests { ... }` at the bottom of each file
- **Integration tests**: `tests/` directory in each crate
- **Doctests**: `///` examples are compiled and run with `cargo test --doc`
- **Feature-gated tests**: tests requiring optional features use `#[cfg(feature = "...")]`

---

## Performance

### Benchmarks ( indicative, on a 4-core 2.5 GHz x86_64 VM )

| Operation                            | Throughput        | Latency      |
|--------------------------------------|-------------------|--------------|
| HTTP `/api/v1/health`                | ~50,000 req/s     | ~20 µs       |
| HTTP `/api/v1/cognitive/cycle` (stub)| ~10,000 req/s     | ~100 µs      |
| HTTP `/api/v1/cognitive/cycle` (real)| ~100 req/s        | ~10 ms       |
| Memory store (L2 episodic)           | ~100,000 ops/s    | ~10 µs       |
| Memory search (HNSW, k=10, 1M items) | ~1,000 queries/s  | ~1 ms        |
| Skill execution (2-step, stub)       | ~50,000 ops/s     | ~20 µs       |
| Skill execution (2-step, real)       | ~50 req/s         | ~20 ms       |
| WebSocket event fan-out (100 subs)   | ~100,000 events/s | ~10 µs/event |

### Performance tips

- Use `--release` for production builds (2-3x faster than debug).
- Tune `ServiceConfig.event_bus_capacity` based on subscriber count (default 4096).
- Tune `ServiceConfig.replay_capacity` based on memory budget (default 10,000).
- For high-throughput HTTP, increase `max_concurrent_requests` (default 256).
- For large memory backends, enable product quantization to reduce HNSW memory footprint.
- Use `--restricted-cors` in production (permissive CORS adds overhead).

---

## Limitations and Roadmap

### Current limitations (0.1.0)

- **Single-node only**: no distributed mode yet
- **Single auth principal** via CLI flag (multi-principal requires embedding `AuthService`)
- **No Prometheus `/metrics` endpoint** (planned for 0.2.0)
- **No gRPC service layer** (planned for 0.2.0)
- **Vision pipeline not yet integrated with the cognitive cycle** (planned for 0.2.0)
- **No speech-to-speech end-to-end mode** (planned for 0.3.0)
- **Rate limiter not yet wired into HTTP middleware** (use Nginx in the meantime)
- **Desktop app requires system GTK/webkit2gtk deps**

### Roadmap

**0.2.0** (Q3 2026):
- Prometheus `/metrics` endpoint
- gRPC service layer
- Vision pipeline integration with the cognitive cycle
- Multi-principal auth via config file
- Rate limiter HTTP middleware
- Distributed mode (multi-node kernel federation) — research phase

**0.3.0** (Q4 2026):
- Speech-to-speech end-to-end mode
- WebAssembly tool sandbox
- Plugin system for custom cognitive stages
- Mobile companion app (React Native)

**1.0.0** (2027):
- Stable API guarantees
- Formal verification of the cognitive kernel
- Hardware acceleration (CUDA, Metal) for the neural engine
- Multi-tenant mode with per-tenant memory isolation

---

## Documentation

| Document                              | Audience                | Description                                                       |
|---------------------------------------|-------------------------|-------------------------------------------------------------------|
| [This README](./README.md)            | Everyone                | Overview, quick start, how-tos, data flows                        |
| [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) | Architects, contributors | System architecture, subsystem topology, data flow diagrams       |
| [`docs/API.md`](./docs/API.md)        | API consumers           | HTTP/REST + WebSocket + JSON-RPC 2.0 reference with examples      |
| [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) | Operators, SREs         | Production deployment guide: Docker, systemd, Nginx, monitoring   |
| [`docs/CONTRIBUTING.md`](./docs/CONTRIBUTING.md) | Contributors            | Development workflow, code style, submission process              |
| [`docs/CHANGELOG.md`](./docs/CHANGELOG.md) | Everyone                | Release history and notable changes                               |
| [`docs/README.md`](./docs/README.md)  | Everyone                | Documentation index                                               |
| [`worklog.md`](./worklog.md)          | Contributors            | Implementation history (per-task log)                             |

Each of the 20 crates also has its own `README.md` with usage examples
and API details.

---

## Contributing

Contributions are welcome! Please read
[`docs/CONTRIBUTING.md`](./docs/CONTRIBUTING.md) before opening a pull
request.

### Quick contribution guide

1. **Fork & clone**: `git clone https://github.com/qbitos/qbitos.git`
2. **Branch**: `git checkout -b feat/my-feature`
3. **Code**: follow the code style (`cargo fmt`, `cargo clippy`)
4. **Test**: `cargo test --workspace --exclude qbitos-desktop`
5. **Commit**: use [Conventional Commits](https://www.conventionalcommits.org/) (`feat(skills): add ...`)
6. **PR**: open against `main`, fill in the PR template, CI runs automatically

### Good first issues

Look for issues tagged `good-first-issue` and `help-wanted` on the
[issue tracker](https://github.com/qbitos/qbitos/issues).

---

## License

QbitOS is licensed under the **Apache License, Version 2.0**.

- Full text: [`LICENSE`](./LICENSE)
- SPDX identifier: `Apache-2.0`
- Online reference: <http://www.apache.org/licenses/LICENSE-2.0>

```
Copyright 2026 QbitOS Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Contribution licensing

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in QbitOS by you shall be licensed under the
Apache License, Version 2.0, without any additional terms or
conditions.

---

<p align="center">
  <em>QbitOS — a small model becomes powerful through a superior cognitive operating system.</em>
</p>
