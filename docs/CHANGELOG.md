# Changelog

All notable changes to QbitOS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Production Deployment Infrastructure

- `Dockerfile` - Multi-stage Docker build (Rust 1.85 builder + Debian slim runtime)
- `docker-compose.yml` - Full stack: QbitOS + PostgreSQL (pgvector) + Qdrant + Redis + MongoDB
- `.env.example` - Complete environment variable template
- `deploy/install.sh` - Linux systemd service installer (user creation, dirs, service registration)
- `deploy/install.ps1` - Windows Service installer (sc.exe create)
- `deploy/start.sh` - Linux/macOS quick start script (build + load .env + run)
- `deploy/start.ps1` - Windows quick start script
- `deploy/build.sh` - Linux/macOS build script with feature flag args (--release, --all-backends, etc.)
- `deploy/build.ps1` - Windows build script with feature flag args
- `deploy/healthcheck.sh` - Health check script (curl /api/v1/health)
- `deploy/backup.sh` - Backup script (state snapshot + PostgreSQL dump + RocksDB data)
- `deploy/qbitos.service` - systemd unit file with security hardening (NoNewPrivileges, ProtectSystem, etc.)

### Planned

- Prometheus `/metrics` endpoint
- gRPC service layer
- Vision pipeline integration with the cognitive cycle
- Speech-to-speech end-to-end mode
- Distributed mode (multi-node kernel federation)

---

## [0.1.0] — 2026-07-19

The inaugural release of QbitOS — a Rust 2024 native multimodal
cognitive AI operating system.

### Added — Core (16 crates)

#### `qbitos-kernel`
- 8-stage cognitive loop (Observe → Understand → Create Goals → Plan → Execute → Verify → Learn → Improve)
- Priority-aware preemptive scheduler (Realtime / Interactive / Normal / Background)
- Global event bus with topic-prefix + wildcard subscriptions
- Per-agent resource budgets (CPU / memory / GPU / tokens / IO)
- Bootable runtime with graceful shutdown

#### `qbitos-memory`
- L0-L5 biological memory fabric (Sensory / Working / Episodic / Semantic / Procedural / Meta)
- HNSW vector index with filtered ANN, product quantization, BM25 hybrid search, MMR re-ranking, adaptive `ef`
- 5 pluggable backends: PostgreSQL + pgvector, Qdrant, Redis, RocksDB, Sled (default)
- Memory consolidation engine (L2 → L3 → L4 promotion based on importance/recency)

#### `qbitos-planner`
- Goal engine with hierarchical goal decomposition
- Task graph (DAG) with dependency resolution
- Reasoning engine integration
- Decision engine with multi-criteria selection

#### `qbitos-skills`
- Skill genome: registry, generator, optimizer, executor
- 5 step kinds: Read, Infer, Tool, Verify, Return
- Skill confidence with Beta(2,2) prior + sample-size scaling
- Optional real backends via `real-backends` feature:
  - `ToolModuleBackend` bridges to `qbitos_tools::ToolModule`
  - `NativeAgentBackend` bridges to `NativeAgent` with `ArcSwapOption` late binding

#### `qbitos-tools`
- 9 built-in tools: filesystem, terminal, http, calculator, summarizer, file_analyzer, sentiment, entity_extractor, time
- Tool VM with permission system (Read / Write / Execute / Network)
- Tool executor with memory recording

#### `qbitos-neural-engine`
- Native cognitive agent (no external LLM required)
- Chain-of-thought reasoning with self-consistency
- Self-reflection, constitutional AI, uncertainty quantification
- Optional candle-core backend for external models (Qwen2, etc.)
- Quantization, sampling (TopP, TopK, Temperature)

#### `qbitos-vision`
- HOG + color histogram + LBP feature extraction
- OCR (Otsu thresholding + connected components + template matching)
- Object detection (contour analysis + NMS)

#### `qbitos-speech`
- STT (FFT + MFCC + VAD)
- TTS (formant synthesis)

#### `qbitos-world-model`
- Knowledge graph (petgraph DiGraph)
- Entity engine with alias resolution
- Causal reasoning
- Simulation engine

#### `qbitos-evolution`
- Evaluator (multi-criteria skill assessment)
- Improvement engine (mutate skill parameters)
- Regression guard (reject changes that break tests)

#### `qbitos-data`
- 32+ dataset loaders: JSONL, CSV, Parquet, SQLite, PDF, HTML, image, audio, HuggingFace Hub, S3/MinIO, WebDataset, LMDB, HDF5, Excel, and more
- Streaming load with backpressure
- Format auto-detection
- Synthetic data generator

#### `qbitos-reasoning`
- Propositional logic (resolution theorem prover)
- CSP solver (AC-3 + forward checking)
- Bayesian networks
- MDP solver (value iteration, policy iteration)
- Game-tree search (minimax + alpha-beta)

#### `qbitos-learning`
- Multi-armed bandits (UCB1, ε-greedy)
- Preference learning (Bradley-Terry)
- Feedback collector
- Q-learning

#### `qbitos-safety`
- Content filter
- Prompt-injection detector
- Output validator
- Rate limiter (token bucket)
- Audit logger

#### `qbitos-training`
- Curriculum learning (5 stages: Trivial → Easy → Medium → Hard → Expert, with per-stage promotion thresholds)
- Experience replay (uniform + priority-weighted sampling with alpha exponent)
- Self-play engine (arbitrary games via 4 closures)
- Self-improvement cycle (detect weakness → propose fix → test → accept/reject)
- Meta-learning (UCB1 + epsilon-greedy strategy selection with Wilson 95% confidence intervals)
- Training loop orchestrator

### Added — Service Layer (1 crate)

#### `qbitos-services`
- HTTP/REST API (Axum 0.8): 19 endpoints
- WebSocket real-time event streaming (`/api/v1/events/ws`)
- JSON-RPC 2.0 server with 7 default methods + batch support
- OpenAPI 3.1 spec generation at `/api/v1/openapi.json`
- Event bus (broadcast channel, 13 event variants, lag-tolerant retry)
- **Persistent event store**: backs the in-memory buffer with the memory fabric's semantic layer; events survive across restarts
- **Bearer-token authentication** (`AuthService`): SHA-256 hashed, constant-time compared via `subtle::ConstantTimeEq`; public path / public prefix / public read bypass modes
- **Real cognitive kernel integration**: `--real-kernel` flag dispatches `/api/v1/cognitive/cycle` through the real 8-stage loop
- CORS policies: permissive (dev) or restricted (production)
- `CognitiveService` orchestrator wires MemoryFabric + SkillModule + Curriculum + ExperienceReplay + MetaLearner + SelfImprovementCycle + optional PersistentEventStore + optional CognitiveKernel

### Added — CLI (1 crate)

#### `qbitos-cli`
- `qbitos` binary with 15 subcommands: `repl`, `status`, `ask`, `dataset`, `memory`, `skill`, `tool`, `world`, `evolution`, `state`, `backend`, `reason`, `safety`, `learn`, `serve`
- `App` owns `Arc` handles to all 16 subsystems
- `KernelStageHandler` drives every cognitive stage with the real subsystems
- `DatasetToMemory` pipeline: load → embed → write to memory fabric
- `DatasetToSkills` pipeline: mine agent trajectories / instructions → register skills
- Interactive REPL with `:status`, `:memory`, `:skills`, `:tools`, `:world`, `:evolution`, `:quit`
- `qbitos serve` subcommand: starts the full HTTP + WebSocket service layer with optional auth, real kernel, persistent events, restricted CORS

### Added — Desktop (1 crate)

#### `qbitos-desktop`
- Tauri v2 backend (Rust) + React 19 + TypeScript + Tailwind CSS 4 frontend
- 9 views: Chat, Voice, Vision, Memory, Skills, World Model, Agent Monitor, Datasets, Service
- Service view: live WebSocket event stream, health monitoring, cognitive cycle tester, manual event publisher, connection panel with optional Bearer auth
- Live kernel status sidebar (CPU, RAM, current stage, cycle count)
- Dual-mode: `cli` (default, no GTK deps) and `desktop` (full Tauri, requires system libs)

### Added — Examples (1 crate)

#### `qbitos-examples`
- `cognitive_cycle`: HTTP REST demo against live server — 7 endpoints
- `ws_events`: WebSocket event stream consumer
- `rpc_client`: in-process + HTTP JSON-RPC 2.0 client with batch handling
- `self_improvement_loop`: full training loop in-process (200 steps, curriculum advancement, 4-strategy meta-learning)

### Documentation

- Top-level `README.md` with architecture overview, quick start, and test summary
- `docs/ARCHITECTURE.md` — system architecture with subsystem topology diagrams
- `docs/API.md` — HTTP/REST + WebSocket + JSON-RPC reference
- `docs/DEPLOYMENT.md` — production deployment guide (Docker, systemd, Nginx)
- `docs/CONTRIBUTING.md` — development workflow and submission process
- `docs/CHANGELOG.md` — this file
- `data/README.md` — dataset loader documentation
- Per-crate `README.md` files (planned for 0.2.0)

### Test Suite

- **196 tests passing** across all subsystems (186 default + 10 feature-gated)
- Skills: 24 default / 33 with `real-backends`
- Training: 29
- Services: 39 (including 11 auth + 5 event_store)
- CLI integration: 10 default / 13 with `real-backends`
- Plus 84 tests across the other 12 crates

### Build System

- Workspace with 20 crates, resolver = "3"
- Rust 2024 edition, rust-version = 1.85
- `release` profile with `lto = "fat"`, `panic = "abort"`
- Feature flags for DB backends (postgres-backed default / qdrant-backed / redis-backed / mongodb-backed / duckdb-backed / rocksdb-backed) and `real-backends`
- ~37 000 LOC, 190 Rust files

### Known Limitations

- No multi-tenant authentication (single admin principal via `--auth-token`)
- No Prometheus `/metrics` endpoint (planned for 0.2.0)
- No gRPC service layer (planned for 0.2.0)
- Vision pipeline not yet integrated with the cognitive cycle
- No distributed mode (single-node only)
- Desktop app requires system GTK/webkit2gtk deps

### Security

- Bearer-token auth with constant-time comparison
- Public path bypass for health/version/openapi/WebSocket
- Rate limiting available via `qbitos_safety::RateLimiter` (not yet wired into HTTP middleware)
- Content filter, prompt-injection detector, output validator available
- Audit logger available

---

## Version Numbering

QbitOS follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: incompatible API changes (e.g., removing an endpoint, changing a response shape)
- **MINOR**: backwards-compatible new functionality (e.g., new endpoint, new feature flag)
- **PATCH**: backwards-compatible bug fixes

Until 1.0.0, MINOR version bumps may include breaking changes (per SemVer 0.y.z rules).

---

## Links

- [Repository](https://github.com/qbitos/qbitos)
- [Issues](https://github.com/qbitos/qbitos/issues)
- [Releases](https://github.com/qbitos/qbitos/releases)
- [Documentation](https://github.com/qbitos/qbitos/tree/main/docs)
