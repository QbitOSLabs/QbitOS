# QbitOS Documentation

> **A Rust 2024 native multimodal cognitive AI operating system.**
> *A small 3B AI model becomes powerful through a superior cognitive operating system.*

Welcome to the QbitOS documentation. This directory contains the
canonical reference for system architecture, APIs, deployment, and
contribution processes.

## Documentation Index

| Document                  | Audience                  | Description                                                       |
|---------------------------|---------------------------|-------------------------------------------------------------------|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Architects, contributors | System architecture, subsystem topology, data flow diagrams       |
| [`API.md`](./API.md)                 | API consumers            | HTTP/REST + WebSocket + JSON-RPC 2.0 reference with examples      |
| [`DEPLOYMENT.md`](./DEPLOYMENT.md)   | Operators, SREs          | Production deployment guide: Docker, systemd, Nginx, monitoring   |
| [`DATABASE.md`](./DATABASE.md)       | Database admins, devs    | Full schema reference: 14 tables, 82+ indexes, seed data, functions |
| [`CONTRIBUTING.md`](./CONTRIBUTING.md) | Contributors            | Development workflow, code style, submission process              |
| [`CHANGELOG.md`](./CHANGELOG.md)     | Everyone                 | Release history and notable changes                               |

## Project-Level Documents

| Document                          | Description                                  |
|-----------------------------------|----------------------------------------------|
| [`../README.md`](../README.md)    | Top-level overview, quick start, test summary|
| [`../worklog.md`](../worklog.md)  | Implementation history (per-task log)        |
| [`../LICENSE`](../LICENSE)                | Apache 2.0 license                         |

## Quick Links

- **New to QbitOS?** Read [`../README.md`](../README.md) first.
- **Want to deploy?** Read [`DEPLOYMENT.md`](./DEPLOYMENT.md).
- **Database setup?** Read [`DATABASE.md`](./DATABASE.md) or run `./scripts/setup-db.sh`.
- **Building a client?** Read [`API.md`](./API.md).
- **Want to contribute?** Read [`CONTRIBUTING.md`](./CONTRIBUTING.md).
- **Looking for an overview?** Read [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Per-Crate READMEs

Each crate has its own README with usage examples and API details:

| Crate                  | Purpose                                                |
|------------------------|--------------------------------------------------------|
| [`kernel`](../kernel/README.md)             | Cognitive kernel, scheduler, event bus                |
| [`memory`](../memory/README.md)             | L0-L5 biological memory fabric                         |
| [`planner`](../planner/README.md)           | Goal engine, task graph, decision engine               |
| [`skills`](../skills/README.md)             | Skill genome: registry, generator, optimizer, executor|
| [`tools`](../tools/README.md)               | Tool VM with sandbox and permissions                   |
| [`neural_engine`](../neural_engine/README.md) | Native cognitive agent + candle-based inference     |
| [`vision`](../vision/README.md)             | Image processing, OCR, object detection                |
| [`speech`](../speech/README.md)             | STT and TTS pipelines                                  |
| [`world_model`](../world_model/README.md)   | Knowledge graph, causal reasoning, simulation          |
| [`evolution`](../evolution/README.md)       | Evaluator, improvement engine, regression guard        |
| [`data`](../data/README.md)                 | 32+ dataset loaders, streaming, S3/HF Hub              |
| [`reasoning`](../reasoning/README.md)       | Propositional logic, CSP, Bayesian, MDP, game tree     |
| [`learning`](../learning/README.md)         | Bandits, preference learning, Q-learning               |
| [`safety`](../safety/README.md)             | Content filter, injection detector, audit log          |
| [`training`](../training/README.md)         | Curriculum, replay, self-play, self-improvement        |
| [`services`](../services/README.md)         | HTTP/REST, WebSocket, JSON-RPC, OpenAPI                |
| [`cli`](../cli/README.md)                   | `qbitos` binary orchestrator                           |
| [`desktop`](../desktop/README.md)           | Tauri v2 desktop app                                   |
| [`examples_pkg`](../examples_pkg/README.md) | End-to-end demos                                       |

## Project Statistics

- **21 crates** in the workspace
- **280+ Rust source files**
- **~65,000 lines of Rust code**
- **407+ tests passing** (offline-verified)
- **14 database tables**, 82+ indexes, 200+ seed records
- **25 HTTP routes**, 15 Tauri commands, 14 built-in tools
- **Rust 2024 edition** (rust-version 1.85)
- **Licensed** under Apache-2.0

## Getting Help

- 📖 Read the [architecture overview](./ARCHITECTURE.md)
- 🔍 Search [existing issues](https://github.com/qbitos/qbitos/issues)
- 💬 Open a [discussion](https://github.com/qbitos/qbitos/discussions)
- 🐛 File a [bug report](https://github.com/qbitos/qbitos/issues/new?labels=bug)
- ✨ Request a [feature](https://github.com/qbitos/qbitos/issues/new?labels=enhancement)
