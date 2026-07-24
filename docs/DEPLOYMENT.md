# QbitOS Deployment Guide

This guide covers production deployment of the QbitOS service layer.

## Table of Contents

- [Quick Start](#quick-start)
- [Database Setup (Automated)](#database-setup-automated)
- [Build Configurations](#build-configurations)
- [Running the Server](#running-the-server)
- [Authentication](#authentication)
- [Persistent Events](#persistent-events)
- [Real Cognitive Kernel](#real-cognitive-kernel)
- [Memory Backends](#memory-backends)
- [Process Supervision](#process-supervision)
- [Docker](#docker)
- [Systemd](#systemd)
- [Nginx Reverse Proxy](#nginx-reverse-proxy)
- [Monitoring](#monitoring)
- [Backup and Restore](#backup-and-restore)
- [Troubleshooting](#troubleshooting)


## Database Setup (Automated)

QbitOS ships with a powerful database setup script that handles everything:
user creation, password, database creation, pgvector extension, schema
migration, seed data, and `.env` file generation — all in one command.

### Linux / macOS

```bash
# Interactive (prompts for password)
./scripts/setup-db.sh

# Non-interactive with explicit password
./scripts/setup-db.sh --password "mySecurePassword123"

# Reset everything and start fresh
./scripts/setup-db.sh --drop-existing --password "mySecret"

# Use Docker (auto-starts pgvector/pgvector:pg16 container)
./scripts/setup-db.sh --docker --password "mySecret"

# Skip seed data (empty database)
./scripts/setup-db.sh --password "mySecret" --skip-seed
```

### Windows (PowerShell)

```powershell
# Interactive
.\scripts\setup-db.ps1

# Non-interactive
.\scripts\setup-db.ps1 -Password "mySecret"

# Docker mode (auto-starts container with pgvector)
.\scripts\setup-db.ps1 -Docker -Password "mySecret"

# Reset
.\scripts\setup-db.ps1 -DropExisting -Password "mySecret"
```

### What the setup script does

1. Checks PostgreSQL is installed and running
2. Creates the `qbitos` role with your password
3. Creates the `qbitos` database owned by the `qbitos` role
4. Installs pgvector + 7 other extensions (pg_trgm, btree_gin, etc.)
5. Runs `001_full_schema.sql` (14 tables, 82+ indexes)
6. Runs `002_production_indexes.sql` (production tuning indexes)
7. Runs `003_seed_data.sql` (50+ memories, 15 skills, 100+ knowledge triples, training data, safety rules)
8. Generates `.env` file with the correct `DATABASE_URL`
9. Validates the setup with test queries

### Docker Compose (one-command full stack)

```bash
docker compose up -d
```

This boots the complete stack with auto-seeded PostgreSQL:

| Service | Port | Purpose |
|---------|------|---------|
| qbitos | 8080 | Cognitive AI service (HTTP/WS/RPC) |
| postgres | 5432 | PostgreSQL 16 + pgvector (auto-seeded) |
| qdrant | 6333 | Vector database for high-performance ANN |
| redis | 6379 | Hot cache layer |
| mongodb | 27017 | Document database (optional backend) |
| pgadmin | 5050 | PostgreSQL web admin (debug profile) |
| prometheus | 9090 | Metrics scraper (monitoring profile) |
| grafana | 3000 | Dashboards (monitoring profile) |

### Seed Data Contents

The `003_seed_data.sql` migration gives the cognitive agent a strong kick-start:

- **50+ memories** across all 6 layers (L0 Sensory → L5 Meta)
- **15 skills** (qa, web-search, code-analyzer, summarize, calculate, translate, file-analyzer, sentiment-analysis, entity-extraction, http-request, reason, recall, time-query, hash-compute, json-query)
- **100+ knowledge graph triples** (QbitOS architecture, Rust, ML concepts, components)
- **5 curriculum stages** with success rates (trivial → expert)
- **4 meta-learning strategies** with reward history (epsilon_greedy, ucb1, softmax, thompson_sampling)
- **17 safety guardrail rules** (banned patterns, PII detection, allowed URLs)
- **21 cognitive events** (sample 8-stage cycles, skill executions, training steps)
- **14 tool statistics** (all built-in tools with call/success/error counts)
- **15 audit log entries** (auth, tool calls, safety checks, cognitive cycles)

---


## Production Deployment Scripts

The workspace includes production-ready deployment scripts for both Linux and Windows:

| File | Platform | Purpose |
|------|----------|---------|
| `Dockerfile` | Docker | Multi-stage build: Rust 1.85 builder + Debian slim runtime |
| `docker-compose.yml` | Docker | Full stack: QbitOS + PostgreSQL + Qdrant + Redis + MongoDB |
| `.env.example` | All | Environment variable template (copy to `.env`) |
| `deploy/install.sh` | Linux | systemd service installation (user creation, dirs, service) |
| `deploy/install.ps1` | Windows | Windows Service installation (sc.exe create) |
| `deploy/start.sh` | Linux/macOS | Quick start (build if needed, load .env, run) |
| `deploy/start.ps1` | Windows | Quick start (build if needed, load .env, run) |
| `deploy/build.sh` | Linux/macOS | Build script with feature flag args |
| `deploy/build.ps1` | Windows | Build script with feature flag args |
| `deploy/healthcheck.sh` | Linux | Health check script (curl /api/v1/health) |
| `deploy/backup.sh` | Linux | Backup script (state snapshot + PostgreSQL + RocksDB) |
| `deploy/qbitos.service` | Linux | systemd unit file with security hardening |

### Quick Start (Linux)

```bash
# 1. Build
cargo build --release -p qbitos-cli

# 2. Install as systemd service
sudo ./deploy/install.sh

# 3. Configure
sudo nano /etc/qbitos/qbitos.env

# 4. Start
sudo systemctl start qbitos

# 5. Check
sudo systemctl status qbitos
```

### Quick Start (Windows)

```powershell
# 1. Build
cargo build --release -p qbitos-cli

# 2. Install as Windows Service (run as Admin)
.\deploy\install.ps1

# 3. Configure
notepad C:\ProgramData\QbitOS\qbitos.env

# 4. Start
Start-Service QbitOS

# 5. Check
Get-Service QbitOS
```

### Quick Start (Docker)

```bash
# Full stack with one command
docker compose up -d

# Check health
./deploy/healthcheck.sh

# View logs
docker compose logs -f qbitos

# Stop
docker compose down
```

### Build Scripts

```bash
# Linux/macOS - build with specific backends
./deploy/build.sh --release --all-backends --real-backends

# Windows
.\deploy\build.ps1 -Release -AllBackends -RealBackends
```

---

## Quick Start

```bash
# 1. Build the CLI with all hardening features
cargo build --release -p qbitos-cli --features real-backends,postgres-backed

# 2. Set environment
export QBITOS_TOKEN="$(openssl rand -hex 32)"
export QBITOS_DATA_DIR=/var/lib/qbitos

# 3. Run
./target/release/qbitos serve \
  --addr 0.0.0.0:8080 \
  --name qbitos-prod \
  --auth-token "$QBITOS_TOKEN" \
  --real-kernel \
  --persistent-events \
  --restricted-cors \
  --allow-self-improve
```

---

## Build Configurations

### Production (recommended)

```bash
cargo build --release -p qbitos-cli --features real-backends,postgres-backed
```

- `--release`: optimized binary, stripped debug info
- `real-backends`: wires NativeAgent + ToolModule into SkillModule
- `postgres-backed`: PostgreSQL primary backend (pgvector, HNSW, FTS)

Binary size: ~80 MB (release-stripped).
Build time: ~10 min (candle-core is heavy).

### Development

```bash
cargo build -p qbitos-cli
```

Fast build (~30s), noop backends, smaller binary (~50 MB). Use for
REPL, tests, and rapid iteration.

### Full (all DB backends)

```bash
cargo build --release -p qbitos-cli --features real-backends,all-backends
```

Enables Sled + Postgres + Qdrant + Redis + RocksDB. Use when you need
multiple backend types in the same binary (e.g., for migration).

### Cross-compilation

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl \
  -p qbitos-cli --features real-backends,postgres-backed
```

Produces a fully-static binary suitable for scratch containers.

---

## Running the Server

### CLI flags

```
qbitos serve [OPTIONS]

Options:
      --addr <ADDR>              Bind address [default: 127.0.0.1:8080]
      --name <NAME>             Instance name [default: qbitos-cli]
      --no-safety               Disable safety enforcement
      --allow-self-improve      Enable self-improvement (writes to skill registry)
      --auth-token <TOK>        Enable Bearer-token auth (registers "admin" principal)
      --real-kernel             Attach the real CognitiveKernel
      --persistent-events       Attach persistent event store (survives restarts)
      --restricted-cors         Use restricted CORS policy
  -h, --help                    Print help
```

### Example: minimal local dev

```bash
./target/debug/qbitos serve --addr 127.0.0.1:8080
```

- No auth
- No persistent events
- Stub cognitive cycle
- Permissive CORS

### Example: hardened production

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

---

## Authentication

### Token generation

Generate a strong token:

```bash
export QBITOS_TOKEN="$(openssl rand -hex 32)"
echo "Token: $QBITOS_TOKEN"
```

### Single-principal mode (default)

`--auth-token <TOKEN>` registers a single principal named `admin` with
the `admin` role. All non-public routes require this token.

### Multi-principal mode

For multi-tenant deployments, embed the `AuthService` in your own
binary and register multiple principals:

```rust
use qbitos_services::{AuthService, AuthConfig};

let auth = AuthService::new(AuthConfig {
    enabled: true,
    allow_public_read: false,
    public_paths: vec!["/api/v1/health".into()],
    public_prefixes: vec![],
});
auth.register_token_with_role("admin", admin_token, "admin");
auth.register_token_with_role("agent-1", agent1_token, "write");
auth.register_token_with_role("watcher", watcher_token, "read");
```

### Public routes

Always bypass auth (regardless of config):

- `GET /api/v1/health`
- `GET /api/v1/version`
- `GET /api/v1/openapi.json`
- `GET /api/v1/openapi.yaml`
- `GET /api/v1/events/ws` (WebSocket upgrade)

---

## Persistent Events

Enable with `--persistent-events`. Events are stored as semantic-memory
concepts labeled `qbitos:event:<id>` in the memory fabric's L3 layer.

On server restart, call `POST /api/v1/events/replay` to restore the
event history:

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
     http://localhost:8080/api/v1/events/replay
# → {"ok": true, "data": {"replayed": 42, "buffer_len": 50}}
```

Or auto-replay on startup by adding to your startup script:

```bash
./target/release/qbitos serve --persistent-events --auth-token "$TOKEN" &
sleep 2
curl -X POST -H "Authorization: Bearer $TOKEN" \
     http://localhost:8080/api/v1/events/replay
```

---

## Real Cognitive Kernel

Enable with `--real-kernel`. The `/api/v1/cognitive/cycle` endpoint
dispatches through the real 8-stage `CognitiveKernel` instead of the
stub. The kernel uses the App's:

- `MemoryFabric` (recall + store)
- `SkillModule` (skill execution)
- `ToolModule` (tool dispatch)
- `WorldModel` (knowledge graph)
- `Planner` (goal derivation)
- `Evolution` (improvement loop)
- `NeuralEngine` (NativeAgent inference)
- `Vision` / `Speech` (multimodal I/O)
- `Data` (dataset registry)

Confidence is `0.9` when `cycle.verified == Some(true)`, `0.7` otherwise.

---

## Memory Backends

### Sled (default, embedded)

Pure Rust embedded key-value store. No external dependencies. Data
stored at `$QBITOS_ROCKSDB_PATH` (default: `/var/lib/qbitos/rocksdb`).

```bash
mkdir -p /var/lib/qbitos
export QBITOS_DATA_DIR=/var/lib/qbitos
```

### PostgreSQL + pgvector

```bash
cargo build --release -p qbitos-cli --features postgres-backed
export DATABASE_URL=postgres://user:pass@localhost:5432/qbitos
./target/release/qbitos serve --addr 0.0.0.0:8080
```

Requires `pgvector` extension: `CREATE EXTENSION vector;`

### Qdrant

```bash
cargo build --release -p qbitos-cli --features qdrant-backed
export QDRANT_URL=http://localhost:6333
./target/release/qbitos serve --addr 0.0.0.0:8080
```

### Redis (cache layer)

```bash
cargo build --release -p qbitos-cli --features redis-backed
export REDIS_URL=redis://localhost:6379
./target/release/qbitos serve --addr 0.0.0.0:8080
```

### Multiple backends

Backends can be stacked — the memory fabric queries all attached
backends and merges results:

```bash
cargo build --release -p qbitos-cli --features all-backends
export DATABASE_URL=postgres://...
export QDRANT_URL=http://...
export REDIS_URL=redis://...
./target/release/qbitos serve
```

The CLI auto-attaches all configured backends at startup
(`App::auto_attach_backends`).

---

## Process Supervision

### Systemd

`/etc/systemd/system/qbitos.service`:

```ini
[Unit]
Description=QbitOS Cognitive Service
After=network.target

[Service]
Type=simple
User=qbitos
Group=qbitos
WorkingDirectory=/var/lib/qbitos
Environment=QBITOS_DATA_DIR=/var/lib/qbitos
Environment=QBITOS_TOKEN_FILE=/etc/qbitos/token
EnvironmentFile=-/etc/qbitos/env
ExecStart=/usr/local/bin/qbitos serve \
  --addr 0.0.0.0:8080 \
  --name qbitos-prod \
  --auth-token $(cat $QBITOS_TOKEN_FILE) \
  --real-kernel \
  --persistent-events \
  --restricted-cors
Restart=on-failure
RestartSec=5
LimitNOFILE=65535
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable qbitos
sudo systemctl start qbitos
sudo journalctl -u qbitos -f
```

---

## Docker

### Dockerfile

```dockerfile
FROM rust:1.85-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p qbitos-cli --features real-backends,postgres-backed

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/qbitos /usr/local/bin/qbitos
VOLUME /var/lib/qbitos
EXPOSE 8080
ENTRYPOINT ["qbitos"]
CMD ["serve", "--addr", "0.0.0.0:8080", "--real-kernel", "--persistent-events"]
```

### Build and run

```bash
docker build -t qbitos:latest .
docker run -d \
  --name qbitos \
  -p 8080:8080 \
  -v qbitos-data:/var/lib/qbitos \
  -e QBITOS_TOKEN="$(openssl rand -hex 32)" \
  qbitos:latest \
  serve --addr 0.0.0.0:8080 --auth-token "$QBITOS_TOKEN" --real-kernel --persistent-events
```

### Docker Compose

```yaml
version: "3.9"
services:
  qbitos:
    image: qbitos:latest
    ports:
      - "8080:8080"
    volumes:
      - qbitos-data:/var/lib/qbitos
    environment:
      QBITOS_TOKEN: ${QBITOS_TOKEN}
    command: >
      serve --addr 0.0.0.0:8080
            --auth-token ${QBITOS_TOKEN}
            --real-kernel
            --persistent-events
            --restricted-cors
    restart: unless-stopped

volumes:
  qbitos-data:
```

---

## Nginx Reverse Proxy

For TLS termination and rate limiting:

```nginx
upstream qbitos {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name qbitos.example.com;

    ssl_certificate     /etc/ssl/certs/qbitos.pem;
    ssl_certificate_key /etc/ssl/private/qbitos.key;

    # Rate limit
    limit_req_zone $binary_remote_addr zone=qbitos:10m rate=10r/s;

    location /api/ {
        limit_req zone=qbitos burst=20 nodelay;
        proxy_pass http://qbitos;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /api/v1/events/ws {
        proxy_pass http://qbitos;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

---

## Monitoring

### Health check

```bash
curl http://localhost:8080/api/v1/health | jq .data.healthy
# → true
```

### Stats endpoint

```bash
watch -n 5 'curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8080/api/v1/stats | jq'
```

### WebSocket event stream

Connect to `ws://localhost:8080/api/v1/events/ws` and watch for
`subsystem_started`, `subsystem_stopped`, `safety_event`, and
`log` events.

### Logging

QbitOS uses `tracing` with `tracing-subscriber`. Set the log level via
`RUST_LOG`:

```bash
RUST_LOG=info,qbitos=debug,hyper=warn ./target/release/qbitos serve
```

For JSON logs (production):

```bash
RUST_LOG=info QBITOS_LOG_FORMAT=json ./target/release/qbitos serve
```

### Prometheus

QbitOS exposes a `/metrics` endpoint in the Prometheus text exposition format.
Scrape it with Prometheus, Grafana Agent, VictoriaMetrics, or any compatible
collector:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'qbitos'
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:8080']
```

Exposed metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `qbitos_requests_total` | counter | Total HTTP requests received |
| `qbitos_invocations_total` | counter | Total subsystem invocations |
| `qbitos_invocations_failed_total` | counter | Total failed subsystem invocations |
| `qbitos_memory_episodes` | gauge | Episodic memory entries |
| `qbitos_memory_working` | gauge | Working memory entries |
| `qbitos_memory_sensory` | gauge | Sensory memory slots |
| `qbitos_skills_total` | gauge | Registered skills |
| `qbitos_replay_buffer_len` | gauge | Experience replay buffer length |
| `qbitos_curriculum_stage` | gauge | Current curriculum stage index |
| `qbitos_ws_sessions_active` | gauge | Active WebSocket sessions |
| `qbitos_healthy` | gauge | 1 if healthy, 0 if shutting down |

---

## Backup and Restore

### State snapshot

The CLI can serialize all state (skills, episodes, knowledge graph,
datasets) to a JSON file:

```bash
# Backup
./target/release/qbitos state save /backup/qbitos-$(date +%Y%m%d).json

# Restore
./target/release/qbitos state load /backup/qbitos-20260719.json
```

### Sled data backup

If using Sled, simply copy the data directory:

```bash
tar czf /backup/qbitos-rocksdb-$(date +%Y%m%d).tar.gz /var/lib/qbitos/rocksdb
```

### PostgreSQL backup

```bash
pg_dump qbitos > /backup/qbitos-$(date +%Y%m%d).sql
```

---

## Troubleshooting

### "Address already in use"

```bash
sudo lsof -i :8080
# or
sudo ss -tnlp | grep 8080
```

Kill the conflicting process or use a different port: `--addr 127.0.0.1:8081`.

### "wired real NativeAgent into SkillModule InferBackend" not in logs

You compiled without `--features real-backends`. Rebuild:

```bash
cargo build --release -p qbitos-cli --features real-backends
```

### 401 Unauthorized on every request

- You passed `--auth-token` but didn't send the `Authorization: Bearer <TOKEN>` header.
- Your token doesn't match.
- You're hitting a non-public route without auth.

### WebSocket connection fails with 101 expected

The WebSocket endpoint is public by default. If you've customized
`AuthConfig.public_paths` and removed `/api/v1/events/ws`, re-add it.

### High memory usage

- The replay buffer defaults to 10 000 entries. Lower it via `ServiceConfig.replay_capacity`.
- The persistent event store stores events as semantic-memory concepts. Periodically prune old concepts.
- The HNSW index grows with episodic memory. Configure `ef_construction` and `M` parameters.

### Build fails with "No space left on device"

candle-core + tokenizers + parquet are heavy. Either:
- Use `CARGO_INCREMENTAL=0` to reduce disk usage during builds.
- Build with `--no-default-features` to exclude heavy optional deps.
- Use the `cargo-nextest` binary cache.

---

## See Also

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — System architecture
- [`API.md`](./API.md) — HTTP/REST + WebSocket + JSON-RPC reference
- [`../README.md`](../README.md) — Top-level overview
