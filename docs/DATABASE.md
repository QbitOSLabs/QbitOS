# QbitOS Database Schema Documentation

This document describes the complete PostgreSQL database schema for QbitOS,
including all tables, indexes, extensions, functions, views, and seed data.

## Overview

QbitOS uses PostgreSQL 16+ with the [pgvector](https://github.com/pgvector/pgvector)
extension for vector similarity search. The schema consists of:

| Component | Count |
|-----------|------:|
| Tables | 14 |
| Indexes | 82+ |
| Extensions | 11 |
| Server-side functions | 6 |
| Monitoring views | 4 |
| Seed data records | 200+ |

## Quick Setup

```bash
# Linux/macOS — one command sets up everything
./scripts/setup-db.sh --password "yourPassword"

# Windows PowerShell
.\scripts\setup-db.ps1 -Password "yourPassword"

# Docker (auto-seeds on first boot)
docker compose up -d
```

See [Deployment Guide → Database Setup](./DEPLOYMENT.md#database-setup-automated) for details.

## Extensions

The schema requires and installs these PostgreSQL extensions:

| Extension | Purpose |
|-----------|---------|
| `vector` | pgvector — 384-dimensional vector columns for semantic similarity search |
| `pg_trgm` | Trigram similarity for fuzzy text matching |
| `btree_gin` | GIN index support for btree-summable types |
| `btree_gist` | GiST index support for btree-summable types |
| `pgcrypto` | Cryptographic functions (`gen_random_uuid()`) |
| `unaccent` | Accent-insensitive text search |
| `fuzzystrmatch` | Levenshtein, soundex, metaphone |
| `hstore` | Key-value storage |
| `ltree` | Hierarchical label tree |
| `bloom` | Bloom filter index |
| `pg_stat_statements` | Query statistics (monitoring) |

## Tables

### 1. `qbitos_memories`

Stores all memory objects across the 6-layer memory fabric (L0–L5).

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID PK | Unique memory ID (auto-generated) |
| `layer` | TEXT | Memory layer: `sensory`, `working`, `episodic`, `semantic`, `procedural`, `meta` |
| `content` | JSONB | Memory payload (type-specific) |
| `embedding` | vector(384) | Semantic embedding for similarity search |
| `importance` | FLOAT | Importance score [0, 1] — controls retention |
| `created_at` | TIMESTAMPTZ | Creation timestamp |
| `updated_at` | TIMESTAMPTZ | Last modification timestamp |
| `access_count` | INTEGER | Number of times this memory was accessed |
| `last_accessed_at` | TIMESTAMPTZ | Last access timestamp |
| `expires_at` | TIMESTAMPTZ | Expiration time (NULL = permanent) |

**Indexes:** 8 (including HNSW vector index on `embedding`, GIN on `content`, btree on `layer` + `importance`)

### 2. `qbitos_skills`

The skill genome — registered skills with execution statistics.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID PK | Unique skill ID |
| `name` | TEXT UNIQUE | Human-readable skill name |
| `description` | TEXT | What the skill does |
| `steps` | JSONB | Ordered array of `SkillStep` objects |
| `execution_count` | BIGINT | Total executions |
| `success_count` | BIGINT | Successful executions |
| `total_latency_ms` | BIGINT | Cumulative latency |
| `confidence` | FLOAT | Beta(2,2) confidence score [0, 1] |
| `created_at` | TIMESTAMPTZ | Creation time |
| `updated_at` | TIMESTAMPTZ | Last update time |

### 3. `qbitos_knowledge_triples`

World model knowledge graph — (subject, predicate, object) triples.

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | Triple ID |
| `subject` | TEXT | Subject entity |
| `predicate` | TEXT | Relationship type |
| `object` | TEXT | Object entity |
| `confidence` | FLOAT | Confidence in this fact [0, 1] |
| `source` | TEXT | Where this triple came from (`seed`, `inferred`, `user`) |
| `created_at` | TIMESTAMPTZ | Creation time |

**Indexes:** 6 (on subject, predicate, object, confidence, and composite)

### 4. `qbitos_events`

Append-only event log for cognitive cycles, skill executions, training, safety checks.

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | Event ID (monotonic sequence number) |
| `event_type` | TEXT | Event type (e.g. `cognitive_cycle_completed`, `memory_stored`) |
| `payload` | JSONB | Event-specific data |
| `created_at` | TIMESTAMPTZ | Event timestamp |

**Indexes:** 4 (on event_type, created_at, id for replay)

### 5. `qbitos_curriculum`

Training curriculum stages with success rates.

| Column | Type | Description |
|--------|------|-------------|
| `stage` | TEXT PK | Stage name (`trivial`, `easy`, `medium`, `hard`, `expert`) |
| `level` | INTEGER | Difficulty level (0–4) |
| `success_rate` | FLOAT | Current success rate at this stage |
| `samples_at_level` | BIGINT | Training samples at this level |
| `created_at` | TIMESTAMPTZ | |
| `updated_at` | TIMESTAMPTZ | |

### 6. `qbitos_meta_strategies`

Meta-learning strategy registry with reward tracking.

| Column | Type | Description |
|--------|------|-------------|
| `name` | TEXT PK | Strategy name (`epsilon_greedy`, `ucb1`, `softmax`, `thompson_sampling`) |
| `reward_mean` | FLOAT | Mean reward across all uses |
| `sample_count` | BIGINT | Number of times this strategy was selected |
| `last_used_at` | TIMESTAMPTZ | Last selection time |
| `created_at` | TIMESTAMPTZ | |

### 7. `qbitos_tool_stats`

Per-tool execution statistics for all 14 built-in tools.

| Column | Type | Description |
|--------|------|-------------|
| `tool_name` | TEXT PK | Tool name |
| `call_count` | BIGINT | Total calls |
| `success_count` | BIGINT | Successful calls |
| `error_count` | BIGINT | Failed calls |
| `total_duration_ms` | BIGINT | Cumulative duration |
| `last_called_at` | TIMESTAMPTZ | Last call time |

### 8. `qbitos_audit_log`

Append-only audit trail for security and compliance.

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | Audit entry ID |
| `event_type` | TEXT | `auth`, `tool_call`, `safety_check`, `cognitive_cycle`, etc. |
| `principal` | TEXT | Who initiated the action |
| `action` | TEXT | What was done |
| `resource` | TEXT | What was acted upon |
| `outcome` | TEXT | `success`, `failure`, `blocked` |
| `timestamp` | TIMESTAMPTZ | When it happened |
| `metadata` | JSONB | Additional context |

### 9–14. Additional Tables

- `qbitos_safety_rules` — Guardrail rules (banned patterns, PII, allowed URLs)
- `qbitos_experience_replay` — Training experience buffer
- `qbitos_world_entities` — World model entity registry
- `qbitos_causal_graph` — Causal relationship graph
- `qbitos_simulation_scenarios` — Simulation scenarios for world model
- `qbitos_federation_participants` — Federated learning participants

## Server-Side Functions

### `search_memories(query_text TEXT, query_embedding vector(384), match_count INT)`

Performs semantic similarity search across the memory fabric. Returns the
top-N memories ranked by cosine distance to the query embedding.

```sql
SELECT * FROM search_memories(
    'What is QbitOS?',
    (SELECT array_agg(random())::vector(384) FROM generate_series(1, 384)),
    5
);
```

### `recompute_skill_confidence(skill_uuid UUID)`

Recalculates a skill's confidence score using the Beta(2,2) prior with
sample-size scaling. Called automatically after each execution.

### `consolidate_memories(source_layer TEXT, target_layer TEXT)`

Moves memories from one layer to another during sleep consolidation.
Decays importance scores and evicts low-importance items.

### `prune_low_confidence_skills(min_samples INT, threshold FLOAT)`

Removes skills with success rate below `threshold` after at least
`min_samples` executions. Returns the list of pruned skill IDs.

### `record_cognitive_cycle(cycle_data JSONB)`

Atomically records a completed cognitive cycle and its 8-stage results
into the events table, updating relevant skill and tool statistics.

### `cleanup_expired_memories()`

Deletes all sensory and working memories that have passed their
`expires_at` timestamp. Designed to run as a periodic cron job.

## Monitoring Views

### `v_memory_stats`

Per-layer summary of memory counts, average importance, and total size.

```sql
SELECT * FROM v_memory_stats;
--  layer      | count | avg_importance | total_size
-- ------------+-------+----------------+------------
--  sensory    |    10 |         0.3000 |    4.2 KB
--  working    |     5 |         0.7300 |    3.1 KB
--  episodic   |     8 |         0.6375 |    8.7 KB
--  semantic   |    15 |         0.8833 |   12.4 KB
--  procedural |     5 |         0.8300 |    4.8 KB
--  meta       |     6 |         0.6967 |    5.2 KB
```

### `v_skill_performance`

Per-skill success rate, mean latency, and confidence ranking.

### `v_tool_health`

Per-tool error rate, mean latency, and last-called timestamp. Used by
the Grafana "Tool Dispatch Error Rate" panel.

### `v_audit_summary`

Hourly summary of audit events grouped by type and outcome. Used for
security monitoring dashboards.

## Seed Data

The `003_seed_data.sql` migration inserts comprehensive seed data to give
the cognitive agent a strong kick-start:

### Memories (50+)

| Layer | Count | Content |
|-------|------:|---------|
| L0 Sensory | 10 | Sample user inputs and raw perceptions |
| L1 Working | 5 | Active task contexts (QA, code analysis, planning) |
| L2 Episodic | 8 | Past experiences (user interactions, training cycles, safety events) |
| L3 Semantic | 15 | Facts about QbitOS, Rust, ML concepts, architecture |
| L4 Procedural | 5 | Skill references with success rates |
| L5 Meta | 6 | Learning strategies and insights |

### Skills (15)

| Skill | Steps | Success Rate | Description |
|-------|------:|:---:|-------------|
| qa | 3 | 92% | Answer a question using LLM inference |
| web-search | 4 | 83% | Search the web for current information |
| code-analyzer | 5 | 100% | Analyze source code for issues |
| summarize | 3 | 93% | Summarize a long text |
| calculate | 3 | 100% | Perform mathematical calculations |
| translate | 4 | 83% | Translate between languages |
| file-analyzer | 4 | 100% | Analyze a file's contents |
| sentiment-analysis | 3 | 100% | Analyze text sentiment |
| entity-extraction | 3 | 100% | Extract named entities |
| http-request | 4 | 89% | Make HTTP requests |
| reason | 5 | 80% | Multi-step graph-of-thoughts reasoning |
| recall | 4 | 94% | Search memory for information |
| time-query | 3 | 100% | Get current time/date |
| hash-compute | 3 | 100% | Compute hash of data |
| json-query | 3 | 100% | Query JSON with path expressions |

### Knowledge Graph (100+ triples)

Covers QbitOS architecture, cognitive kernel stages, memory layers,
neural engine features, vision/speech capabilities, world model,
planner, skills, tools, safety, training modules, evolution, services,
Rust programming language, and ML concepts.

### Training Data

- **5 curriculum stages** with success rates and sample counts
- **4 meta-learning strategies** with reward history and last-used timestamps
- **21 cognitive events** showing sample 8-stage cycles in action

### Safety Rules (17)

- 4 banned input patterns (prompt injection attempts)
- 4 banned output patterns (dangerous commands)
- 4 PII detection patterns (email, phone, SSN, IP)
- 5 allowed URL domains (wikipedia, github, stackoverflow, docs.rs, crates.io)

## Backup and Restore

```bash
# Full backup (all databases + data dir)
./deploy/backup-full.sh --output-dir /backups --retention-days 7

# Restore from backup
./deploy/restore-full.sh --backup-file /backups/qbitos-backup-20260115.tar.gz
```

See [Deployment Guide → Backup and Restore](./DEPLOYMENT.md#backup-and-restore) for details.

## Migration Files

| File | Description |
|------|-------------|
| `001_full_schema.sql` | Full schema: 14 tables, extensions, functions, views |
| `002_production_indexes.sql` | Production-tuning indexes (HNSW, composite, partial) |
| `003_seed_data.sql` | Comprehensive seed data (50+ memories, 15 skills, 100+ triples) |

All migrations are idempotent (safe to run multiple times) and are
auto-applied by the setup scripts and Docker Compose on first boot.
