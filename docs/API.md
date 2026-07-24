# QbitOS HTTP / WebSocket / JSON-RPC API Reference

> API version: **0.1.0** · OpenAPI 3.1 spec: [`/api/v1/openapi.json`](http://localhost:8080/api/v1/openapi.json)

This document is the canonical reference for the QbitOS service layer.
All endpoints accept and return JSON. All responses are wrapped in the
standard [`ApiResponse`](#apiresponse-envelope) envelope.

## Table of Contents

- [Authentication](#authentication)
- [Standard Envelopes](#standard-envelopes)
- [System Endpoints](#system-endpoints)
  - [GET /api/v1/health](#get-apiv1health)
  - [GET /api/v1/version](#get-apiv1version)
  - [GET /api/v1/stats](#get-apiv1stats)
- [Cognitive Endpoints](#cognitive-endpoints)
  - [POST /api/v1/cognitive/cycle](#post-apiv1cognitivecycle)
- [Memory Endpoints](#memory-endpoints)
  - [POST /api/v1/memory](#post-apiv1memory)
  - [POST /api/v1/memory/search](#post-apiv1memorysearch)
  - [DELETE /api/v1/memory/{id}](#delete-apiv1memoryid)
- [Skill Endpoints](#skill-endpoints)
  - [GET /api/v1/skills](#get-apiv1skills)
  - [POST /api/v1/skills/execute](#post-apiv1skillsexecute)
- [Training Endpoints](#training-endpoints)
  - [POST /api/v1/training/step](#post-apiv1trainingstep)
  - [GET /api/v1/curriculum](#get-apiv1curriculum)
  - [GET /api/v1/replay/stats](#get-apiv1replaystats)
  - [GET /api/v1/meta/strategies](#get-apiv1metastrategies)
- [Event Endpoints](#event-endpoints)
  - [POST /api/v1/events/subscribe](#post-apiv1eventssubscribe)
  - [POST /api/v1/events/poll](#post-apiv1eventspoll)
  - [POST /api/v1/events/replay](#post-apiv1eventsreplay)
  - [GET /api/v1/events/lookup/{id}](#get-apiv1eventslookupid)
  - [GET /api/v1/events/ws](#get-apiv1eventsws-websocket)
- [JSON-RPC 2.0](#json-rpc-20)
- [OpenAPI Spec](#openapi-spec)
- [Error Codes](#error-codes)

---

## Authentication

When the server is started with `--auth-token <TOKEN>`, all non-public
routes require an `Authorization: Bearer <TOKEN>` header.

### Public routes (no auth required)

- `GET /api/v1/health`
- `GET /api/v1/version`
- `GET /api/v1/openapi.json`
- `GET /api/v1/openapi.yaml`
- `GET /api/v1/events/ws` (WebSocket upgrade)

### Example: authenticated request

```bash
curl -H "Authorization: Bearer $QBITOS_TOKEN" \
     http://localhost:8080/api/v1/stats
```

### Auth failure responses

| Status | Body                                                                  |
|--------|-----------------------------------------------------------------------|
| 401    | `{"error":"missing authorization header"}` (with `WWW-Authenticate: Bearer`) |
| 401    | `{"error":"invalid token"}`                                           |
| 401    | `{"error":"invalid authorization header: <reason>"}`                  |

Tokens are SHA-256 hashed at registration time and compared in constant
time using `subtle::ConstantTimeEq`.

---

## Standard Envelopes

### `ApiResponse` envelope

Every response is wrapped in this envelope:

```json
{
  "ok": true,
  "data": { /* payload */ },
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-07-19T12:34:56.789Z"
}
```

On error:

```json
{
  "ok": false,
  "error": "not found: endpoint not found",
  "error_code": -32001,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-07-19T12:34:56.789Z"
}
```

---

## System Endpoints

### `GET /api/v1/health`

Returns server health, instance metadata, and current statistics. **Public**.

**Response (`HealthResponse`):**

```json
{
  "ok": true,
  "data": {
    "instance_id": "550e8400-e29b-41d4-a716-446655440000",
    "instance_name": "qbitos-prod",
    "healthy": true,
    "api_version": "0.1.0",
    "uptime_seconds": 3600,
    "subsystems": ["memory", "kernel", "skills", "tools", "..."],
    "stats": {
      "total_requests": 12345,
      "total_events": 9876,
      "active_ws_sessions": 3,
      "uptime_seconds": 3600,
      "subsystem_invocations": { "kernel": 100, "skills": 50 },
      "subsystem_errors": {}
    }
  },
  "request_id": "...",
  "timestamp": "..."
}
```

### `GET /api/v1/version`

Returns API and Rust edition versions. **Public**.

```json
{
  "ok": true,
  "data": { "api": "0.1.0", "rust": "2024", "edition": "2024" }
}
```

### `GET /api/v1/stats`

Returns runtime statistics. **Requires auth** when auth is enabled.

```json
{
  "ok": true,
  "data": {
    "total_requests": 12345,
    "total_events": 9876,
    "active_ws_sessions": 3,
    "uptime_seconds": 3600,
    "subsystem_invocations": { "kernel": 100, "skills": 50 },
    "subsystem_errors": {}
  }
}
```

---

## Cognitive Endpoints

### `POST /api/v1/cognitive/cycle`

Runs a single cognitive cycle (Observe → Understand → Create Goals →
Plan → Execute → Verify → Learn → Improve).

When `--real-kernel` is passed to `qbitos serve`, dispatches through the
real `CognitiveKernel`. Otherwise returns a deterministic stub response.

**Request (`CognitiveCycleRequest`):**

```json
{
  "input": "What is 2+2?",
  "modality": "text",
  "stream_thoughts": false,
  "max_cycles": 1
}
```

| Field            | Type    | Default | Notes                                       |
|------------------|---------|---------|---------------------------------------------|
| `input`          | string  | —       | Required. The prompt or observation.        |
| `modality`       | string? | `"text"`| Hint: `text`, `image`, `audio`, etc.        |
| `stream_thoughts`| bool    | `false` | If true, intermediate thoughts are emitted. |
| `max_cycles`     | int     | `1`     | Clamped to `[1, 16]`.                       |

**Response (`CognitiveCycleResponse`):**

```json
{
  "ok": true,
  "data": {
    "cycle_id": "550e8400-e29b-41d4-a716-446655440000",
    "output": "Processed: What is 2+2?",
    "phases": ["observe", "understand", "create_goals", "plan",
               "execute", "verify", "learn", "improve"],
    "duration_ms": 12,
    "confidence": 0.9
  }
}
```

`confidence` is `0.9` when the real kernel's `cycle.verified == Some(true)`,
`0.7` when verification fails, `0.85` for the stub path, `0.5` for empty input.

---

## Memory Endpoints

### `POST /api/v1/memory`

Stores a memory in the specified layer. Triggers a `memory_stored` event.

**Request (`StoreMemoryRequest`):**

```json
{
  "content": "The user prefers concise answers.",
  "layer": "L3_semantic",
  "metadata": { "source": "user_feedback" }
}
```

| Field      | Type   | Default        | Notes                                              |
|------------|--------|----------------|----------------------------------------------------|
| `content`  | string | —              | Required. The memory content (text).               |
| `layer`    | string | `"L1_working"` | One of L0_sensory, L1_working, L2_episodic, L3_semantic, L4_procedural, L5_meta. |
| `metadata` | object | `{}`           | Optional metadata payload.                         |

**Response:**

```json
{
  "ok": true,
  "data": { "memory_id": "mem-550e8400-e29b-41d4-a716-446655440000", "stored": true }
}
```

### `POST /api/v1/memory/search`

Searches memories by similarity. Triggers no events.

**Request (`SearchMemoryRequest`):**

```json
{
  "query": "user preferences",
  "k": 10,
  "min_score": 0.5,
  "layer": "L3_semantic"
}
```

**Response:** Array of `MemorySearchResult`:

```json
{
  "ok": true,
  "data": [
    {
      "id": "mem-abc123",
      "content": "The user prefers concise answers.",
      "score": 0.92,
      "layer": "L3_semantic",
      "metadata": { "source": "user_feedback" }
    }
  ]
}
```

### `DELETE /api/v1/memory/{id}`

Deletes a memory by ID.

```bash
curl -X DELETE -H "Authorization: Bearer $TOKEN" \
     http://localhost:8080/api/v1/memory/mem-abc123
```

---

## Skill Endpoints

### `GET /api/v1/skills`

Lists all registered skills with their stats.

```json
{
  "ok": true,
  "data": {
    "skills": [
      {
        "id": 1,
        "name": "summarize",
        "description": "Summarize text",
        "confidence": 0.85,
        "execution_count": 42,
        "success_rate": 0.95,
        "steps": 2
      }
    ]
  }
}
```

### `POST /api/v1/skills/execute`

Executes a skill by name or numeric ID. The skill's `Infer` steps
dispatch through the `InferBackend` (NativeAgent if real-backends is
enabled, stub otherwise). `Tool` steps dispatch through the
`ToolBackend` (real ToolModule if real-backends, noop otherwise).

**Request (`ExecuteSkillRequest`):**

```json
{
  "skill_id": "summarize",
  "input": { "text": "Long text to summarize..." },
  "timeout_ms": 5000
}
```

**Response (`SkillExecutionResult`):**

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
        { "name": "infer", "kind": "infer", "output": {...}, "duration_ms": 12, "success": true },
        { "name": "return", "kind": "return", "output": "Summary: ...", "duration_ms": 0, "success": true }
      ],
      "total_duration_ms": 15
    },
    "duration_ms": 16
  }
}
```

---

## Training Endpoints

### `POST /api/v1/training/step`

Runs N training steps. Each step:
1. Records an experience in the replay buffer.
2. Records a curriculum attempt.
3. Attempts curriculum advancement if thresholds are met.

**Request:**

```json
{ "steps": 10, "stage": "easy" }
```

**Response (`TrainReportResponse`):**

```json
{
  "ok": true,
  "data": {
    "steps_executed": 10,
    "metric": 0.34,
    "duration_ms": 2,
    "improvements_applied": 0
  }
}
```

### `GET /api/v1/curriculum`

Returns the current curriculum state.

```json
{
  "ok": true,
  "data": {
    "current_stage": 1,
    "total_stages": 5,
    "stages": [
      { "id": "stage_1", "name": "Warm-up", "difficulty": "trivial", "success_rate": 0.95, "attempts": 20, "successes": 19 },
      { "id": "stage_2", "name": "Basic skills", "difficulty": "easy", "success_rate": 0.85, "attempts": 16, "successes": 14 },
      { "id": "stage_3", "name": "Intermediate", "difficulty": "medium", "success_rate": 0, "attempts": 0, "successes": 0 },
      { "id": "stage_4", "name": "Advanced", "difficulty": "hard", "success_rate": 0, "attempts": 0, "successes": 0 },
      { "id": "stage_5", "name": "Frontier", "difficulty": "expert", "success_rate": 0, "attempts": 0, "successes": 0 }
    ]
  }
}
```

### `GET /api/v1/replay/stats`

Experience replay buffer statistics.

```json
{
  "ok": true,
  "data": { "len": 150, "mean_reward": 0.72 }
}
```

### `GET /api/v1/meta/strategies`

Meta-learner strategy statistics.

```json
{
  "ok": true,
  "data": {
    "strategies": [
      { "strategy": "curriculum", "uses": 30, "mean_reward": 0.78, "last_reward": 0.85, "confidence": 0.92 },
      { "strategy": "experience_replay", "uses": 25, "mean_reward": 0.65, "last_reward": 0.70, "confidence": 0.88 }
    ]
  }
}
```

---

## Event Endpoints

### `POST /api/v1/events/subscribe`

Subscribe to the event stream for HTTP polling.

```json
{ "filter": "cognitive_cycle_completed" }
```

Returns a `subscription_id` and the current event ID cursor:

```json
{
  "ok": true,
  "data": { "subscription_id": 1, "from_event_id": 42 }
}
```

### `POST /api/v1/events/poll`

Poll for events since the last cursor.

```json
{ "subscription_id": 1, "max": 100 }
```

Returns an array of events:

```json
{
  "ok": true,
  "data": {
    "subscription_id": 1,
    "events": [
      {
        "id": 43,
        "event": {
          "kind": "cognitive_cycle_completed",
          "cycle_id": "abc-123",
          "phase": "improve",
          "duration_ms": 12,
          "summary": "Processed: hello"
        }
      }
    ]
  }
}
```

### `POST /api/v1/events/replay`

Replays events from the persistent event store back into the in-memory
buffer. Only works if `--persistent-events` was passed to `qbitos serve`.

```json
{ /* empty body */ }
```

**Response:**

```json
{
  "ok": true,
  "data": { "replayed": 42, "buffer_len": 50 }
}
```

### `GET /api/v1/events/lookup/{id}`

Look up an event by ID in the persistent store.

```bash
curl http://localhost:8080/api/v1/events/lookup/43
```

```json
{
  "ok": true,
  "data": {
    "found": true,
    "event": {
      "id": 43,
      "timestamp": "2026-07-19T12:34:56.789Z",
      "event": { "kind": "cognitive_cycle_completed", "..." : "..." }
    }
  }
}
```

### `GET /api/v1/events/ws` (WebSocket)

Bidirectional WebSocket connection. **Public** (no auth required).

On connect, the server sends:

```json
{ "kind": "session_started", "payload": { "session_id": 1, "instance_id": "...", "instance_name": "..." } }
```

Then continuously streams events as they are published:

```json
{ "kind": "event", "payload": { "id": 43, "timestamp": "...", "event": { "kind": "...", ... } } }
```

Client can send commands:

```json
{ "kind": "ping", "payload": { "hello": "world" } }
```

Response:

```json
{ "kind": "pong", "payload": { "echo": { "hello": "world" } } }
```

Other commands: `cycle` (run cognitive cycle), `stats` (get stats).

---

## Event Types

The `event` field in event envelopes is tagged with `kind`:

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

## JSON-RPC 2.0

The QbitOS service also supports JSON-RPC 2.0 over the same HTTP
endpoint (programmatic clients). All methods are also exposed via the
in-process `RpcServer` for embedded use.

### Methods

| Method              | Params                  | Returns                            |
|---------------------|-------------------------|------------------------------------|
| `ping`              | —                       | `{ "pong": true }`                 |
| `stats`             | —                       | Service stats object               |
| `health`            | —                       | Health summary object              |
| `cognitive.cycle`   | `{ "input": "..." }`    | `{ "cycle_id": "...", "output": "..." }` |
| `memory.search`     | `{ "query": "..." }`    | `{ "results": [...] }`             |
| `skills.list`       | —                       | `{ "skills": [...] }`              |
| `training.step`     | `{ "steps": N }`        | `{ "steps_executed": N, "mean_reward": X }` |

### Single request

```bash
curl -X POST http://localhost:8080/api/v1/rpc \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"ping","id":1}'
```

Response:

```json
{ "jsonrpc": "2.0", "result": { "pong": true }, "id": 1 }
```

### Batch request

```bash
curl -X POST http://localhost:8080/api/v1/rpc \
     -H "Content-Type: application/json" \
     -d '[
       {"jsonrpc":"2.0","method":"ping","id":1},
       {"jsonrpc":"2.0","method":"stats","id":2}
     ]'
```

Response: array of responses in same order.

### Error object

```json
{
  "jsonrpc": "2.0",
  "error": { "code": -32601, "message": "Method not found: foo.bar" },
  "id": 1
}
```

### Standard error codes

| Code   | Meaning                |
|--------|------------------------|
| -32700 | Parse error            |
| -32600 | Invalid request        |
| -32601 | Method not found       |
| -32602 | Invalid params         |
| -32603 | Internal error         |
| -32001 | Not found              |
| -32002 | Unauthorized           |
| -32003 | Conflict               |
| -32004 | Rate limited           |
| -32005 | Timeout                |
| -32006 | Shutting down          |
| -32007 | Unsupported capability |
| -32008 | Not initialized        |

---

## OpenAPI Spec

The full OpenAPI 3.1 specification is generated at runtime and served at:

- `GET /api/v1/openapi.json` — JSON
- `GET /api/v1/openapi.yaml` — YAML (if compiled with `serde_yaml`)

Use this to generate client SDKs in any language:

```bash
curl http://localhost:8080/api/v1/openapi.json > qbitos-openapi.json
npx openapi-typescript qbitos-openapi.json -o qbitos-api.ts
```

---

## SDK Examples

### Python (requests)

```python
import requests

BASE = "http://localhost:8080"
TOKEN = "secret"

headers = {"Authorization": f"Bearer {TOKEN}"}

# Run a cognitive cycle
r = requests.post(
    f"{BASE}/api/v1/cognitive/cycle",
    headers=headers,
    json={"input": "Hello, QbitOS!", "max_cycles": 1},
)
print(r.json()["data"]["output"])
```

### JavaScript (browser)

```javascript
const BASE = "http://localhost:8080";
const TOKEN = "secret";

// Cognitive cycle
const resp = await fetch(`${BASE}/api/v1/cognitive/cycle`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${TOKEN}`,
  },
  body: JSON.stringify({ input: "Hello, QbitOS!", max_cycles: 1 }),
});
const data = await resp.json();
console.log(data.data.output);
```

### WebSocket (browser)

```javascript
const ws = new WebSocket("ws://localhost:8080/api/v1/events/ws");
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.kind === "event") {
    console.log("event:", msg.payload.event.kind);
  }
};
ws.onopen = () => {
  ws.send(JSON.stringify({ kind: "ping", payload: { hello: "world" } }));
};
```

### Rust (reqwest)

```rust
use reqwest::Client;

let client = Client::new();
let resp: serde_json::Value = client
    .post("http://localhost:8080/api/v1/cognitive/cycle")
    .bearer_auth("secret")
    .json(&serde_json::json!({"input": "Hello!"}))
    .send()
    .await?
    .json()
    .await?;
println!("{}", resp["data"]["output"]);
```

---

## See Also

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — System architecture
- [`DEPLOYMENT.md`](./DEPLOYMENT.md) — Production deployment
- [`../README.md`](../README.md) — Top-level overview
