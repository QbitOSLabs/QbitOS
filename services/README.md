# qbitos-services

HTTP/REST (Axum), WebSocket, JSON-RPC 2.0, OpenAPI 3.1, auth.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Exposes the cognitive OS to external clients via four protocols.

## Protocols

| Protocol   | Endpoint                  | Use case                              |
|------------|---------------------------|---------------------------------------|
| HTTP/REST  | /api/v1/*                 | Browser, curl, scripts                |
| WebSocket  | /api/v1/events/ws         | Live event streaming, command dispatch|
| JSON-RPC   | programmatic              | Programmatic clients (batch support)  |
| OpenAPI    | /api/v1/openapi.json      | Spec generation, client SDK generation|

## Features

- **19 HTTP endpoints**: health, version, stats, cognitive cycle, memory CRUD, skills, training, curriculum, replay, meta-strategies, events (subscribe/poll/replay/lookup), WebSocket, OpenAPI
- **Bearer-token auth**: SHA-256 hashed, constant-time compared via subtle::ConstantTimeEq
- **Persistent event store**: events survive across restarts via the memory fabric's semantic layer
- **Real cognitive kernel integration**: --real-kernel flag dispatches through the real 8-stage loop
- **CORS policies**: permissive (dev) or restricted (production)

## CLI Integration

```bash
qbitos serve --addr 0.0.0.0:8080 \
  --auth-token $QBITOS_TOKEN \
  --real-kernel \
  --persistent-events \
  --restricted-cors
```

## Usage

```rust
use qbitos_services::{CognitiveService, HttpServer, ServiceConfig};
let svc = CognitiveService::new(ServiceConfig::default());
let server = HttpServer::new(svc);
let handle = server.serve(([0,0,0,0], 8080).into()).await?;
handle.wait().await?;
```

## Testing

39 tests including 11 auth tests and 5 event_store tests.
