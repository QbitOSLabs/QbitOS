# qbitos-examples

End-to-end demos for QbitOS.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Four runnable examples demonstrating QbitOS functionality.

## Examples

### cognitive_cycle

HTTP REST demo against a live server — exercises 7 endpoints:

```bash
qbitos serve --addr 127.0.0.1:8080 &
cargo run -p qbitos-examples --example cognitive_cycle
```

### ws_events

Connects to the WebSocket event stream and prints events.

### rpc_client

JSON-RPC 2.0 client — in-process and HTTP.

### self_improvement_loop

Full training loop in-process (200 steps, curriculum advancement, 4-strategy meta-learning).

## Running

```bash
cargo run -p qbitos-examples --example <name>
```
