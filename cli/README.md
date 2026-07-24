# qbitos-cli

Top-level orchestrator: qbitos binary wiring all subsystems.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

The qbitos binary is the user-facing entry point. It wires all 16 QbitOS subsystems into a coherent runtime.

## Subcommands

| Command   | Description                                       |
|-----------|---------------------------------------------------|
| repl      | Boot the cognitive kernel and drop into REPL      |
| status    | Show kernel status (memory, cycles, CPU/RAM)      |
| ask       | Run a one-shot cognitive cycle on a prompt        |
| serve     | Start the HTTP/REST + WebSocket service layer     |
| dataset   | Dataset operations (list, ingest, mine)           |
| memory    | Memory operations (stats, search)                 |
| skill     | Skill operations (list, show)                     |
| tool      | Tool operations (list, invoke)                    |
| world     | World model operations (stats, query)             |
| evolution | Evolution engine operations (run, status)         |
| state     | State persistence (save, load)                    |
| backend   | Memory backend operations (attach, list)          |
| reason    | Reasoning engine operations (prove, solve)        |
| safety    | Safety operations (check, audit)                  |
| learn     | Learning operations (feedback, train)             |

## The serve Subcommand

```bash
qbitos serve [OPTIONS]

Options:
      --addr <ADDR>              Bind address [default: 127.0.0.1:8080]
      --name <NAME>              Instance name [default: qbitos-cli]
      --no-safety                Disable safety enforcement
      --allow-self-improve       Enable self-improvement
      --auth-token <TOK>         Enable Bearer-token auth
      --real-kernel              Attach the real CognitiveKernel
      --persistent-events        Attach persistent event store
      --restricted-cors          Use restricted CORS policy
```

## Features

- default = [postgres-backed] — PostgreSQL primary backend
- real-backends — Wire real NativeAgent + ToolModule into SkillModule
- postgres-backed / qdrant-backed / redis-backed / rocksdb-backed — DB backend passthrough
- all-backends — Enable all DB backends at once

## Testing

10 integration tests (default) / 13 (with real-backends).
