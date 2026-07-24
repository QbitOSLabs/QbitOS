# qbitos-tools

Tool VM with sandbox, permissions, and 9 built-in tools.

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

Provides a sandboxed execution environment for tool calls.

## Built-in Tools

| Tool                | Description                              | Permission |
|---------------------|------------------------------------------|------------|
| filesystem          | Read/write files within a root directory | Read/Write |
| terminal            | Execute shell commands (with timeout)    | Execute    |
| http                | HTTP GET/POST/PUT/DELETE requests        | Network    |
| calculator          | Arithmetic expression evaluator          | Read       |
| summarizer          | Text summarization                       | Read       |
| file_analyzer       | File format detection and metadata       | Read       |
| sentiment           | Sentiment analysis                       | Read       |
| entity_extractor    | Named entity extraction                  | Read       |
| time                | Current time / date arithmetic           | Read       |

## Permission System

Each agent has a set of permissions. Tool calls are checked against the agent's permissions before execution.

## Usage

```rust
use qbitos_tools::{ToolModule, tool_vm::ToolCall};
let tools = ToolModule::new().with_builtin_tools().with_memory(memory);
let call = ToolCall::new("calculator", json!({"expression": "2+2"}), agent_id);
let outcome = tools.executor.execute(call).await?;
```

## Testing

9 tests covering tool registration, dispatch, and permissions.
