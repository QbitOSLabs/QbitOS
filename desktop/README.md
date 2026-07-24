# qbitos-desktop

Tauri v2 desktop app (React 19 + TypeScript + Tailwind CSS 4).

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

A desktop GUI for QbitOS built with Tauri v2.

## Views

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

## Development

```bash
cd desktop
npm install
npm run tauri dev
```

## Build

```bash
npm run tauri build
```

Produces native installers for Linux, macOS, and Windows.

## Architecture

- **Backend**: Rust (Tauri v2) — wires to the QbitOS workspace crates
- **Frontend**: React 19 + TypeScript + Tailwind CSS 4
- **State**: Zustand store
- **Icons**: lucide-react
