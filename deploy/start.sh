#!/usr/bin/env bash
set -euo pipefail

# QbitOS Quick Start Script (Linux/macOS)
# Starts QbitOS with the configured backends

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Load .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null || true
fi

# Default values
ADDR="${QBITOS_SERVE_ADDR:-0.0.0.0:8080}"
NAME="${QBITOS_INSTANCE_NAME:-qbitos-dev}"

# Build if needed
if [ ! -f target/release/qbitos ]; then
    echo "Building QbitOS (release)..."
    cargo build --release -p qbitos-cli
fi

# Start
echo "Starting QbitOS on $ADDR..."
exec ./target/release/qbitos serve \
    --addr "$ADDR" \
    --name "$NAME" \
    --persistent-events \
    "$@"
