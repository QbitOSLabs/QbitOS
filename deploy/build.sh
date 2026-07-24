#!/usr/bin/env bash
set -euo pipefail

# QbitOS Build Script (Linux/macOS)
# Usage: ./deploy/build.sh [--release] [--all-backends] [--real-backends]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RELEASE=""
FEATURES=""

for arg in "$@"; do
    case $arg in
        --release) RELEASE="--release" ;;
        --all-backends) FEATURES="$FEATURES all-backends" ;;
        --real-backends) FEATURES="$FEATURES real-backends" ;;
        --postgres) FEATURES="$FEATURES postgres-backed" ;;
        --qdrant) FEATURES="$FEATURES qdrant-backed" ;;
        --redis) FEATURES="$FEATURES redis-backed" ;;
        --rocksdb) FEATURES="$FEATURES rocksdb-backed" ;;
        --mongodb) FEATURES="$FEATURES mongodb-backed" ;;
        --duckdb) FEATURES="$FEATURES duckdb-backed" ;;
    esac
done

FEATURES=$(echo $FEATURES | xargs)

echo "Building QbitOS..."
echo "  Release: $RELEASE"
echo "  Features: ${FEATURES:-default}"

if [ -n "$FEATURES" ]; then
    cargo build $RELEASE -p qbitos-cli --features "$FEATURES"
else
    cargo build $RELEASE -p qbitos-cli
fi

echo "Build complete!"
echo "Binary: target/${RELEASE:+release}/debug/qbitos"
if [ -n "$RELEASE" ]; then
    echo "  target/release/qbitos"
fi
