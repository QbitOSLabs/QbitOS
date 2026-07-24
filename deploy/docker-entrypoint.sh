#!/usr/bin/env bash
set -euo pipefail

# QbitOS Docker Entrypoint
# Waits for PostgreSQL to be ready, then starts QbitOS

echo "[QbitOS] Docker entrypoint starting..."

# Wait for PostgreSQL if DATABASE_URL is set
if [ -n "${DATABASE_URL:-}" ]; then
    echo "[QbitOS] Waiting for PostgreSQL..."
    for i in $(seq 1 30); do
        if pg_isready -d "$DATABASE_URL" >/dev/null 2>&1; then
            echo "[QbitOS] PostgreSQL is ready."
            break
        fi
        echo "[QbitOS]   PostgreSQL not ready (attempt $i/30), retrying in 2s..."
        sleep 2
    done
fi

# Wait for Qdrant if QDRANT_URL is set
if [ -n "${QDRANT_URL:-}" ]; then
    QDRANT_HOST=$(echo "$QDRANT_URL" | sed 's|http://||' | sed 's|:.*||')
    QDRANT_PORT=$(echo "$QDRANT_URL" | sed 's|.*:||')
    echo "[QbitOS] Waiting for Qdrant at $QDRANT_HOST:$QDRANT_PORT..."
    for i in $(seq 1 30); do
        if curl -sf "http://$QDRANT_HOST:$QDRANT_PORT/healthz" >/dev/null 2>&1; then
            echo "[QbitOS] Qdrant is ready."
            break
        fi
        echo "[QbitOS]   Qdrant not ready (attempt $i/30), retrying in 2s..."
        sleep 2
    done
fi

# Wait for Redis if REDIS_URL is set
if [ -n "${REDIS_URL:-}" ]; then
    REDIS_HOST=$(echo "$REDIS_URL" | sed 's|redis://||' | sed 's|:.*||')
    REDIS_PORT=$(echo "$REDIS_URL" | sed 's|.*:||')
    echo "[QbitOS] Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
    for i in $(seq 1 30); do
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
            echo "[QbitOS] Redis is ready."
            break
        fi
        echo "[QbitOS]   Redis not ready (attempt $i/30), retrying in 2s..."
        sleep 2
    done
fi

# Run migrations if auto-migrate is enabled (default)
if [ "${QBITOS_AUTO_MIGRATE:-true}" = "true" ] && [ -n "${DATABASE_URL:-}" ]; then
    echo "[QbitOS] Running database migrations..."
    # The qbitos binary auto-migrates on startup when --migrate flag is passed
    # or when QBITOS_AUTO_MIGRATE=true env var is set
fi

# Create data directory
mkdir -p "${QBITOS_DATA_DIR:-/var/lib/qbitos}"

# Start QbitOS
echo "[QbitOS] Starting QbitOS..."
exec qbitos "$@"
