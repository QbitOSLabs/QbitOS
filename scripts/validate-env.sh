#!/usr/bin/env bash
# QbitOS Environment Validation Script (Unix / Linux / macOS)
#
# Validates that all required environment variables and external dependencies
# are configured correctly before launching the QbitOS service. Exits 0 if
# everything is healthy, 1 with a diagnostic message otherwise.
#
# Usage:
#   ./scripts/validate-env.sh
#
# Environment variables consulted:
#   RUST_LOG                  - tracing filter directive (required)
#   QBITOS_MEMORY_BACKEND     - one of: postgres, qdrant, redis, mongodb, rocksdb, sqlite, duckdb, sled
#   DATABASE_URL              - PostgreSQL URL (required if backend=postgres)
#   QDRANT_URL                - Qdrant HTTP endpoint (required if backend=qdrant)
#   REDIS_URL                 - Redis URL (required if backend=redis)
#   MONGODB_URL               - MongoDB URL (required if backend=mongodb)
#   QBITOS_DATA_DIR           - writable data directory (default: ./data)
#   QBITOS_MIGRATIONS_DIR     - migrations directory (default: ./memory/migrations)

set -euo pipefail

# ─── Helpers ─────────────────────────────────────────────────────────────────
RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RESET=$'\033[0m'

log_pass() { printf "${GREEN}✓ %s${RESET}\n" "$1"; }
log_warn() { printf "${YELLOW}⚠ %s${RESET}\n" "$1"; }
log_fail() { printf "${RED}✗ %s${RESET}\n" "$1" >&2; }
fail()     { log_fail "$1"; exit 1; }

# Track warnings to allow exit-with-warning if requested.
WARNINGS=0
warn() { log_warn "$1"; WARNINGS=$((WARNINGS + 1)); }

# ─── 1. RUST_LOG ─────────────────────────────────────────────────────────────
if [[ -z "${RUST_LOG:-}" ]]; then
    fail "RUST_LOG is not set. Example: export RUST_LOG=info,qbitos=debug"
fi
log_pass "RUST_LOG is set: ${RUST_LOG}"

# ─── 2. Memory backend selection ─────────────────────────────────────────────
BACKEND="${QBITOS_MEMORY_BACKEND:-postgres}"
case "${BACKEND}" in
    postgres|qdrant|redis|mongodb|rocksdb|sqlite|duckdb|sled) ;;
    *) fail "QBITOS_MEMORY_BACKEND='${BACKEND}' is not a recognized backend";;
esac
log_pass "Memory backend: ${BACKEND}"

# ─── 3. DATABASE_URL format (postgres) ───────────────────────────────────────
if [[ "${BACKEND}" == "postgres" ]]; then
    if [[ -z "${DATABASE_URL:-}" ]]; then
        fail "QBITOS_MEMORY_BACKEND=postgres requires DATABASE_URL to be set"
    fi
    if [[ ! "${DATABASE_URL}" =~ ^postgres(ql)?://[^:]+:[^@]+@[^:/]+:[0-9]+/.+$ ]]; then
        fail "DATABASE_URL has invalid format. Expected: postgres://user:pass@host:5432/dbname (got: ${DATABASE_URL})"
    fi
    log_pass "DATABASE_URL format is valid"

    # ─── 3a. Migrations directory exists (postgres) ───────────────────────────
    MIGRATIONS_DIR="${QBITOS_MIGRATIONS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/memory/migrations}"
    if [[ ! -d "${MIGRATIONS_DIR}" ]]; then
        fail "Migrations directory not found: ${MIGRATIONS_DIR} (set QBITOS_MIGRATIONS_DIR to override)"
    fi
    if ! compgen -G "${MIGRATIONS_DIR}/*.sql" > /dev/null; then
        warn "Migrations directory exists but contains no .sql files: ${MIGRATIONS_DIR}"
    else
        log_pass "Migrations directory: ${MIGRATIONS_DIR}"
    fi
fi

# ─── 4. QDRANT_URL reachable (qdrant) ────────────────────────────────────────
if [[ "${BACKEND}" == "qdrant" ]]; then
    if [[ -z "${QDRANT_URL:-}" ]]; then
        fail "QBITOS_MEMORY_BACKEND=qdrant requires QDRANT_URL to be set"
    fi
    if ! curl -fsS --max-time 5 "${QDRANT_URL%/}/healthz" > /dev/null 2>&1; then
        fail "QDRANT_URL (${QDRANT_URL}) is not reachable at /healthz"
    fi
    log_pass "Qdrant reachable: ${QDRANT_URL}"
fi

# ─── 5. REDIS_URL reachable (redis) ──────────────────────────────────────────
if [[ "${BACKEND}" == "redis" ]]; then
    if [[ -z "${REDIS_URL:-}" ]]; then
        fail "QBITOS_MEMORY_BACKEND=redis requires REDIS_URL to be set"
    fi
    if ! command -v redis-cli > /dev/null 2>&1; then
        warn "redis-cli not installed; cannot ping ${REDIS_URL} — skipping reachability check"
    else
        # Strip the redis:// scheme if present.
        REDIS_PING_TARGET="${REDIS_URL#redis://}"
        REDIS_PING_TARGET="${REDIS_PING_TARGET#rediss://}"
        if ! redis-cli -u "redis://${REDIS_PING_TARGET}" ping > /dev/null 2>&1; then
            fail "REDIS_URL (${REDIS_URL}) is not reachable (redis-cli ping failed)"
        fi
        log_pass "Redis reachable: ${REDIS_URL}"
    fi
fi

# ─── 6. MONGODB_URL format (mongodb) ─────────────────────────────────────────
if [[ "${BACKEND}" == "mongodb" ]]; then
    if [[ -z "${MONGODB_URL:-}" ]]; then
        fail "QBITOS_MEMORY_BACKEND=mongodb requires MONGODB_URL to be set"
    fi
    if [[ ! "${MONGODB_URL}" =~ ^mongodb(\+srv)?://.+ ]]; then
        fail "MONGODB_URL has invalid format. Expected: mongodb://[user:pass@]host[:port]/dbname (got: ${MONGODB_URL})"
    fi
    log_pass "MONGODB_URL format is valid"
fi

# ─── 7. Data directory writable ──────────────────────────────────────────────
DATA_DIR="${QBITOS_DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
if [[ ! -d "${DATA_DIR}" ]]; then
    if ! mkdir -p "${DATA_DIR}" 2>/dev/null; then
        fail "Data directory does not exist and could not be created: ${DATA_DIR}"
    fi
fi
TEST_FILE="${DATA_DIR}/.qbitos-write-test-$$"
if ! printf "test" > "${TEST_FILE}" 2>/dev/null; then
    fail "Data directory is not writable: ${DATA_DIR}"
fi
rm -f "${TEST_FILE}"
log_pass "Data directory writable: ${DATA_DIR}"

# ─── Summary ─────────────────────────────────────────────────────────────────
if [[ "${WARNINGS}" -gt 0 ]]; then
    printf "${YELLOW}Validation completed with ${WARNINGS} warning(s).${RESET}\n"
fi
printf "${GREEN}All environment checks passed.${RESET}\n"
exit 0
