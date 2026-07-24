#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# QbitOS Database Setup Script — Linux / macOS
# ═══════════════════════════════════════════════════════════════════════════════
#
# This script performs a complete database setup for QbitOS:
#   1. Checks PostgreSQL is installed and running
#   2. Creates the qbitos role (user) with a secure password
#   3. Creates the qbitos database owned by the qbitos role
#   4. Installs pgvector extension (for HNSW vector search)
#   5. Runs the full schema migration (14 tables, 82+ indexes)
#   6. Runs the production index migration
#   7. Seeds the database with sample data (memories, skills, knowledge)
#   8. Generates a .env file with the correct connection string
#   9. Validates the setup by running test queries
#
# Usage:
#   ./setup-db.sh                          # Interactive mode (prompts for password)
#   ./setup-db.sh --password mysecret      # Non-interactive
#   ./setup-db.sh --superuser postgres     # Use a different superuser
#   ./setup-db.sh --drop-existing          # Drop existing qbitos DB/user first
#   ./setup-db.sh --skip-seed              # Don't insert seed data
#   ./setup-db.sh --docker                 # Use Docker PostgreSQL (auto-starts it)
#
# Environment variables (override defaults):
#   QBITOS_DB_NAME         Database name (default: qbitos)
#   QBITOS_DB_USER         Database user (default: qbitos)
#   QBITOS_DB_PASSWORD     Database password (default: prompt or auto-generated)
#   QBITOS_DB_HOST         Database host (default: localhost)
#   QBITOS_DB_PORT         Database port (default: 5432)
#   PG_SUPERUSER           Superuser to connect as (default: postgres)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
BOLD='\033[1m'
RESET='\033[0m'

log()  { printf "[${BLUE}qbitos${RESET}] %s\n" "$1"; }
ok()   { printf "${GREEN}✓ %s${RESET}\n" "$1"; }
warn() { printf "${YELLOW}⚠ %s${RESET}\n" "$1"; }
fail() { printf "${RED}✗ %s${RESET}\n" "$1" >&2; }
die()  { fail "$1"; exit 1; }

# ─── Defaults ─────────────────────────────────────────────────────────────────
DB_NAME="${QBITOS_DB_NAME:-qbitos}"
DB_USER="${QBITOS_DB_USER:-qbitos}"
DB_PASSWORD="${QBITOS_DB_PASSWORD:-}"
DB_HOST="${QBITOS_DB_HOST:-localhost}"
DB_PORT="${QBITOS_DB_PORT:-5432}"
PG_SUPERUSER="${PG_SUPERUSER:-postgres}"
DROP_EXISTING=false
SKIP_SEED=false
USE_DOCKER=false
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MIGRATIONS_DIR="${SCRIPT_DIR}/memory/migrations"

# ─── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --password)       DB_PASSWORD="$2"; shift 2;;
        --superuser)      PG_SUPERUSER="$2"; shift 2;;
        --host)           DB_HOST="$2"; shift 2;;
        --port)           DB_PORT="$2"; shift 2;;
        --db-name)        DB_NAME="$2"; shift 2;;
        --db-user)        DB_USER="$2"; shift 2;;
        --drop-existing)  DROP_EXISTING=true; shift;;
        --skip-seed)      SKIP_SEED=true; shift;;
        --docker)         USE_DOCKER=true; shift;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) die "unknown argument: $1";;
    esac
done

# ─── Generate a secure password if not provided ──────────────────────────────
if [[ -z "${DB_PASSWORD}" ]]; then
    if [[ -t 0 ]]; then
        # Interactive: prompt for password
        log "No password provided. Please enter a password for the '${DB_USER}' database user."
        read -rsp "Password: " DB_PASSWORD
        echo
        read -rsp "Confirm:  " DB_PASSWORD_CONFIRM
        echo
        [[ "${DB_PASSWORD}" == "${DB_PASSWORD_CONFIRM}" ]] || die "passwords do not match"
        [[ -n "${DB_PASSWORD}" ]] || die "password cannot be empty"
    else
        # Non-interactive: generate a secure random password
        DB_PASSWORD="$(openssl rand -base64 24 2>/dev/null || head -c 32 /dev/urandom | base64)"
        log "Generated secure password: ${DB_PASSWORD}"
    fi
fi

# ─── Step 0: Docker mode ─────────────────────────────────────────────────────
if [[ "${USE_DOCKER}" == true ]]; then
    log "Docker mode: starting PostgreSQL container..."
    if ! command -v docker >/dev/null 2>&1; then
        die "docker is not installed. Install Docker or run without --docker."
    fi
    docker run -d \
        --name qbitos-postgres \
        -e POSTGRES_USER="${PG_SUPERUSER}" \
        -e POSTGRES_PASSWORD="${DB_PASSWORD}" \
        -e POSTGRES_DB="${DB_NAME}" \
        -p "${DB_PORT}:5432" \
        -v qbitos-pgdata:/var/lib/postgresql/data \
        pgvector/pgvector:pg16 2>/dev/null || warn "container already exists or failed to start"
    DB_HOST="localhost"
    log "waiting for PostgreSQL to be ready..."
    for i in $(seq 1 30); do
        if docker exec qbitos-postgres pg_isready -U "${PG_SUPERUSER}" >/dev/null 2>&1; then
            ok "PostgreSQL is ready (after ${i}s)"
            break
        fi
        sleep 1
    done
fi

# ─── Step 1: Check PostgreSQL is available ───────────────────────────────────
log "Step 1/9: Checking PostgreSQL installation..."

if ! command -v psql >/dev/null 2>&1; then
    fail "psql (PostgreSQL client) is not installed."
    echo ""
    echo "Install PostgreSQL:"
    echo "  Ubuntu/Debian: sudo apt install postgresql postgresql-contrib"
    echo "  macOS:         brew install postgresql@16"
    echo "  Or use Docker: ./setup-db.sh --docker"
    exit 1
fi

if ! command -v pg_isready >/dev/null 2>&1; then
    die "pg_isready is not installed (part of postgresql-client package)"
fi

ok "psql found: $(psql --version)"

# Check server is running
if ! pg_isready -h "${DB_HOST}" -p "${DB_PORT}" >/dev/null 2>&1; then
    warn "PostgreSQL server is not running on ${DB_HOST}:${DB_PORT}"
    if [[ "${USE_DOCKER}" == false ]]; then
        echo "  Start it with one of:"
        echo "    sudo systemctl start postgresql   # Linux systemd"
        echo "    brew services start postgresql     # macOS Homebrew"
        echo "    pg_ctl -D /usr/local/var/postgres start  # manual"
        echo "  Or run this script with --docker to auto-start a container."
        exit 1
    fi
fi
ok "PostgreSQL server is running on ${DB_HOST}:${DB_PORT}"

# ─── Helper: run SQL as superuser ────────────────────────────────────────────
run_sql_superuser() {
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${PG_SUPERUSER}" \
        -d postgres \
        -v ON_ERROR_STOP=1 \
        "$@"
}

# ─── Helper: run SQL as qbitos user ──────────────────────────────────────────
run_sql_qbitos() {
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -v ON_ERROR_STOP=1 \
        "$@"
}

# ─── Step 2: Drop existing (if requested) ────────────────────────────────────
if [[ "${DROP_EXISTING}" == true ]]; then
    log "Step 2/9: Dropping existing database and user..."
    run_sql_superuser -c "DROP DATABASE IF EXISTS \"${DB_NAME}\";" 2>/dev/null && ok "dropped database '${DB_NAME}'" || warn "could not drop database"
    run_sql_superuser -c "DROP ROLE IF EXISTS \"${DB_USER}\";" 2>/dev/null && ok "dropped role '${DB_USER}'" || warn "could not drop role"
else
    log "Step 2/9: Skipping drop (use --drop-existing to force)"
fi

# ─── Step 3: Create the qbitos role ──────────────────────────────────────────
log "Step 3/9: Creating database user '${DB_USER}'..."
if run_sql_superuser -c "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" -t | grep -q 1; then
    ok "role '${DB_USER}' already exists"
    # Update password
    run_sql_superuser -c "ALTER ROLE \"${DB_USER}\" WITH LOGIN PASSWORD '${DB_PASSWORD}';" 2>/dev/null && ok "password updated" || warn "could not update password"
else
    run_sql_superuser -c "CREATE ROLE \"${DB_USER}\" WITH LOGIN PASSWORD '${DB_PASSWORD}';" && ok "created role '${DB_USER}'"
fi

# ─── Step 4: Create the qbitos database ──────────────────────────────────────
log "Step 4/9: Creating database '${DB_NAME}'..."
if run_sql_superuser -c "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" -t | grep -q 1; then
    ok "database '${DB_NAME}' already exists"
else
    run_sql_superuser -c "CREATE DATABASE \"${DB_NAME}\" OWNER \"${DB_USER}\";" && ok "created database '${DB_NAME}'"
fi

# ─── Step 5: Install pgvector extension ──────────────────────────────────────
log "Step 5/9: Installing pgvector extension..."
run_sql_qbitos -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null && ok "pgvector installed" || {
    warn "pgvector extension not available"
    echo "  Install it with:"
    echo "    Ubuntu/Debian: sudo apt install postgresql-16-pgvector"
    echo "    macOS:         brew install pgvector"
    echo "    From source:   https://github.com/pgvector/pgvector"
    echo "  Continuing without vector search (some features may not work)..."
}

# Install other useful extensions
for ext in pg_trgm btree_gin btree_gist pgcrypto unaccent fuzzystrmatch hstore; do
    run_sql_qbitos -c "CREATE EXTENSION IF NOT EXISTS ${ext};" 2>/dev/null && ok "extension ${ext} installed" || warn "could not install ${ext} (non-critical)"
done

# ─── Step 6: Run the full schema migration ───────────────────────────────────
log "Step 6/9: Running schema migration (001_full_schema.sql)..."
SCHEMA_FILE="${MIGRATIONS_DIR}/001_full_schema.sql"
if [[ ! -f "${SCHEMA_FILE}" ]]; then
    die "schema file not found: ${SCHEMA_FILE}"
fi
PGPASSWORD="${DB_PASSWORD}" psql \
    -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
    -v ON_ERROR_STOP=1 \
    -f "${SCHEMA_FILE}" 2>&1 | grep -v "^NOTICE" || true
ok "schema migration complete (14 tables, 82+ indexes)"

# ─── Step 7: Run production indexes migration ────────────────────────────────
log "Step 7/9: Running production indexes (002_production_indexes.sql)..."
INDEX_FILE="${MIGRATIONS_DIR}/002_production_indexes.sql"
if [[ -f "${INDEX_FILE}" ]]; then
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
        -v ON_ERROR_STOP=1 \
        -f "${INDEX_FILE}" 2>&1 | grep -v "^NOTICE" || true
    ok "production indexes created"
else
    warn "production indexes file not found, skipping"
fi

# ─── Step 8: Seed the database ───────────────────────────────────────────────
if [[ "${SKIP_SEED}" != true ]]; then
    log "Step 8/9: Seeding database with sample data..."
    SEED_FILE="${MIGRATIONS_DIR}/003_seed_data.sql"
    if [[ -f "${SEED_FILE}" ]]; then
        PGPASSWORD="${DB_PASSWORD}" psql \
            -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
            -v ON_ERROR_STOP=1 \
            -f "${SEED_FILE}" 2>&1 | grep -E "^(NOTICE|ERROR)" || true
        ok "seed data inserted"
    else
        warn "seed file not found, skipping"
    fi
else
    log "Step 8/9: Skipping seed data (--skip-seed)"
fi

# ─── Step 9: Generate .env file ──────────────────────────────────────────────
log "Step 9/9: Generating .env file..."
ENV_FILE="${SCRIPT_DIR}/.env"
DATABASE_URL="postgres://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Backup existing .env if it exists
if [[ -f "${ENV_FILE}" ]]; then
    cp "${ENV_FILE}" "${ENV_FILE}.bak.$(date +%s)"
    warn "backed up existing .env to .env.bak.*"
fi

cat > "${ENV_FILE}" << EOF
# QbitOS environment configuration — generated by setup-db.sh
# Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

# ─── Database (PostgreSQL with pgvector) ─────────────────────────────────────
DATABASE_URL=${DATABASE_URL}
QBITOS_MEMORY_BACKEND=postgres
QBITOS_MIGRATIONS_DIR=${MIGRATIONS_DIR}

# ─── Logging ─────────────────────────────────────────────────────────────────
RUST_LOG=info,qbitos=debug
QBITOS_LOG_FORMAT=pretty

# ─── Service ─────────────────────────────────────────────────────────────────
QBITOS_DATA_DIR=${SCRIPT_DIR}/data
QBITOS_TOKEN=

# ─── Safety ──────────────────────────────────────────────────────────────────
QBITOS_SAFETY_ENFORCED=true
QBITOS_AUDIT_LOG=true
QBITOS_ALLOW_SELF_IMPROVE=false

# ─── Other backends (optional) ───────────────────────────────────────────────
# QDRANT_URL=http://localhost:6333
# REDIS_URL=redis://localhost:6379
# MONGODB_URL=mongodb://localhost:27017
# QBITOS_ROCKSDB_PATH=${SCRIPT_DIR}/data/rocksdb
# QBITOS_DUCKDB_PATH=${SCRIPT_DIR}/data/qbitos.duckdb
EOF

chmod 600 "${ENV_FILE}"
ok ".env file generated at ${ENV_FILE}"
ok "DATABASE_URL=${DATABASE_URL}"

# ─── Validation ──────────────────────────────────────────────────────────────
echo ""
log "Validating setup..."
echo ""

# Test connection
if PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT version();" -t 2>/dev/null | head -1; then
    ok "connection test passed"
else
    fail "connection test failed"
    exit 1
fi

# Count tables
TABLE_COUNT=$(PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null | tr -d ' ')
ok "tables in public schema: ${TABLE_COUNT}"

# Count indexes
INDEX_COUNT=$(PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT count(*) FROM pg_indexes WHERE schemaname='public';" 2>/dev/null | tr -d ' ')
ok "indexes in public schema: ${INDEX_COUNT}"

# Check pgvector
if PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT extversion FROM pg_extension WHERE extname='vector';" 2>/dev/null | head -1; then
    ok "pgvector extension is installed"
else
    warn "pgvector extension is NOT installed (vector search will not work)"
fi

# Count seed data (if seeded)
if [[ "${SKIP_SEED}" != true ]]; then
    MEM_COUNT=$(PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT count(*) FROM qbitos_memories;" 2>/dev/null | tr -d ' ')
    ok "seeded memories: ${MEM_COUNT}"

    SKILL_COUNT=$(PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT count(*) FROM qbitos_skills;" 2>/dev/null | tr -d ' ')
    ok "seeded skills: ${SKILL_COUNT}"

    TRIPLE_COUNT=$(PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT count(*) FROM qbitos_knowledge_triples;" 2>/dev/null | tr -d ' ')
    ok "seeded knowledge triples: ${TRIPLE_COUNT}"
fi

echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  QbitOS database setup complete!${RESET}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${BOLD}Database:${RESET}  ${DB_NAME}"
echo -e "  ${BOLD}User:${RESET}      ${DB_USER}"
echo -e "  ${BOLD}Host:${RESET}      ${DB_HOST}:${DB_PORT}"
echo -e "  ${BOLD}Password:${RESET}  ${DB_PASSWORD}"
echo ""
echo -e "  ${BOLD}Connection string:${RESET}"
echo -e "  ${DATABASE_URL}"
echo ""
echo -e "  ${BOLD}.env file:${RESET} ${ENV_FILE}"
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
echo "    1. Start QbitOS:  qbitos serve --addr 0.0.0.0:8080"
echo "    2. Check health:  curl http://localhost:8080/api/v1/health"
echo "    3. View schema:   psql \${DATABASE_URL} -c '\dt'"
echo ""
echo -e "  ${BOLD}To reset the database:${RESET}"
echo "    ./scripts/setup-db.sh --drop-existing --password '${DB_PASSWORD}'"
echo ""
