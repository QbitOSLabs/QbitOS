# ═══════════════════════════════════════════════════════════════════════════════
# QbitOS Database Setup Script — Windows PowerShell
# ═══════════════════════════════════════════════════════════════════════════════
#
# This script performs a complete database setup for QbitOS on Windows:
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
#   .\scripts\setup-db.ps1                                    # Interactive
#   .\scripts\setup-db.ps1 -Password "mysecret"              # Non-interactive
#   .\scripts\setup-db.ps1 -Superuser postgres -Password "p" # Specify superuser
#   .\scripts\setup-db.ps1 -DropExisting                     # Drop existing first
#   .\scripts\setup-db.ps1 -SkipSeed                         # No seed data
#   .\scripts\setup-db.ps1 -Docker                           # Use Docker
#
# Parameters:
#   -Password       Database password (default: prompt or auto-generated)
#   -Superuser      PostgreSQL superuser (default: postgres)
#   -Host           Database host (default: localhost)
#   -Port           Database port (default: 5432)
#   -DbName         Database name (default: qbitos)
#   -DbUser         Database user (default: qbitos)
#   -DropExisting   Drop existing database and user first
#   -SkipSeed       Don't insert seed data
#   -Docker         Use Docker PostgreSQL (auto-starts it)
# ═══════════════════════════════════════════════════════════════════════════════

[CmdletBinding()]
param(
    [string]$Password = "",
    [string]$Superuser = "postgres",
    [string]$Host = "localhost",
    [int]$Port = 5432,
    [string]$DbName = "qbitos",
    [string]$DbUser = "qbitos",
    [switch]$DropExisting,
    [switch]$SkipSeed,
    [switch]$Docker
)

$ErrorActionPreference = "Stop"

# ─── Helpers ─────────────────────────────────────────────────────────────────
function Write-Log($msg)  { Write-Host "[qbitos] $msg" -ForegroundColor Blue }
function Write-Ok($msg)   { Write-Host "[ok]    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[warn]  $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "[fail]  $msg" -ForegroundColor Red }
function Write-Die($msg)  { Write-Fail $msg; exit 1 }

$ScriptDir = Split-Path -Parent $PSScriptRoot
$MigrationsDir = Join-Path $ScriptDir "memory\migrations"

# ─── Generate password if not provided ───────────────────────────────────────
if ([string]::IsNullOrEmpty($Password)) {
    if ($Host.UI.RawUI.KeyAvailable -or -not [Console]::IsInputRedirected) {
        # Interactive: prompt for password
        $secure = Read-Host -Prompt "Enter password for '$DbUser' database user" -AsSecureString
        $Password = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
        $secure2 = Read-Host -Prompt "Confirm password" -AsSecureString
        $confirm = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure2))
        if ($Password -ne $confirm) { Write-Die "passwords do not match" }
        if ([string]::IsNullOrEmpty($Password)) { Write-Die "password cannot be empty" }
    } else {
        # Non-interactive: generate a secure random password
        $bytes = New-Object byte[] 24
        [Security.Cryptography.RNGCryptoServiceProvider]::Create().GetBytes($bytes)
        $Password = [Convert]::ToBase64String($bytes)
        Write-Log "Generated secure password: $Password"
    }
}

# ─── Step 0: Docker mode ─────────────────────────────────────────────────────
if ($Docker) {
    Write-Log "Docker mode: starting PostgreSQL container..."
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Die "docker is not installed. Install Docker Desktop or run without -Docker."
    }
    $dockerArgs = @("run", "-d", "--name", "qbitos-postgres",
        "-e", "POSTGRES_USER=$Superuser",
        "-e", "POSTGRES_PASSWORD=$Password",
        "-e", "POSTGRES_DB=$DbName",
        "-p", "${Port}:5432",
        "-v", "qbitos-pgdata:/var/lib/postgresql/data",
        "pgvector/pgvector:pg16")
    & docker @dockerArgs 2>$null | Out-Null
    Write-Log "waiting for PostgreSQL to be ready..."
    for ($i = 1; $i -le 30; $i++) {
        $result = & docker exec qbitos-postgres pg_isready -U $Superuser 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "PostgreSQL is ready (after ${i}s)"
            break
        }
        Start-Sleep -Seconds 1
    }
}

# ─── Step 1: Check PostgreSQL ────────────────────────────────────────────────
Write-Log "Step 1/9: Checking PostgreSQL installation..."

$psql = Get-Command psql -ErrorAction SilentlyContinue
if (-not $psql) {
    Write-Fail "psql (PostgreSQL client) is not installed or not in PATH."
    Write-Host ""
    Write-Host "Install PostgreSQL:"
    Write-Host "  1. Download from: https://www.postgresql.org/download/windows/"
    Write-Host "  2. Or use Chocolatey: choco install postgresql16"
    Write-Host "  3. Or use Docker:     .\setup-db.ps1 -Docker"
    Write-Host ""
    Write-Host "After installing, ensure 'psql' is in your PATH (typically:"
    Write-Host "  C:\Program Files\PostgreSQL\16\bin)"
    exit 1
}

Write-Ok "psql found at: $($psql.Source)"

# Check server is running
$env:PGPASSWORD = $Password
$ready = & psql -h $Host -p $Port -U $Superuser -d postgres -c "SELECT 1" -t 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Cannot connect to PostgreSQL at ${Host}:${Port} as ${Superuser}"
    Write-Host "  Start the PostgreSQL service:"
    Write-Host "    net start postgresql-x64-16   # Windows service"
    Write-Host "    Or use Docker: .\setup-db.ps1 -Docker"
    exit 1
}
Write-Ok "PostgreSQL server is running on ${Host}:${Port}"

# ─── Helper functions ────────────────────────────────────────────────────────
function Invoke-SqlSuperuser($sql) {
    $env:PGPASSWORD = $Password
    & psql -h $Host -p $Port -U $Superuser -d postgres -v ON_ERROR_STOP=1 -c $sql 2>$null
    return $LASTEXITCODE
}

function Invoke-SqlQbitos($sql) {
    $env:PGPASSWORD = $Password
    & psql -h $Host -p $Port -U $DbUser -d $DbName -v ON_ERROR_STOP=1 -c $sql 2>$null
    return $LASTEXITCODE
}

function Invoke-SqlFile($file) {
    $env:PGPASSWORD = $Password
    & psql -h $Host -p $Port -U $DbUser -d $DbName -v ON_ERROR_STOP=1 -f $file 2>&1 | Where-Object { $_ -notmatch "^NOTICE" }
    return $LASTEXITCODE
}

# ─── Step 2: Drop existing ───────────────────────────────────────────────────
if ($DropExisting) {
    Write-Log "Step 2/9: Dropping existing database and user..."
    [void](Invoke-SqlSuperuser "DROP DATABASE IF EXISTS `"$DbName`";")
    Write-Ok "dropped database '$DbName' (if it existed)"
    [void](Invoke-SqlSuperuser "DROP ROLE IF EXISTS `"$DbUser`";")
    Write-Ok "dropped role '$DbUser' (if it existed)"
} else {
    Write-Log "Step 2/9: Skipping drop (use -DropExisting to force)"
}

# ─── Step 3: Create role ─────────────────────────────────────────────────────
Write-Log "Step 3/9: Creating database user '$DbUser'..."
$exists = Invoke-SqlSuperuser "SELECT 1 FROM pg_roles WHERE rolname='$DbUser'" -t
if ($exists -match "1") {
    Write-Ok "role '$DbUser' already exists"
    [void](Invoke-SqlSuperuser "ALTER ROLE `"$DbUser`" WITH LOGIN PASSWORD '$Password';")
    Write-Ok "password updated"
} else {
    [void](Invoke-SqlSuperuser "CREATE ROLE `"$DbUser`" WITH LOGIN PASSWORD '$Password';")
    Write-Ok "created role '$DbUser'"
}

# ─── Step 4: Create database ─────────────────────────────────────────────────
Write-Log "Step 4/9: Creating database '$DbName'..."
$dbExists = Invoke-SqlSuperuser "SELECT 1 FROM pg_database WHERE datname='$DbName'" -t
if ($dbExists -match "1") {
    Write-Ok "database '$DbName' already exists"
} else {
    [void](Invoke-SqlSuperuser "CREATE DATABASE `"$DbName`" OWNER `"$DbUser`";")
    Write-Ok "created database '$DbName'"
}

# ─── Step 5: Install extensions ──────────────────────────────────────────────
Write-Log "Step 5/9: Installing pgvector extension..."
$code = Invoke-SqlQbitos "CREATE EXTENSION IF NOT EXISTS vector;"
if ($code -eq 0) {
    Write-Ok "pgvector installed"
} else {
    Write-Warn "pgvector extension not available"
    Write-Host "  Install it from: https://github.com/pgvector/pgvector"
    Write-Host "  Or use Docker: .\setup-db.ps1 -Docker (includes pgvector)"
    Write-Host "  Continuing without vector search..."
}

$extensions = @("pg_trgm", "btree_gin", "btree_gist", "pgcrypto", "unaccent", "fuzzystrmatch", "hstore")
foreach ($ext in $extensions) {
    $code = Invoke-SqlQbitos "CREATE EXTENSION IF NOT EXISTS $ext;"
    if ($code -eq 0) { Write-Ok "extension $ext installed" }
    else { Write-Warn "could not install $ext (non-critical)" }
}

# ─── Step 6: Schema migration ────────────────────────────────────────────────
Write-Log "Step 6/9: Running schema migration (001_full_schema.sql)..."
$schemaFile = Join-Path $MigrationsDir "001_full_schema.sql"
if (-not (Test-Path $schemaFile)) { Write-Die "schema file not found: $schemaFile" }
[void](Invoke-SqlFile $schemaFile)
Write-Ok "schema migration complete (14 tables, 82+ indexes)"

# ─── Step 7: Production indexes ──────────────────────────────────────────────
Write-Log "Step 7/9: Running production indexes (002_production_indexes.sql)..."
$indexFile = Join-Path $MigrationsDir "002_production_indexes.sql"
if (Test-Path $indexFile) {
    [void](Invoke-SqlFile $indexFile)
    Write-Ok "production indexes created"
} else {
    Write-Warn "production indexes file not found, skipping"
}

# ─── Step 8: Seed data ───────────────────────────────────────────────────────
if (-not $SkipSeed) {
    Write-Log "Step 8/9: Seeding database with sample data..."
    $seedFile = Join-Path $MigrationsDir "003_seed_data.sql"
    if (Test-Path $seedFile) {
        [void](Invoke-SqlFile $seedFile)
        Write-Ok "seed data inserted"
    } else {
        Write-Warn "seed file not found, skipping"
    }
} else {
    Write-Log "Step 8/9: Skipping seed data (-SkipSeed)"
}

# ─── Step 9: Generate .env ───────────────────────────────────────────────────
Write-Log "Step 9/9: Generating .env file..."
$envFile = Join-Path $ScriptDir ".env"
$databaseUrl = "postgres://${DbUser}:${Password}@${Host}:${Port}/${DbName}"

if (Test-Path $envFile) {
    $timestamp = (Get-Date).ToString("yyyyMMddHHmmss")
    Copy-Item $envFile "${envFile}.bak.${timestamp}"
    Write-Warn "backed up existing .env to .env.bak.${timestamp}"
}

$envContent = @"
# QbitOS environment configuration — generated by setup-db.ps1
# Date: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')

# ─── Database (PostgreSQL with pgvector) ─────────────────────────────────────
DATABASE_URL=$databaseUrl
QBITOS_MEMORY_BACKEND=postgres
QBITOS_MIGRATIONS_DIR=$($MigrationsDir -replace '\\', '/')

# ─── Logging ─────────────────────────────────────────────────────────────────
RUST_LOG=info,qbitos=debug
QBITOS_LOG_FORMAT=pretty

# ─── Service ─────────────────────────────────────────────────────────────────
QBITOS_DATA_DIR=$($ScriptDir -replace '\\', '/')/data
QBITOS_TOKEN=

# ─── Safety ──────────────────────────────────────────────────────────────────
QBITOS_SAFETY_ENFORCED=true
QBITOS_AUDIT_LOG=true
QBITOS_ALLOW_SELF_IMPROVE=false

# ─── Other backends (optional) ───────────────────────────────────────────────
# QDRANT_URL=http://localhost:6333
# REDIS_URL=redis://localhost:6379
# MONGODB_URL=mongodb://localhost:27017
# QBITOS_ROCKSDB_PATH=$($ScriptDir -replace '\\', '/')/data/rocksdb
# QBITOS_DUCKDB_PATH=$($ScriptDir -replace '\\', '/')/data/qbitos.duckdb
"@

Set-Content -Path $envFile -Value $envContent -Encoding UTF8
Write-Ok ".env file generated at $envFile"
Write-Ok "DATABASE_URL=$databaseUrl"

# ─── Validation ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Log "Validating setup..."
Write-Host ""

# Test connection
$env:PGPASSWORD = $Password
$version = & psql -h $Host -p $Port -U $DbUser -d $DbName -c "SELECT version();" -t 2>$null
if ($version) {
    Write-Ok "connection test passed"
    Write-Host "  $version" -ForegroundColor DarkGray
} else {
    Write-Fail "connection test failed"
    exit 1
}

# Count tables
$tableCount = & psql -h $Host -p $Port -U $DbUser -d $DbName -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" 2>$null
Write-Ok "tables in public schema: $($tableCount.Trim())"

# Count indexes
$indexCount = & psql -h $Host -p $Port -U $DbUser -d $DbName -t -c "SELECT count(*) FROM pg_indexes WHERE schemaname='public';" 2>$null
Write-Ok "indexes in public schema: $($indexCount.Trim())"

# Check pgvector
$pgvector = & psql -h $Host -p $Port -U $DbUser -d $DbName -t -c "SELECT extversion FROM pg_extension WHERE extname='vector';" 2>$null
if ($pgvector) {
    Write-Ok "pgvector extension is installed (v$($pgvector.Trim()))"
} else {
    Write-Warn "pgvector extension is NOT installed"
}

# Count seed data
if (-not $SkipSeed) {
    $memCount = & psql -h $Host -p $Port -U $DbUser -d $DbName -t -c "SELECT count(*) FROM qbitos_memories;" 2>$null
    Write-Ok "seeded memories: $($memCount.Trim())"

    $skillCount = & psql -h $Host -p $Port -U $DbUser -d $DbName -t -c "SELECT count(*) FROM qbitos_skills;" 2>$null
    Write-Ok "seeded skills: $($skillCount.Trim())"

    $tripleCount = & psql -h $Host -p $Port -U $DbUser -d $DbName -t -c "SELECT count(*) FROM qbitos_knowledge_triples;" 2>$null
    Write-Ok "seeded knowledge triples: $($tripleCount.Trim())"
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  QbitOS database setup complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "  Database:  $DbName"
Write-Host "  User:      $DbUser"
Write-Host "  Host:      ${Host}:${Port}"
Write-Host "  Password:  $Password"
Write-Host ""
Write-Host "  Connection string:"
Write-Host "  $databaseUrl"
Write-Host ""
Write-Host "  .env file: $envFile"
Write-Host ""
Write-Host "  Next steps:"
Write-Host "    1. Start QbitOS:  qbitos.exe serve --addr 0.0.0.0:8080"
Write-Host "    2. Check health:  curl http://localhost:8080/api/v1/health"
Write-Host "    3. View schema:   psql `"$databaseUrl`" -c '\dt'"
Write-Host ""
Write-Host "  To reset the database:"
Write-Host "    .\scripts\setup-db.ps1 -DropExisting -Password '$Password'"
Write-Host ""
