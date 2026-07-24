<#
.SYNOPSIS
    QbitOS Environment Validation Script (Windows PowerShell).

.DESCRIPTION
    Validates that all required environment variables and external dependencies
    are configured correctly before launching the QbitOS service. Exits 0 if
    everything is healthy, 1 with a diagnostic message otherwise.

.EXAMPLE
    .\scripts\validate-env.ps1

.ENVIRONMENT
    RUST_LOG                  - tracing filter directive (required)
    QBITOS_MEMORY_BACKEND     - one of: postgres, qdrant, redis, mongodb, rocksdb, sqlite, duckdb, sled
    DATABASE_URL              - PostgreSQL URL (required if backend=postgres)
    QDRANT_URL                - Qdrant HTTP endpoint (required if backend=qdrant)
    REDIS_URL                 - Redis URL (required if backend=redis)
    MONGODB_URL               - MongoDB URL (required if backend=mongodb)
    QBITOS_DATA_DIR           - writable data directory (default: .\data)
    QBITOS_MIGRATIONS_DIR     - migrations directory (default: .\memory\migrations)
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3

# ─── Helpers ─────────────────────────────────────────────────────────────────
$script:Warnings = 0

function Write-Pass { param([string]$Message) Write-Host "PASS: $Message" -ForegroundColor Green }
function Write-WarnMsg {
    param([string]$Message)
    Write-Host "WARN: $Message" -ForegroundColor Yellow
    $script:Warnings++
}
function Write-Fail {
    param([string]$Message)
    Write-Host "FAIL: $Message" -ForegroundColor Red
}
function Exit-Fail {
    param([string]$Message)
    Write-Fail $Message
    exit 1
}

# Resolve repo root (parent of scripts dir).
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# ─── 1. RUST_LOG ─────────────────────────────────────────────────────────────
if (-not $env:RUST_LOG) {
    Exit-Fail "RUST_LOG is not set. Example: `$env:RUST_LOG = 'info,qbitos=debug'"
}
Write-Pass "RUST_LOG is set: $env:RUST_LOG"

# ─── 2. Memory backend selection ─────────────────────────────────────────────
$Backend = if ($env:QBITOS_MEMORY_BACKEND) { $env:QBITOS_MEMORY_BACKEND } else { "postgres" }
$ValidBackends = @("postgres", "qdrant", "redis", "mongodb", "rocksdb", "sqlite", "duckdb", "sled")
if ($ValidBackends -notcontains $Backend) {
    Exit-Fail "QBITOS_MEMORY_BACKEND='$Backend' is not a recognized backend"
}
Write-Pass "Memory backend: $Backend"

# ─── 3. DATABASE_URL format (postgres) ───────────────────────────────────────
if ($Backend -eq "postgres") {
    if (-not $env:DATABASE_URL) {
        Exit-Fail "QBITOS_MEMORY_BACKEND=postgres requires DATABASE_URL to be set"
    }
    if ($env:DATABASE_URL -notmatch '^postgres(ql)?://[^:]+:[^@]+@[^:/]+:[0-9]+/.+$') {
        Exit-Fail "DATABASE_URL has invalid format. Expected: postgres://user:pass@host:5432/dbname (got: $($env:DATABASE_URL))"
    }
    Write-Pass "DATABASE_URL format is valid"

    # ─── 3a. Migrations directory exists (postgres) ─────────────────────────
    $MigrationsDir = if ($env:QBITOS_MIGRATIONS_DIR) {
        $env:QBITOS_MIGRATIONS_DIR
    } else {
        Join-Path $RepoRoot "memory\migrations"
    }
    if (-not (Test-Path $MigrationsDir -PathType Container)) {
        Exit-Fail "Migrations directory not found: $MigrationsDir (set QBITOS_MIGRATIONS_DIR to override)"
    }
    $sqlFiles = Get-ChildItem -Path $MigrationsDir -Filter "*.sql" -File -ErrorAction SilentlyContinue
    if (-not $sqlFiles -or $sqlFiles.Count -eq 0) {
        Write-WarnMsg "Migrations directory exists but contains no .sql files: $MigrationsDir"
    } else {
        Write-Pass "Migrations directory: $MigrationsDir"
    }
}

# ─── 4. QDRANT_URL reachable (qdrant) ────────────────────────────────────────
if ($Backend -eq "qdrant") {
    if (-not $env:QDRANT_URL) {
        Exit-Fail "QBITOS_MEMORY_BACKEND=qdrant requires QDRANT_URL to be set"
    }
    $HealthUrl = $env:QDRANT_URL.TrimEnd('/') + "/healthz"
    try {
        $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) {
            Exit-Fail "QDRANT_URL ($($env:QDRANT_URL)) returned status $($response.StatusCode) at /healthz"
        }
        Write-Pass "Qdrant reachable: $($env:QDRANT_URL)"
    } catch {
        Exit-Fail "QDRANT_URL ($($env:QDRANT_URL)) is not reachable at /healthz. Error: $($_.Exception.Message)"
    }
}

# ─── 5. REDIS_URL reachable (redis) ──────────────────────────────────────────
if ($Backend -eq "redis") {
    if (-not $env:REDIS_URL) {
        Exit-Fail "QBITOS_MEMORY_BACKEND=redis requires REDIS_URL to be set"
    }
    $redisCli = Get-Command redis-cli -ErrorAction SilentlyContinue
    if (-not $redisCli) {
        Write-WarnMsg "redis-cli not installed; cannot ping $($env:REDIS_URL) - skipping reachability check"
    } else {
        try {
            $pingOutput = & redis-cli -u $env:REDIS_URL ping 2>&1
            if ($LASTEXITCODE -ne 0 -or $pingOutput -notmatch 'PONG') {
                Exit-Fail "REDIS_URL ($($env:REDIS_URL)) is not reachable (redis-cli ping failed: $pingOutput)"
            }
            Write-Pass "Redis reachable: $($env:REDIS_URL)"
        } catch {
            Exit-Fail "REDIS_URL ($($env:REDIS_URL)) is not reachable. Error: $($_.Exception.Message)"
        }
    }
}

# ─── 6. MONGODB_URL format (mongodb) ─────────────────────────────────────────
if ($Backend -eq "mongodb") {
    if (-not $env:MONGODB_URL) {
        Exit-Fail "QBITOS_MEMORY_BACKEND=mongodb requires MONGODB_URL to be set"
    }
    if ($env:MONGODB_URL -notmatch '^mongodb(\+srv)?://.+') {
        Exit-Fail "MONGODB_URL has invalid format. Expected: mongodb://[user:pass@]host[:port]/dbname (got: $($env:MONGODB_URL))"
    }
    Write-Pass "MONGODB_URL format is valid"
}

# ─── 7. Data directory writable ──────────────────────────────────────────────
$DataDir = if ($env:QBITOS_DATA_DIR) { $env:QBITOS_DATA_DIR } else { Join-Path $RepoRoot "data" }
if (-not (Test-Path $DataDir -PathType Container)) {
    try {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    } catch {
        Exit-Fail "Data directory does not exist and could not be created: $DataDir. Error: $($_.Exception.Message)"
    }
}
$testFile = Join-Path $DataDir ".qbitos-write-test-$([System.Diagnostics.Process]::GetCurrentProcess().Id)"
try {
    Set-Content -Path $testFile -Value "test" -Encoding ASCII -ErrorAction Stop
    Remove-Item -Path $testFile -Force -ErrorAction SilentlyContinue
} catch {
    Exit-Fail "Data directory is not writable: $DataDir. Error: $($_.Exception.Message)"
}
Write-Pass "Data directory writable: $DataDir"

# ─── Summary ─────────────────────────────────────────────────────────────────
if ($script:Warnings -gt 0) {
    Write-Host "Validation completed with $($script:Warnings) warning(s)." -ForegroundColor Yellow
}
Write-Host "All environment checks passed." -ForegroundColor Green
exit 0
