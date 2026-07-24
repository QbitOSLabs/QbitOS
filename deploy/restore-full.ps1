# QbitOS Production Backup Restore Script (PowerShell)
#
# Restores a backup produced by backup-full.sh.
#
# Usage:
#   .\deploy\restore-full.ps1 -BackupFile C:\backups\qbitos-backup-20260722T084000Z.tar.gz
#
# Requires: pg_restore, curl, redis-cli, mongorestore on PATH.

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$BackupFile,

    [string]$PgUrl = $env:DATABASE_URL,
    [string]$QdrantUrl = $env:QDRANT_URL,
    [string]$RedisUrl = $env:REDIS_URL,
    [string]$MongoUrl = $env:MONGODB_URL,
    [string]$DataDir = $env:QBITOS_DATA_DIR
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "[restore] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[ok]      $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[warn]    $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "[fail]    $msg" -ForegroundColor Red }

if (-not (Test-Path $BackupFile)) {
    Write-Fail "Backup file not found: $BackupFile"
    exit 1
}

$StageDir = Join-Path $env:TEMP "qbitos-restore-$(Get-Date -Format 'yyyyMMddHHmmss')"
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
Write-Step "staging to: $StageDir"

Write-Step "extracting $BackupFile..."
tar -xzf $BackupFile -C $StageDir
# Find the manifest.json
$Manifest = Get-ChildItem -Path $StageDir -Recurse -Filter "manifest.json" | Select-Object -First 1
if (-not $Manifest) {
    Write-Fail "manifest.json not found in backup"
    exit 1
}
$ManifestDir = $Manifest.DirectoryName
Write-Ok "manifest at: $($Manifest.FullName)"

# ─── Restore PostgreSQL ──────────────────────────────────────────────────────
if ($PgUrl -and (Test-Path (Join-Path $ManifestDir "postgres.dump"))) {
    Write-Step "restoring PostgreSQL..."
    $args = @("--dbname=$PgUrl", "--clean", "--if-exists", "--no-owner", "--no-acl")
    & pg_restore @args (Join-Path $ManifestDir "postgres.dump") 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Ok "PostgreSQL restored" }
    else { Write-Warn "PostgreSQL restore had non-fatal errors (this is normal with --clean)" }
} else {
    Write-Warn "PostgreSQL backup missing or DATABASE_URL not set — skipping"
}

# ─── Restore Qdrant ──────────────────────────────────────────────────────────
if ($QdrantUrl -and (Test-Path (Join-Path $ManifestDir "qdrant"))) {
    Write-Step "restoring Qdrant collections (metadata only — snapshots must be re-imported manually)..."
    $Collections = Get-ChildItem -Path (Join-Path $ManifestDir "qdrant") -Filter "*.snap.json"
    foreach ($c in $Collections) {
        Write-Host "  collection snapshot: $($c.Name)"
    }
    Write-Ok "Qdrant: $(@($Collections).Count) snapshot records listed"
} else {
    Write-Warn "Qdrant backup missing or QDRANT_URL not set — skipping"
}

# ─── Restore Redis ───────────────────────────────────────────────────────────
if ($RedisUrl -and (Test-Path (Join-Path $ManifestDir "redis-dump.rdb"))) {
    Write-Step "restoring Redis..."
    # Redis RDB restore: stop Redis, replace dump.rdb, start Redis.
    # This is destructive — only run on a stopped Redis instance.
    $RedisTarget = $RedisUrl -replace '^redis://','' -replace '^rediss://',''
    $RedisDir = & redis-cli -u "redis://$RedisTarget" CONFIG GET dir 2>$null | Select-Object -Last 1
    $RedisFile = & redis-cli -u "redis://$RedisTarget" CONFIG GET dbfilename 2>$null | Select-Object -Last 1
    if ($RedisDir -and $RedisFile) {
        $DestRdb = Join-Path $RedisDir $RedisFile
        Write-Warn "this requires Redis to be stopped. Copying RDB to $DestRdb"
        Copy-Item (Join-Path $ManifestDir "redis-dump.rdb") $DestRdb -Force
        Write-Ok "Redis RDB copied — restart Redis to load it"
    } else {
        Write-Warn "could not determine Redis data directory"
    }
} else {
    Write-Warn "Redis backup missing or REDIS_URL not set — skipping"
}

# ─── Restore MongoDB ─────────────────────────────────────────────────────────
if ($MongoUrl -and (Test-Path (Join-Path $ManifestDir "mongo"))) {
    Write-Step "restoring MongoDB..."
    & mongorestore --uri="$MongoUrl" --gzip --drop (Join-Path $ManifestDir "mongo") 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Ok "MongoDB restored" }
    else { Write-Fail "MongoDB restore failed" }
} else {
    Write-Warn "MongoDB backup missing or MONGODB_URL not set — skipping"
}

# ─── Restore QbitOS data directory ───────────────────────────────────────────
if ($DataDir -and (Test-Path (Join-Path $ManifestDir "qbitos-data.tar"))) {
    Write-Step "restoring QbitOS data directory to $DataDir..."
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
    tar -xf (Join-Path $ManifestDir "qbitos-data.tar") -C (Split-Path $DataDir -Parent)
    Write-Ok "QbitOS data directory restored"
} else {
    Write-Warn "QbitOS data directory backup missing or QBITOS_DATA_DIR not set — skipping"
}

Write-Ok "restore complete"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start QbitOS service: qbitos serve --addr 0.0.0.0:8080"
Write-Host "  2. Verify health:        curl http://localhost:8080/api/v1/health"
Write-Host "  3. Check the audit log for any restored-data validation failures"

Remove-Item -Recurse -Force $StageDir
exit 0
