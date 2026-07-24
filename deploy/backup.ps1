# QbitOS Backup Script (Windows)
# Usage: .\deploy\backup.ps1 [-BackupDir "C:\Backups"]
param(
    [string]$BackupDir = "C:\Backups\QbitOS"
)

$ErrorActionPreference = "Stop"
$Date = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupPath = Join-Path $BackupDir "qbitos-$Date"

Write-Host "Creating backup at $BackupPath..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $BackupPath | Out-Null

# State snapshot
$qbitosExe = Get-Command qbitos -ErrorAction SilentlyContinue
if ($qbitosExe) {
    Write-Host "Saving state snapshot..." -ForegroundColor Yellow
    try {
        & qbitos state save (Join-Path $BackupPath "state.json") 2>$null
        Write-Host "  State saved." -ForegroundColor Green
    } catch {
        Write-Host "  State save failed: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "  qbitos not in PATH, skipping state snapshot" -ForegroundColor Yellow
}

# PostgreSQL backup
$env:DATABASE_URL | Out-Null 2>$null
if ($env:DATABASE_URL) {
    Write-Host "Backing up PostgreSQL..." -ForegroundColor Yellow
    $pgDump = Get-Command pg_dump -ErrorAction SilentlyContinue
    if ($pgDump) {
        try {
            & pg_dump $env:DATABASE_URL | Out-File (Join-Path $BackupPath "postgres.sql") -Encoding UTF8
            Write-Host "  PostgreSQL backup complete." -ForegroundColor Green
        } catch {
            Write-Host "  pg_dump failed: $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  pg_dump not found, skipping PostgreSQL backup" -ForegroundColor Yellow
    }
}

# RocksDB backup
$rocksdbPath = $env:QBITOS_ROCKSDB_PATH
if ($rocksdbPath -and (Test-Path $rocksdbPath)) {
    Write-Host "Backing up RocksDB..." -ForegroundColor Yellow
    try {
        $dest = Join-Path $BackupPath "rocksdb"
        Copy-Item -Path $rocksdbPath -Destination $dest -Recurse -Force
        Write-Host "  RocksDB backup complete." -ForegroundColor Green
    } catch {
        Write-Host "  RocksDB backup failed: $_" -ForegroundColor Yellow
    }
}

# DuckDB backup
$duckdbPath = $env:QBITOS_DUCKDB_PATH
if ($duckdbPath -and (Test-Path $duckdbPath)) {
    Write-Host "Backing up DuckDB..." -ForegroundColor Yellow
    try {
        Copy-Item -Path $duckdbPath -Destination (Join-Path $BackupPath "qbitos.duckdb") -Force
        Write-Host "  DuckDB backup complete." -ForegroundColor Green
    } catch {
        Write-Host "  DuckDB backup failed: $_" -ForegroundColor Yellow
    }
}

# Compress
Write-Host "Compressing..." -ForegroundColor Yellow
$zipPath = "$BackupPath.zip"
Compress-Archive -Path "$BackupPath\*" -DestinationPath $zipPath -Force
Remove-Item -Recurse -Force $BackupPath

$size = (Get-Item $zipPath).Length / 1MB
Write-Host "Backup complete: $zipPath ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
