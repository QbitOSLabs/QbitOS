# QbitOS Build Script (Windows)
# Usage: .\deploy\build.ps1 [-Release] [-AllBackends] [-RealBackends]

param(
    [switch]$Release,
    [switch]$AllBackends,
    [switch]$RealBackends,
    [switch]$Postgres,
    [switch]$Qdrant,
    [switch]$Redis,
    [switch]$RocksDB,
    [switch]$MongoDB,
    [switch]$DuckDB
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$Args = @()
if ($Release) { $Args += "--release" }

$Features = @()
if ($AllBackends) { $Features += "all-backends" }
if ($RealBackends) { $Features += "real-backends" }
if ($Postgres) { $Features += "postgres-backed" }
if ($Qdrant) { $Features += "qdrant-backed" }
if ($Redis) { $Features += "redis-backed" }
if ($RocksDB) { $Features += "rocksdb-backed" }
if ($MongoDB) { $Features += "mongodb-backed" }
if ($DuckDB) { $Features += "duckdb-backed" }

Write-Host "Building QbitOS..." -ForegroundColor Cyan
Write-Host "  Release: $Release"
Write-Host "  Features: $($Features -join ' ')"

$CmdArgs = @("build", "-p", "qbitos-cli")
if ($Release) { $CmdArgs += "--release" }
if ($Features.Count -gt 0) {
    $CmdArgs += "--features"
    $CmdArgs += ($Features -join ",")
}

& cargo @CmdArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build complete!" -ForegroundColor Green
if ($Release) {
    Write-Host "Binary: target\release\qbitos.exe"
} else {
    Write-Host "Binary: target\debug\qbitos.exe"
}
