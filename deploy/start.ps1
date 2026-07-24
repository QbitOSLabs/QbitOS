# QbitOS Quick Start Script (Windows)
# Starts QbitOS with the configured backends

param(
    [string]$Addr = "0.0.0.0:8080",
    [string]$Name = "qbitos-dev"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

# Load .env if present
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
}

# Build if needed
if (-not (Test-Path "target\release\qbitos.exe")) {
    Write-Host "Building QbitOS (release)..." -ForegroundColor Cyan
    cargo build --release -p qbitos-cli
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
}

# Start
Write-Host "Starting QbitOS on $Addr..." -ForegroundColor Green
& ".\target\release\qbitos.exe" serve --addr $Addr --name $Name --persistent-events
