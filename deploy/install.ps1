# QbitOS Windows Installation Script
# Usage: Run as Administrator: .\deploy\install.ps1
#
# Requirements:
#   - Rust 1.85+ (https://rustup.rs)
#   - Visual Studio C++ Build Tools
#   - PowerShell 5.1+

param(
    [string]$InstallDir = "C:\Program Files\QbitOS",
    [string]$DataDir = "C:\ProgramData\QbitOS",
    [string]$ServiceName = "QbitOS"
)

$ErrorActionPreference = "Stop"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "     QbitOS Windows Installation" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Create directories
Write-Host "Creating directories..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\rocksdb" | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\models" | Out-Null

# Check for binary
$binaryPath = "target\release\qbitos.exe"
if (Test-Path $binaryPath) {
    Write-Host "Installing binary..."
    Copy-Item $binaryPath "$InstallDir\qbitos.exe" -Force
} else {
    Write-Host "Binary not found at $binaryPath" -ForegroundColor Yellow
    Write-Host "Build first with: cargo build --release -p qbitos-cli"
    Write-Host "Or build with PostgreSQL: cargo build --release -p qbitos-cli"
    exit 1
}

# Copy migrations
Write-Host "Copying migrations..."
if (Test-Path "memory\migrations") {
    Copy-Item -Path "memory\migrations" -Destination "$InstallDir\migrations" -Recurse -Force
}

# Create env file
$envFile = "$DataDir\qbitos.env"
if (-not (Test-Path $envFile)) {
    Write-Host "Creating environment config..."
    @"
QBITOS_DATA_DIR=$DataDir
RUST_LOG=info
QBITOS_TOKEN=
DATABASE_URL=
QDRANT_URL=
REDIS_URL=
QBITOS_ROCKSDB_PATH=$DataDir\rocksdb
QBITOS_DUCKDB_PATH=$DataDir\qbitos.duckdb
"@ | Out-File -FilePath $envFile -Encoding UTF8
}

# Create Windows Service using sc.exe
$exePath = "$InstallDir\qbitos.exe"
$serviceExists = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($serviceExists) {
    Write-Host "Service '$ServiceName' already exists. Stopping and removing..."
    Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
    sc.exe delete $ServiceName | Out-Null
}

Write-Host "Creating Windows Service '$ServiceName'..."
sc.exe create $ServiceName binPath= "$exePath serve --addr 0.0.0.0:8080" start= auto | Out-Null
sc.exe description $ServiceName "QbitOS Cognitive AI Operating System" | Out-Null

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host "  Config:  $envFile"
Write-Host "  Start:    Start-Service $ServiceName"
Write-Host "  Stop:     Stop-Service $ServiceName"
Write-Host "  Status:   Get-Service $ServiceName"
Write-Host "  Logs:     Get-EventLog -LogName Application -Source $ServiceName"
Write-Host "==============================================" -ForegroundColor Green
