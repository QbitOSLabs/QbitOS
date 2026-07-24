# QbitOS Uninstall Script (Windows)
# Usage: Run as Administrator: .\deploy\uninstall.ps1

param(
    [string]$InstallDir = "C:\Program Files\QbitOS",
    [string]$DataDir = "C:\ProgramData\QbitOS",
    [string]$ServiceName = "QbitOS"
)

$ErrorActionPreference = "Stop"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "     QbitOS Uninstallation" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Stop service
$service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($service) {
    if ($service.Status -eq 'Running') {
        Write-Host "Stopping QbitOS service..." -ForegroundColor Yellow
        Stop-Service -Name $ServiceName -Force
    }
    Write-Host "Removing Windows service..." -ForegroundColor Yellow
    sc.exe delete $ServiceName | Out-Null
} else {
    Write-Host "Service '$ServiceName' not found, skipping." -ForegroundColor Yellow
}

# Remove binary
$exePath = Join-Path $InstallDir "qbitos.exe"
if (Test-Path $exePath) {
    Write-Host "Removing binary..." -ForegroundColor Yellow
    Remove-Item $exePath -Force
}

# Remove install directory
if (Test-Path $InstallDir) {
    Write-Host "Removing install directory..." -ForegroundColor Yellow
    Remove-Item $InstallDir -Recurse -Force
}

# Remove data directory (ask)
if (Test-Path $DataDir) {
    $removeData = Read-Host "Remove data directory ($DataDir)? [y/N]"
    if ($removeData -eq 'y' -or $removeData -eq 'Y') {
        Write-Host "Removing data directory..." -ForegroundColor Yellow
        Remove-Item $DataDir -Recurse -Force
    } else {
        Write-Host "Data directory preserved at $DataDir" -ForegroundColor Yellow
    }
}

# Remove from PATH
$path = [Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($path -and $path.Contains($InstallDir)) {
    $newPath = $path -replace [regex]::Escape(";$InstallDir"), "" -replace [regex]::Escape("$InstallDir;"), ""
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
    Write-Host "Removed QbitOS from system PATH." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "  Uninstallation complete." -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
