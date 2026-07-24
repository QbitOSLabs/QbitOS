# QbitOS Health Check Script (Windows)
# Usage: .\deploy\healthcheck.ps1 [Url]
param(
    [string]$Url = "http://localhost:8080/api/v1/health"
)

$ErrorActionPreference = "Stop"

try {
    $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 5 -ErrorAction Stop
    $healthy = $response.ok
    if ($healthy) {
        Write-Host "OK: QbitOS is healthy ($Url -> 200, instance=$($response.data.instance_name))" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "WARN: QbitOS responded but reports unhealthy" -ForegroundColor Yellow
        exit 1
    }
} catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 0) {
        Write-Host "FAIL: QbitOS is not responding ($Url)" -ForegroundColor Red
    } else {
        Write-Host "FAIL: QbitOS returned HTTP $code ($Url)" -ForegroundColor Red
    }
    exit 1
}
