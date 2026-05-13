$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$funcExe = "C:\Program Files\Microsoft\Azure Functions Core Tools\func.exe"
$activateScript = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
$smokeScript = Join-Path $repoRoot "scripts\smoke-local.ps1"

function Test-PortListening {
    param([int]$Port)

    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    return $null -ne $conn
}

function Test-AzuriteReady {
    return (Test-PortListening -Port 20000) -and (Test-PortListening -Port 20001) -and (Test-PortListening -Port 20002)
}

function Wait-AzuriteReady {
    param(
        [int]$MaxAttempts = 20,
        [int]$DelaySeconds = 1
    )

    for ($i = 1; $i -le $MaxAttempts; $i++) {
        if (Test-AzuriteReady) {
            Write-Host "Azurite ports are ready (attempt $i/$MaxAttempts)."
            return $true
        }

        Start-Sleep -Seconds $DelaySeconds
    }

    return $false
}

function Wait-HttpReady {
    param(
        [string]$Url,
        [int]$MaxAttempts = 30,
        [int]$DelaySeconds = 2
    )

    for ($i = 1; $i -le $MaxAttempts; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -Method GET -UseBasicParsing -TimeoutSec 10
            if ($resp.StatusCode -eq 200) {
                Write-Host "Ready: $Url (attempt $i/$MaxAttempts)"
                return $true
            }
        }
        catch {
            if ($i -eq $MaxAttempts) {
                return $false
            }
        }

        Start-Sleep -Seconds $DelaySeconds
    }

    return $false
}

$azuriteReady = Test-AzuriteReady
if (-not $azuriteReady) {
    $anyAzuritePort = (Test-PortListening -Port 20000) -or (Test-PortListening -Port 20001) -or (Test-PortListening -Port 20002)
    if ($anyAzuritePort) {
        Write-Host "Detected partial Azurite port state. Waiting for startup to finish..."
    }
    else {
        Write-Host "Starting Azurite on ports 20000/20001/20002..."
        $azCmd = "azurite --location `"$repoRoot\.azurite`" --blobHost 127.0.0.1 --queueHost 127.0.0.1 --tableHost 127.0.0.1 --blobPort 20000 --queuePort 20001 --tablePort 20002 --skipApiVersionCheck"
        Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $azCmd | Out-Null
    }
}

if (-not (Wait-AzuriteReady)) {
    throw "Azurite failed to start on ports 20000/20001/20002."
}
Write-Host "Azurite is ready."

if (-not (Test-PortListening -Port 7071)) {
    Write-Host "Starting Azure Functions host on port 7071..."
    if (-not (Test-Path $activateScript)) {
        throw "Missing activate script: $activateScript"
    }
    if (-not (Test-Path $funcExe)) {
        throw "Missing func executable: $funcExe"
    }

    $funcCmd = "& '$activateScript'; & '$funcExe' host start"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $funcCmd | Out-Null
}

if (-not (Wait-HttpReady -Url "http://localhost:7071/health")) {
    throw "Functions host did not become healthy in time."
}

if (-not (Test-Path $smokeScript)) {
    throw "Missing smoke script: $smokeScript"
}

Write-Host "Running smoke checks..."
& $smokeScript
Write-Host "Local dev stack is ready."
