$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$maxAttempts = 6
$retryDelaySeconds = 5
$requestTimeoutSeconds = 15

$endpoints = @(
    @{ Name = "health"; Url = "http://localhost:7071/health"; ExpectedStatus = 200 },
    @{ Name = "root"; Url = "http://localhost:7071/"; ExpectedStatus = 200 }
)

foreach ($endpoint in $endpoints) {
    $lastError = $null

    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        try {
            $resp = Invoke-WebRequest -Uri $endpoint.Url -Method GET -UseBasicParsing -TimeoutSec $requestTimeoutSeconds
            if ($resp.StatusCode -ne $endpoint.ExpectedStatus) {
                throw "Unexpected status for $($endpoint.Name): $($resp.StatusCode)"
            }

            Write-Host "PASS $($endpoint.Name): $($resp.StatusCode) (attempt $attempt/$maxAttempts)"
            $lastError = $null
            break
        }
        catch {
            $lastError = $_.Exception.Message
            if ($attempt -lt $maxAttempts) {
                Write-Host "RETRY $($endpoint.Name): $lastError (attempt $attempt/$maxAttempts)"
                Start-Sleep -Seconds $retryDelaySeconds
            }
        }
    }

    if ($lastError) {
        Write-Error "FAIL $($endpoint.Name): $lastError"
        exit 1
    }
}

Write-Host "Smoke checks passed."
