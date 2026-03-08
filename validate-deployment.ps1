# Azure Deployment Validation Script
# Run this before deploying to Azure to check if everything is configured correctly

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "JarvisAI Azure Deployment Validation" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$errors = 0
$warnings = 0

# Check Azure CLI
Write-Host "Checking Azure CLI..." -ForegroundColor Yellow
try {
    $azVersion = az version --query '"azure-cli"' -o tsv 2>$null
    if ($azVersion) {
        Write-Host "  ✓ Azure CLI installed: $azVersion" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ Azure CLI not found or not working" -ForegroundColor Red
        Write-Host "    Install from: https://docs.microsoft.com/cli/azure/install-azure-cli" -ForegroundColor Gray
        $errors++
    }
}
catch {
    Write-Host "  ✗ Azure CLI not found" -ForegroundColor Red
    $errors++
}

# Check Azure Developer CLI
Write-Host "`nChecking Azure Developer CLI..." -ForegroundColor Yellow
try {
    $azdVersion = azd version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Azure Developer CLI installed" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ Azure Developer CLI not found" -ForegroundColor Red
        Write-Host "    Install from: https://aka.ms/azure-dev/install" -ForegroundColor Gray
        $errors++
    }
}
catch {
    Write-Host "  ✗ Azure Developer CLI not found" -ForegroundColor Red
    $errors++
}

# Check Docker
Write-Host "`nChecking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Docker installed: $dockerVersion" -ForegroundColor Green
        
        # Check if Docker is running
        docker ps 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Docker daemon is running" -ForegroundColor Green
        }
        else {
            Write-Host "  ⚠ Docker daemon is not running" -ForegroundColor Yellow
            Write-Host "    Start Docker Desktop" -ForegroundColor Gray
            $warnings++
        }
    }
    else {
        Write-Host "  ✗ Docker not found" -ForegroundColor Red
        Write-Host "    Install from: https://www.docker.com/products/docker-desktop" -ForegroundColor Gray
        $errors++
    }
}
catch {
    Write-Host "  ✗ Docker not found" -ForegroundColor Red
    $errors++
}

# Check required files
Write-Host "`nChecking infrastructure files..." -ForegroundColor Yellow
$requiredFiles = @(
    "azure.yaml",
    "Dockerfile",
    "requirements.txt",
    ".env.example",
    "infra\main.bicep",
    "infra\containerApp.bicep",
    "infra\database.bicep",
    "infra\redis.bicep",
    "infra\monitoring.bicep",
    "infra\keyvault.bicep",
    "infra\containerRegistry.bicep",
    "infra\abbreviations.json"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ $file missing" -ForegroundColor Red
        $missingFiles += $file
        $errors++
    }
}

# Check .env file
Write-Host "`nChecking environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  ✓ .env file exists" -ForegroundColor Green
    
    # Check for required variables
    $envContent = Get-Content .env -Raw
    $requiredVars = @("DATABASE_ADMIN_PASSWORD", "SECRET_KEY")
    
    foreach ($var in $requiredVars) {
        if ($envContent -match "$var=\S+") {
            Write-Host "    ✓ $var is set" -ForegroundColor Green
        }
        else {
            Write-Host "    ⚠ $var not set or empty" -ForegroundColor Yellow
            $warnings++
        }
    }
}
else {
    Write-Host "  ⚠ .env file not found" -ForegroundColor Yellow
    Write-Host "    Copy .env.example to .env and configure" -ForegroundColor Gray
    $warnings++
}

# Check Azure login
Write-Host "`nChecking Azure authentication..." -ForegroundColor Yellow
try {
    $account = az account show 2>$null | ConvertFrom-Json
    if ($account) {
        Write-Host "  ✓ Logged in to Azure" -ForegroundColor Green
        Write-Host "    Account: $($account.user.name)" -ForegroundColor Gray
        Write-Host "    Subscription: $($account.name)" -ForegroundColor Gray
    }
    else {
        Write-Host "  ⚠ Not logged in to Azure" -ForegroundColor Yellow
        Write-Host "    Run: az login" -ForegroundColor Gray
        $warnings++
    }
}
catch {
    Write-Host "  ⚠ Could not verify Azure login" -ForegroundColor Yellow
    Write-Host "    Run: az login" -ForegroundColor Gray
    $warnings++
}

# Check Bicep CLI
Write-Host "`nChecking Bicep CLI..." -ForegroundColor Yellow
try {
    $bicepVersion = az bicep version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Bicep CLI available" -ForegroundColor Green
    }
    else {
        Write-Host "  ⚠ Bicep CLI not found, attempting auto-install..." -ForegroundColor Yellow
        az bicep install 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Bicep CLI installed successfully" -ForegroundColor Green
        }
        else {
            Write-Host "  ✗ Failed to install Bicep CLI" -ForegroundColor Red
            $errors++
        }
    }
}
catch {
    Write-Host "  ⚠ Could not check Bicep CLI" -ForegroundColor Yellow
    $warnings++
}

# Check Python
Write-Host "`nChecking Python environment..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Python installed: $pythonVersion" -ForegroundColor Green
    }
    else {
        Write-Host "  ⚠ Python not found" -ForegroundColor Yellow
        $warnings++
    }
}
catch {
    Write-Host "  ⚠ Python not found" -ForegroundColor Yellow
    $warnings++
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Validation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "`n✓ All checks passed! Ready to deploy." -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. Review .env configuration" -ForegroundColor White
    Write-Host "  2. Run: azd init" -ForegroundColor White
    Write-Host "  3. Run: azd up" -ForegroundColor White
    Write-Host "`nSee DEPLOYMENT.md for detailed instructions." -ForegroundColor Gray
    exit 0
}
elseif ($errors -eq 0) {
    Write-Host "`n⚠ Validation completed with $warnings warning(s)." -ForegroundColor Yellow
    Write-Host "  You can proceed but should address warnings first." -ForegroundColor Gray
    Write-Host "`nSee DEPLOYMENT.md for help." -ForegroundColor Gray
    exit 0
}
else {
    Write-Host "`n✗ Validation failed with $errors error(s) and $warnings warning(s)." -ForegroundColor Red
    Write-Host "  Please fix the errors before deploying." -ForegroundColor Gray
    Write-Host "`nSee DEPLOYMENT.md for installation instructions." -ForegroundColor Gray
    exit 1
}
