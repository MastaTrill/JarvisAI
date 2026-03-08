#!/bin/bash
# Azure Deployment Validation Script
# Run this before deploying to Azure to check if everything is configured correctly

echo ""
echo "========================================"
echo "JarvisAI Azure Deployment Validation"
echo "========================================"
echo ""

errors=0
warnings=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Check Azure CLI
echo -e "${YELLOW}Checking Azure CLI...${NC}"
if command -v az &> /dev/null; then
    azVersion=$(az version --query '"azure-cli"' -o tsv 2>/dev/null)
    echo -e "  ${GREEN}✓ Azure CLI installed: $azVersion${NC}"
else
    echo -e "  ${RED}✗ Azure CLI not found${NC}"
    echo -e "    ${GRAY}Install from: https://docs.microsoft.com/cli/azure/install-azure-cli${NC}"
    ((errors++))
fi

# Check Azure Developer CLI
echo -e "\n${YELLOW}Checking Azure Developer CLI...${NC}"
if command -v azd &> /dev/null; then
    echo -e "  ${GREEN}✓ Azure Developer CLI installed${NC}"
else
    echo -e "  ${RED}✗ Azure Developer CLI not found${NC}"
    echo -e "    ${GRAY}Install from: https://aka.ms/azure-dev/install${NC}"
    ((errors++))
fi

# Check Docker
echo -e "\n${YELLOW}Checking Docker...${NC}"
if command -v docker &> /dev/null; then
    dockerVersion=$(docker --version 2>/dev/null)
    echo -e "  ${GREEN}✓ Docker installed: $dockerVersion${NC}"
    
    # Check if Docker is running
    if docker ps &> /dev/null; then
        echo -e "  ${GREEN}✓ Docker daemon is running${NC}"
    else
        echo -e "  ${YELLOW}⚠ Docker daemon is not running${NC}"
        echo -e "    ${GRAY}Start Docker service${NC}"
        ((warnings++))
    fi
else
    echo -e "  ${RED}✗ Docker not found${NC}"
    echo -e "    ${GRAY}Install from: https://www.docker.com/products/docker-desktop${NC}"
    ((errors++))
fi

# Check required files
echo -e "\n${YELLOW}Checking infrastructure files...${NC}"
requiredFiles=(
    "azure.yaml"
    "Dockerfile"
    "requirements.txt"
    ".env.example"
    "infra/main.bicep"
    "infra/containerApp.bicep"
    "infra/database.bicep"
    "infra/redis.bicep"
    "infra/monitoring.bicep"
    "infra/keyvault.bicep"
    "infra/containerRegistry.bicep"
    "infra/abbreviations.json"
)

for file in "${requiredFiles[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓ $file${NC}"
    else
        echo -e "  ${RED}✗ $file missing${NC}"
        ((errors++))
    fi
done

# Check .env file
echo -e "\n${YELLOW}Checking environment configuration...${NC}"
if [ -f ".env" ]; then
    echo -e "  ${GREEN}✓ .env file exists${NC}"
    
    # Check for required variables
    requiredVars=("DATABASE_ADMIN_PASSWORD" "SECRET_KEY")
    for var in "${requiredVars[@]}"; do
        if grep -q "^$var=\S" .env 2>/dev/null; then
            echo -e "    ${GREEN}✓ $var is set${NC}"
        else
            echo -e "    ${YELLOW}⚠ $var not set or empty${NC}"
            ((warnings++))
        fi
    done
else
    echo -e "  ${YELLOW}⚠ .env file not found${NC}"
    echo -e "    ${GRAY}Copy .env.example to .env and configure${NC}"
    ((warnings++))
fi

# Check Azure login
echo -e "\n${YELLOW}Checking Azure authentication...${NC}"
if az account show &> /dev/null; then
    accountName=$(az account show --query "user.name" -o tsv 2>/dev/null)
    subscriptionName=$(az account show --query "name" -o tsv 2>/dev/null)
    echo -e "  ${GREEN}✓ Logged in to Azure${NC}"
    echo -e "    ${GRAY}Account: $accountName${NC}"
    echo -e "    ${GRAY}Subscription: $subscriptionName${NC}"
else
    echo -e "  ${YELLOW}⚠ Not logged in to Azure${NC}"
    echo -e "    ${GRAY}Run: az login${NC}"
    ((warnings++))
fi

# Check Bicep CLI
echo -e "\n${YELLOW}Checking Bicep CLI...${NC}"
if az bicep version &> /dev/null; then
    echo -e "  ${GREEN}✓ Bicep CLI available${NC}"
else
    echo -e "  ${YELLOW}⚠ Bicep CLI not found, attempting auto-install...${NC}"
    if az bicep install &> /dev/null; then
        echo -e "  ${GREEN}✓ Bicep CLI installed successfully${NC}"
    else
        echo -e "  ${RED}✗ Failed to install Bicep CLI${NC}"
        ((errors++))
    fi
fi

# Check Python
echo -e "\n${YELLOW}Checking Python environment...${NC}"
if command -v python3 &> /dev/null; then
    pythonVersion=$(python3 --version 2>/dev/null)
    echo -e "  ${GREEN}✓ Python installed: $pythonVersion${NC}"
elif command -v python &> /dev/null; then
    pythonVersion=$(python --version 2>/dev/null)
    echo -e "  ${GREEN}✓ Python installed: $pythonVersion${NC}"
else
    echo -e "  ${YELLOW}⚠ Python not found${NC}"
    ((warnings++))
fi

# Summary
echo ""
echo "========================================"
echo "Validation Summary"
echo "========================================"

if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
    echo -e "\n${GREEN}✓ All checks passed! Ready to deploy.${NC}"
    echo -e "\n${CYAN}Next steps:${NC}"
    echo -e "  ${NC}1. Review .env configuration${NC}"
    echo -e "  ${NC}2. Run: azd init${NC}"
    echo -e "  ${NC}3. Run: azd up${NC}"
    echo -e "\n${GRAY}See DEPLOYMENT.md for detailed instructions.${NC}"
    exit 0
elif [ $errors -eq 0 ]; then
    echo -e "\n${YELLOW}⚠ Validation completed with $warnings warning(s).${NC}"
    echo -e "  ${GRAY}You can proceed but should address warnings first.${NC}"
    echo -e "\n${GRAY}See DEPLOYMENT.md for help.${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Validation failed with $errors error(s) and $warnings warning(s).${NC}"
    echo -e "  ${GRAY}Please fix the errors before deploying.${NC}"
    echo -e "\n${GRAY}See DEPLOYMENT.md for installation instructions.${NC}"
    exit 1
fi
