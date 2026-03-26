# JarvisAI Azure Deployment Guide

This guide provides step-by-step instructions for deploying JarvisAI to Azure Container Apps using Azure Developer CLI (azd) and Bicep infrastructure as code.

## 📋 Prerequisites

### Required Tools

- **Azure CLI** (v2.50.0 or later)

  ```bash
  az --version
  # Install: https://docs.microsoft.com/cli/azure/install-azure-cli
  ```

- **Azure Developer CLI** (azd)

  ```bash
  azd version
  # Install: https://aka.ms/azure-dev/install
  ```

- **Docker Desktop** (for local testing)
  ```bash
  docker --version
  # Install: https://www.docker.com/products/docker-desktop
  ```

### Azure Subscription

- Active Azure subscription with Contributor or Owner permissions
- Sufficient quota for:
  - Container Apps
  - PostgreSQL Flexible Server
  - Azure Cache for Redis (Basic tier)
  - Container Registry
  - Application Insights

## 🚀 Quick Start Deployment

### 1. Clone and Setup

```bash
# Navigate to project directory
cd JarvisAI

# Copy environment template
cp .env.example .env

# Edit .env and set required values
# At minimum, set: SECRET_KEY, DATABASE_ADMIN_PASSWORD
```

### 2. Login to Azure

```bash
# Login to Azure
az login

# Set your subscription (if you have multiple)
az account set --subscription "Your-Subscription-Name"

# Verify your subscription
az account show
```

### 3. Initialize Azure Developer CLI

```bash
# Initialize azd (first time only)
azd init

# When prompted:
# - Environment name: dev (or your preferred environment name)
# - Azure location: eastus (or your preferred region)
```

### 4. Set Environment Variables

```bash
# Set database credentials securely
azd env set DATABASE_ADMIN_LOGIN jarvisadmin
azd env set DATABASE_ADMIN_PASSWORD "YourSecurePassword123!"

# Optional: Configure other settings
azd env set AZURE_LOCATION eastus
```

### 5. Deploy Infrastructure and Application

```bash
# Provision infrastructure and deploy application
azd up

# This command will:
# 1. Provision all Azure resources (5-10 minutes)
# 2. Build and push Docker image to ACR
# 3. Deploy container to Container Apps
# 4. Configure all connections and secrets
```

### 6. Verify Deployment

```bash
# Show deployment endpoints
azd show

# Test the health endpoint
curl https://YOUR-CONTAINER-APP-URL.azurecontainerapps.io/health

# View application logs
azd logs --follow
```

## 📦 Manual Infrastructure Deployment

If you prefer to deploy infrastructure separately:

### 1. Provision Infrastructure Only

```bash
# Provision Azure resources without deploying app
azd provision
```

### 2. Build and Deploy Application

```bash
# Build and deploy the application image
azd deploy
```

## 🔧 Configuration Details

### Environment Variables

The deployment automatically configures these environment variables:

| Variable                                | Source         | Description                  |
| --------------------------------------- | -------------- | ---------------------------- |
| `DATABASE_URL`                          | Key Vault      | PostgreSQL connection string |
| `REDIS_URL`                             | Key Vault      | Redis connection string      |
| `SECRET_KEY`                            | Auto-generated | Application secret key       |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Monitoring     | Application Insights         |
| `AZURE_KEY_VAULT_ENDPOINT`              | Infrastructure | Key Vault endpoint           |

### Scaling Configuration

**Default Auto-scaling Rules:**

- Min replicas: 1
- Max replicas: 10
- CPU threshold: 70%
- Memory threshold: 80%
- HTTP concurrent requests: 100

**To customize scaling:**
Edit `infra/containerApp.bicep` → `scale` section

### Resource Sizing

**Development (Current):**

- Container App: 1.0 CPU, 2.0 Gi memory
- PostgreSQL: Burstable B1ms (1 vCore, 2 GB RAM)
- Redis: Basic C0 (250 MB cache)

**Production Recommendations:**

```bicep
# In containerApp.bicep
param cpu string = '2.0'
param memory string = '4.0Gi'

# In database.bicep
param skuName string = 'Standard_D2ds_v4'
param tier string = 'GeneralPurpose'

# In redis.bicep
param skuName string = 'Standard'
param skuCapacity int = 1
```

## 🗄️ Database Setup

### Initial Database Migration

```bash
# Connect to the deployed database
# Get connection details from Azure Portal or:
azd env get-values | grep DATABASE

# Run migrations (if using Alembic)
# SSH into container or use Azure Container Apps exec:
az containerapp exec \
  --name YOUR-CONTAINER-APP-NAME \
  --resource-group rg-jarvisai-dev \
  --command "alembic upgrade head"
```

### Local Database Testing

```bash
# Use PostgreSQL locally with Docker
docker run --name postgres-local \
  -e POSTGRES_PASSWORD=localpass \
  -e POSTGRES_DB=jarvisai \
  -p 5432:5432 \
  -d postgres:16

# Update .env
DATABASE_URL=postgresql://postgres:localpass@localhost:5432/jarvisai
```

## 🔐 Security Best Practices

### 1. Key Vault Access

All secrets are stored in Azure Key Vault and accessed via managed identity:

```bash
# View Key Vault secrets
az keyvault secret list --vault-name YOUR-KEYVAULT-NAME

# Update a secret
az keyvault secret set \
  --vault-name YOUR-KEYVAULT-NAME \
  --name DATABASE-URL \
  --value "new-connection-string"
```

### 2. Network Security

**Current Configuration (Development):**

- Public endpoints enabled for all services
- Firewall allows Azure services

**Production Recommendations:**

- Enable Virtual Network integration
- Use Private Endpoints for database and Redis
- Configure NSG rules
- Enable Azure DDoS Protection

```bicep
# In database.bicep - switch to private endpoint
resource postgresqlServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-12-01-preview' = {
  properties: {
    network: {
      publicNetworkAccess: 'Disabled'
      delegatedSubnetResourceId: subnetId
      privateDnsZoneArmResourceId: privateDnsZoneId
    }
  }
}
```

### 3. Authentication

Update CORS and authentication settings in `api.py`:

```python
# Configure allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Monitoring and Logging

### Application Insights

```bash
# View application metrics in Azure Portal
az portal open --resource-group rg-jarvisai-dev

# Query logs using KQL
az monitor app-insights query \
  --app YOUR-APP-INSIGHTS-NAME \
  --analytics-query "requests | where timestamp > ago(1h) | summarize count() by resultCode"
```

### Container Logs

```bash
# Stream container logs
azd logs --follow

# Or use Azure CLI
az containerapp logs show \
  --name YOUR-CONTAINER-APP-NAME \
  --resource-group rg-jarvisai-dev \
  --follow
```

### Performance Monitoring

Access Application Insights in Azure Portal:

1. Navigate to Application Insights resource
2. View:
   - Live Metrics
   - Application Map
   - Performance
   - Failures
   - Availability

## 🔄 CI/CD Integration

### GitHub Actions Setup

```bash
# Configure GitHub Actions for automated deployment
azd pipeline config

# Follow prompts to:
# 1. Select GitHub
# 2. Authorize Azure access
# 3. Choose repository
```

This creates `.github/workflows/azure-dev.yml` with:

- Automated build on push
- Infrastructure validation
- Container image build and push
- Deployment to Container Apps

### Manual CI/CD Commands

```bash
# Build container locally
docker build -t jarvisai:local .

# Test locally
docker run -p 8000:8000 --env-file .env jarvisai:local

# Push to ACR
az acr login --name YOUR-ACR-NAME
docker tag jarvisai:local YOUR-ACR-NAME.azurecr.io/jarvisai:latest
docker push YOUR-ACR-NAME.azurecr.io/jarvisai:latest

# Update Container App
azd deploy
```

## 🧪 Testing Deployment

### Health Check

```bash
# Basic health check
curl https://YOUR-APP-URL.azurecontainerapps.io/health

# Expected response:
{"status": "healthy", "timestamp": "2026-03-05T..."}
```

### API Endpoints

```bash
# Test models endpoint
curl https://YOUR-APP-URL.azurecontainerapps.io/models

# Test with authentication
curl -H "Authorization: Bearer YOUR-TOKEN" \
  https://YOUR-APP-URL.azurecontainerapps.io/models
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils  # Linux
brew install apache2  # macOS

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 https://YOUR-APP-URL.azurecontainerapps.io/health
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Container App Not Starting

```bash
# Check container logs
azd logs --follow

# Check revision status
az containerapp revision list \
  --name YOUR-CONTAINER-APP-NAME \
  --resource-group rg-jarvisai-dev

# Common causes:
# - Missing environment variables
# - Database connection failures
# - Port binding issues (ensure app listens on 0.0.0.0:8000)
```

#### 2. Database Connection Failed

```bash
# Test database connectivity
az postgres flexible-server connect \
  --name YOUR-POSTGRES-NAME \
  --admin-user jarvisadmin \
  --database jarvisai

# Check firewall rules
az postgres flexible-server firewall-rule list \
  --name YOUR-POSTGRES-NAME \
  --resource-group rg-jarvisai-dev
```

#### 3. High Memory Usage

```bash
# Check container metrics
az containerapp show \
  --name YOUR-CONTAINER-APP-NAME \
  --resource-group rg-jarvisai-dev \
  --query "properties.template.containers[0].resources"

# Increase memory allocation in containerApp.bicep:
param memory string = '4.0Gi'

# Redeploy
azd deploy
```

#### 4. Slow Performance

```bash
# Check Application Insights for bottlenecks
# Verify database query performance
# Review Redis cache hit rate

# Scale horizontally
az containerapp update \
  --name YOUR-CONTAINER-APP-NAME \
  --resource-group rg-jarvisai-dev \
  --min-replicas 2 \
  --max-replicas 20
```

## 💰 Cost Estimation

### Development Environment (Monthly)

| Service              | Tier                     | Estimated Cost     |
| -------------------- | ------------------------ | ------------------ |
| Container Apps       | Consumption (1 instance) | $20-40             |
| PostgreSQL           | Burstable B1ms           | $15-25             |
| Redis                | Basic C0                 | $16                |
| Container Registry   | Basic                    | $5                 |
| Application Insights | Pay-as-you-go            | $5-15              |
| **Total**            |                          | **~$61-101/month** |

### Production Environment (Monthly)

| Service              | Tier                         | Estimated Cost      |
| -------------------- | ---------------------------- | ------------------- |
| Container Apps       | Consumption (2-10 instances) | $100-200            |
| PostgreSQL           | GeneralPurpose D2ds_v4       | $150-200            |
| Redis                | Standard C1                  | $75                 |
| Container Registry   | Standard                     | $20                 |
| Application Insights | Pay-as-you-go                | $50-100             |
| **Total**            |                              | **~$395-595/month** |

_Costs are estimates and vary by region and usage._

## 🔄 Updates and Maintenance

### Update Application Code

```bash
# Make code changes
# Commit to git

# Deploy updates
azd deploy

# Or redeploy everything
azd up
```

### Update Infrastructure

```bash
# Edit Bicep files in infra/
# For example, change database tier in database.bicep

# Apply infrastructure changes
azd provision

# Or full redeployment
azd up
```

### Rollback Deployment

```bash
# List revisions
az containerapp revision list \
  --name YOUR-CONTAINER-APP-NAME \
  --resource-group rg-jarvisai-dev

# Activate previous revision
az containerapp revision activate \
  --name YOUR-CONTAINER-APP-NAME \
  --resource-group rg-jarvisai-dev \
  --revision REVISION-NAME
```

## 🗑️ Cleanup

### Delete All Resources

```bash
# Remove all Azure resources
azd down

# With resource group deletion confirmation
azd down --force --purge
```

### Delete Specific Resources

```bash
# Delete just the container app
az containerapp delete \
  --name YOUR-CONTAINER-APP-NAME \
  --resource-group rg-jarvisai-dev

# Delete resource group (removes everything)
az group delete --name rg-jarvisai-dev
```

## 📚 Additional Resources

- [Azure Container Apps Documentation](https://docs.microsoft.com/azure/container-apps/)
- [Azure Developer CLI Documentation](https://learn.microsoft.com/azure/developer/azure-developer-cli/)
- [PostgreSQL Flexible Server](https://docs.microsoft.com/azure/postgresql/flexible-server/)
- [Azure Cache for Redis](https://docs.microsoft.com/azure/azure-cache-for-redis/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🆘 Support

For issues and questions:

- GitHub Issues: [JarvisAI Issues](https://github.com/MastaTrill/JarvisAI/issues)
- Azure Support: [Azure Support Portal](https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade)

## 📝 License

JarvisAI is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Last Updated:** March 5, 2026
**Version:** 1.0.0
