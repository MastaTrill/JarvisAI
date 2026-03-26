// Main Bicep orchestrator for JarvisAI deployment
targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment (dev, staging, prod)')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Resource tags for all resources')
param tags object = {}

@description('Database administrator login name')
@secure()
param databaseAdminLogin string = 'jarvisadmin'

@description('Database administrator password')
@secure()
param databaseAdminPassword string

@description('Flag to enable/disable public network access to Key Vault')
param keyVaultPublicNetworkAccess bool = true

// Generate unique suffix for resource names
var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))
var deploymentToken = toLower(uniqueString(deployment().name))

// Resource group
resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: '${abbrs.resourcesResourceGroups}jarvisai-${environmentName}'
  location: location
  tags: union(tags, {
    'azd-env-name': environmentName
    'app': 'jarvisai'
  })
}

// Container Registry
module containerRegistry './containerRegistry.bicep' = {
  name: 'container-registry-${deploymentToken}'
  scope: rg
  params: {
    name: '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    tags: tags
  }
}

// Log Analytics Workspace & Application Insights
module monitoring './monitoring.bicep' = {
  name: 'monitoring-${deploymentToken}'
  scope: rg
  params: {
    location: location
    tags: tags
    logAnalyticsName: '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
    applicationInsightsName: '${abbrs.insightsComponents}${resourceToken}'
  }
}

// Key Vault
module keyVault './keyvault.bicep' = {
  name: 'keyvault-${deploymentToken}'
  scope: rg
  params: {
    name: '${abbrs.keyVaultVaults}${resourceToken}'
    location: location
    tags: tags
    publicNetworkAccess: keyVaultPublicNetworkAccess ? 'Enabled' : 'Disabled'
  }
}

// PostgreSQL Flexible Server
module database './database.bicep' = {
  name: 'database-${deploymentToken}'
  scope: rg
  params: {
    name: '${abbrs.dBforPostgreSQLServers}${resourceToken}'
    location: location
    tags: tags
    administratorLogin: databaseAdminLogin
    administratorPassword: databaseAdminPassword
    databaseName: 'jarvisai'
    skuName: 'Standard_B1ms'
    tier: 'Burstable'
    storageSizeGB: 32
  }
}

// Azure Cache for Redis
module redis './redis.bicep' = {
  name: 'redis-${deploymentToken}'
  scope: rg
  params: {
    name: '${abbrs.cacheRedis}${resourceToken}'
    location: location
    tags: tags
    skuName: 'Basic'
    skuFamily: 'C'
    skuCapacity: 0
  }
}

// Container Apps Environment
module containerAppsEnvironment './containerAppsEnvironment.bicep' = {
  name: 'container-apps-env-${deploymentToken}'
  scope: rg
  params: {
    name: '${abbrs.appManagedEnvironments}${resourceToken}'
    location: location
    tags: tags
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
  }
}

// Container App
module containerApp './containerApp.bicep' = {
  name: 'container-app-${deploymentToken}'
  scope: rg
  params: {
    name: '${abbrs.appContainerApps}jarvisai-${resourceToken}'
    location: location
    tags: union(tags, {
      'azd-service-name': 'api'
    })
    containerAppsEnvironmentId: containerAppsEnvironment.outputs.id
    containerRegistryName: containerRegistry.outputs.name
    applicationInsightsConnectionString: monitoring.outputs.applicationInsightsConnectionString
    databaseUrl: database.outputs.connectionString
    redisUrl: redis.outputs.connectionString
    keyVaultEndpoint: keyVault.outputs.endpoint
    containerImage: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' // Placeholder, will be updated on first deploy
  }
}

// Store secrets in Key Vault
module secrets './keyVaultSecrets.bicep' = {
  name: 'keyvault-secrets-${deploymentToken}'
  scope: rg
  params: {
    keyVaultName: keyVault.outputs.name
    secrets: [
      {
        name: 'DATABASE-URL'
        value: database.outputs.connectionString
      }
      {
        name: 'REDIS-URL'
        value: redis.outputs.connectionString
      }
      {
        name: 'DATABASE-ADMIN-PASSWORD'
        value: databaseAdminPassword
      }
    ]
  }
}

// Grant Container App access to Key Vault
module keyVaultAccess './keyVaultAccess.bicep' = {
  name: 'keyvault-access-${deploymentToken}'
  scope: rg
  params: {
    keyVaultName: keyVault.outputs.name
    principalId: containerApp.outputs.identityPrincipalId
  }
}

// Outputs
output AZURE_LOCATION string = location
output AZURE_RESOURCE_GROUP string = rg.name
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.outputs.loginServer
output AZURE_CONTAINER_REGISTRY_NAME string = containerRegistry.outputs.name
output AZURE_CONTAINER_APPS_ENVIRONMENT_ID string = containerAppsEnvironment.outputs.id
output AZURE_CONTAINER_APP_NAME string = containerApp.outputs.name
output AZURE_CONTAINER_APP_URL string = containerApp.outputs.url
output AZURE_KEY_VAULT_ENDPOINT string = keyVault.outputs.endpoint
output AZURE_KEY_VAULT_NAME string = keyVault.outputs.name
output AZURE_APPLICATION_INSIGHTS_NAME string = monitoring.outputs.applicationInsightsName
output AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING string = monitoring.outputs.applicationInsightsConnectionString
output AZURE_DATABASE_HOST string = database.outputs.host
output AZURE_DATABASE_NAME string = database.outputs.databaseName
output AZURE_REDIS_HOST string = redis.outputs.host
