// Container App for JarvisAI API

@minLength(2)
@maxLength(32)
@description('Name of the Container App')
param name string

@description('Location for the Container App')
param location string = resourceGroup().location

@description('Resource tags')
param tags object = {}

@description('Container Apps Environment ID')
param containerAppsEnvironmentId string

@description('Container Registry name')
param containerRegistryName string

@description('Container image (will be updated by CI/CD)')
param containerImage string

@description('Application Insights connection string')
@secure()
param applicationInsightsConnectionString string

@description('Database connection string')
@secure()
param databaseUrl string

@description('Redis connection string')
@secure()
param redisUrl string

@description('Key Vault endpoint')
param keyVaultEndpoint string

@description('Minimum number of replicas')
@minValue(0)
@maxValue(30)
param minReplicas int = 1

@description('Maximum number of replicas')
@minValue(1)
@maxValue(30)
param maxReplicas int = 10

@description('CPU cores allocated to container')
param cpu string = '1.0'

@description('Memory allocated to container')
param memory string = '2.0Gi'

// Generate a secret key (in production, use Key Vault reference or pass as parameter)
var secretKey = uniqueString(resourceGroup().id, name)

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' existing = {
  name: containerRegistryName
}

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: containerAppsEnvironmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
        corsPolicy: {
          allowedOrigins: [
            '*'
          ]
          allowedMethods: [
            'GET'
            'POST'
            'PUT'
            'DELETE'
            'OPTIONS'
            'PATCH'
          ]
          allowedHeaders: [
            '*'
          ]
          maxAge: 3600
        }
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: 'system'
        }
      ]
      secrets: [
        {
          name: 'database-url'
          value: databaseUrl
        }
        {
          name: 'redis-url'
          value: redisUrl
        }
        {
          name: 'secret-key'
          value: secretKey
        }
        {
          name: 'application-insights-connection-string'
          value: applicationInsightsConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'jarvisai-api'
          image: containerImage
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: [
            {
              name: 'DATABASE_URL'
              secretRef: 'database-url'
            }
            {
              name: 'REDIS_URL'
              secretRef: 'redis-url'
            }
            {
              name: 'SECRET_KEY'
              secretRef: 'secret-key'
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              secretRef: 'application-insights-connection-string'
            }
            {
              name: 'AZURE_KEY_VAULT_ENDPOINT'
              value: keyVaultEndpoint
            }
            {
              name: 'ENVIRONMENT'
              value: 'production'
            }
            {
              name: 'LOG_LEVEL'
              value: 'INFO'
            }
            {
              name: 'WORKERS'
              value: '4'
            }
            {
              name: 'PORT'
              value: '8000'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 30
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 5
              timeoutSeconds: 3
              failureThreshold: 3
            }
            {
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 0
              periodSeconds: 3
              timeoutSeconds: 3
              failureThreshold: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
          {
            name: 'cpu-scaling'
            custom: {
              type: 'cpu'
              metadata: {
                type: 'Utilization'
                value: '70'
              }
            }
          }
          {
            name: 'memory-scaling'
            custom: {
              type: 'memory'
              metadata: {
                type: 'Utilization'
                value: '80'
              }
            }
          }
        ]
      }
    }
  }
}

output id string = containerApp.id
output name string = containerApp.name
output url string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output fqdn string = containerApp.properties.configuration.ingress.fqdn
output identityPrincipalId string = containerApp.identity.principalId
