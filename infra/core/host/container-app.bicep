param name string
param location string = resourceGroup().location
param containerAppsEnvironmentId string
param containerRegistryName string
param imageName string
param targetPort int = 8000
param env array = []
param containerCpuCoreCount string = '1.0'
param containerMemory string = '2Gi'

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: containerRegistryName
}

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: targetPort
        transport: 'http'
        allowInsecure: false
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'jarvis-api'
          image: '${containerRegistry.properties.loginServer}/${imageName}'
          resources: {
            cpu: json(containerCpuCoreCount)
            memory: containerMemory
          }
          env: env
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
        rules: [
          {
            name: 'http-rule'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
}

// Grant the container app identity access to pull from ACR
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(resourceGroup().id, containerApp.id, 'acrpull')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output id string = containerApp.id
output name string = containerApp.name
output url string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
