// Azure Key Vault for secrets management

@minLength(3)
@maxLength(24)
@description('Name of the Key Vault')
param name string

@description('Location for Key Vault')
param location string = resourceGroup().location

@description('Resource tags')
param tags object = {}

@description('Public network access setting')
@allowed([
  'Enabled'
  'Disabled'
])
param publicNetworkAccess string = 'Enabled'

@description('Enable RBAC authorization for Key Vault')
param enableRbacAuthorization bool = true

@description('Enable soft delete')
param enableSoftDelete bool = true

@description('Soft delete retention in days')
@minValue(7)
@maxValue(90)
param softDeleteRetentionInDays int = 7

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: enableRbacAuthorization
    enableSoftDelete: enableSoftDelete
    softDeleteRetentionInDays: softDeleteRetentionInDays
    enablePurgeProtection: true // Required: cannot be set to false once enabled
    publicNetworkAccess: publicNetworkAccess
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

output id string = keyVault.id
output name string = keyVault.name
output endpoint string = keyVault.properties.vaultUri
