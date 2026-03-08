// Store secrets in Key Vault

@description('Name of the Key Vault')
param keyVaultName string

@description('Array of secrets to store')
param secrets array = []

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource keyVaultSecrets 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = [
  for secret in secrets: {
    parent: keyVault
    name: secret.name
    properties: {
      value: secret.value
    }
  }
]

output secretIds array = [for (secret, i) in secrets: keyVaultSecrets[i].id]
