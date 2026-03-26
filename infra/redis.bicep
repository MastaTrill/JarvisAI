// Azure Cache for Redis

@minLength(1)
@maxLength(63)
@description('Name of the Redis cache')
param name string

@description('Location for Redis cache')
param location string = resourceGroup().location

@description('Resource tags')
param tags object = {}

@description('SKU name')
@allowed([
  'Basic'
  'Standard'
  'Premium'
])
param skuName string = 'Basic'

@description('SKU family')
@allowed([
  'C'
  'P'
])
param skuFamily string = 'C'

@description('SKU capacity (0-6 for Basic/Standard, 1-5 for Premium)')
@minValue(0)
@maxValue(6)
param skuCapacity int = 0

@description('Enable non-SSL port (6379)')
param enableNonSslPort bool = false

@description('Minimum TLS version')
@allowed([
  '1.0'
  '1.1'
  '1.2'
])
param minimumTlsVersion string = '1.2'

@description('Public network access')
@allowed([
  'Enabled'
  'Disabled'
])
param publicNetworkAccess string = 'Enabled'

resource redis 'Microsoft.Cache/redis@2023-08-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      name: skuName
      family: skuFamily
      capacity: skuCapacity
    }
    enableNonSslPort: enableNonSslPort
    minimumTlsVersion: minimumTlsVersion
    publicNetworkAccess: publicNetworkAccess
    redisConfiguration: {
      'maxmemory-policy': 'allkeys-lru'
    }
  }
}

output id string = redis.id
output name string = redis.name
output host string = redis.properties.hostName
output port int = redis.properties.port
output sslPort int = redis.properties.sslPort
output connectionString string = '${redis.properties.hostName}:${redis.properties.sslPort},password=${redis.listKeys().primaryKey},ssl=True,abortConnect=False'
output primaryKey string = redis.listKeys().primaryKey
