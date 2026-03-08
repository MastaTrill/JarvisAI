// Azure Database for PostgreSQL Flexible Server

@minLength(3)
@maxLength(63)
@description('Name of the PostgreSQL server')
param name string

@description('Location for the database')
param location string = resourceGroup().location

@description('Resource tags')
param tags object = {}

@description('Administrator login name')
param administratorLogin string

@description('Administrator password')
@secure()
param administratorPassword string

@description('Name of the database')
param databaseName string = 'jarvisai'

@description('PostgreSQL version')
@allowed([
  '11'
  '12'
  '13'
  '14'
  '15'
  '16'
])
param postgresqlVersion string = '16'

@description('SKU name (e.g., Standard_B1ms)')
param skuName string = 'Standard_B1ms'

@description('SKU tier')
@allowed([
  'Burstable'
  'GeneralPurpose'
  'MemoryOptimized'
])
param tier string = 'Burstable'

@description('Storage size in GB')
@minValue(32)
@maxValue(16384)
param storageSizeGB int = 32

@description('Backup retention days')
@minValue(7)
@maxValue(35)
param backupRetentionDays int = 7

@description('Enable point-in-time restore')
param geoRedundantBackup bool = false

@description('Enable high availability')
param highAvailability bool = false

resource postgresqlServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-12-01-preview' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: skuName
    tier: tier
  }
  properties: {
    version: postgresqlVersion
    administratorLogin: administratorLogin
    administratorLoginPassword: administratorPassword
    storage: {
      storageSizeGB: storageSizeGB
      autoGrow: 'Enabled'
    }
    backup: {
      backupRetentionDays: backupRetentionDays
      geoRedundantBackup: geoRedundantBackup ? 'Enabled' : 'Disabled'
    }
    highAvailability: {
      mode: highAvailability ? 'ZoneRedundant' : 'Disabled'
    }
    network: {
      publicNetworkAccess: 'Enabled'
    }
  }
}

// Firewall rule to allow Azure services
resource firewallRule 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-12-01-preview' = {
  parent: postgresqlServer
  name: 'AllowAllAzureServicesAndResourcesWithinAzureIps'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// Create the database
resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-12-01-preview' = {
  parent: postgresqlServer
  name: databaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// PostgreSQL configurations for better performance
resource postgreSqlConfig 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2023-12-01-preview' = {
  parent: postgresqlServer
  name: 'max_connections'
  properties: {
    value: '100'
    source: 'user-override'
  }
}

output id string = postgresqlServer.id
output name string = postgresqlServer.name
output host string = postgresqlServer.properties.fullyQualifiedDomainName
output databaseName string = database.name
output connectionString string = 'postgresql://${administratorLogin}:${administratorPassword}@${postgresqlServer.properties.fullyQualifiedDomainName}:5432/${databaseName}?sslmode=require'
