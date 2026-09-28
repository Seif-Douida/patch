// Azure SQL logical server + the free-offer serverless database (ADR-002).
// Free offer: 100k vCore-s, 32 GB data and 32 GB backup per month. 'AutoPause' pauses the
// database until next month instead of billing overage (ADR-017).
param location string
param tags object

@secure()
param adminPassword string
param entraAdminLogin string
param entraAdminObjectId string
param devIpAddress string

resource server 'Microsoft.Sql/servers@2025-01-01' = {
  name: 'sql-patchpulse-${uniqueString(resourceGroup().id)}'
  location: location
  tags: tags
  properties: {
    administratorLogin: 'ppadmin'
    administratorLoginPassword: adminPassword
    minimalTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    administrators: {
      administratorType: 'ActiveDirectory'
      principalType: 'User'
      login: entraAdminLogin
      sid: entraAdminObjectId
      tenantId: tenant().tenantId
      azureADOnlyAuthentication: false
    }
  }
}

resource database 'Microsoft.Sql/servers/databases@2025-01-01' = {
  parent: server
  name: 'patchpulse'
  location: location
  tags: tags
  sku: {
    name: 'GP_S_Gen5'
    tier: 'GeneralPurpose'
    family: 'Gen5'
    capacity: 2
  }
  properties: {
    useFreeLimit: true
    freeLimitExhaustionBehavior: 'AutoPause'
    // Azure only accepts the default delay (60 min) on a free database with AutoPause; 15 is rejected.
    autoPauseDelay: 60
    minCapacity: json('0.5')
    maxSizeBytes: 34359738368
    requestedBackupStorageRedundancy: 'Local'
    zoneRedundant: false
  }
}

// 0.0.0.0 means "allow Azure services". It admits any Azure tenant; accepted in ADR-002.
resource allowAzureServices 'Microsoft.Sql/servers/firewallRules@2025-01-01' = {
  parent: server
  name: 'AllowAllWindowsAzureIps'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

resource allowDevMachine 'Microsoft.Sql/servers/firewallRules@2025-01-01' = {
  parent: server
  name: 'dev-machine'
  properties: {
    startIpAddress: devIpAddress
    endIpAddress: devIpAddress
  }
}

output serverFqdn string = server.properties.fullyQualifiedDomainName
