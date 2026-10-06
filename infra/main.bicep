// PatchPulse runtime infrastructure, deployed by .github/workflows/infra.yml at resource-group scope.
// The resource group itself comes from infra/bootstrap/bootstrap.ps1 (ADR-016).
// $0 guard values are literals inside the modules on purpose: tests/infra/test_zero_cost_guards.py
// asserts them, and a parameterised guard value fails that test (ADR-017).
targetScope = 'resourceGroup'

@description('Region for everything except the Static Web App.')
param location string = resourceGroup().location

@description('SWA is offered in five regions and West Europe rejects new customers on this subscription (ADR-002). The location only hosts the unused managed API; static content is served globally.')
param staticWebAppLocation string = 'eastus2'

@description('Image for pp-api. infra.yml passes the image that is already running, so an infra deploy never rolls the app back.')
param apiImage string

@description('SQL admin password, from the GitHub environment secret SQL_ADMIN_PASSWORD.')
@secure()
param sqlAdminPassword string

@description('User principal name of the Entra user who becomes SQL Entra admin.')
param sqlEntraAdminLogin string

@description('Object id of that Entra user.')
param sqlEntraAdminObjectId string

@description('Public IPv4 address of the dev machine, allowed through the SQL firewall.')
param devIpAddress string

@description('Image for pp-nightly and pp-migrate. infra.yml passes the image the jobs already run, like apiImage.')
param jobsImage string

@description('Password of the pp_writer database user, from the GitHub environment secret SQL_WRITER_PASSWORD. pp-migrate sets it; pp-nightly logs in with it.')
@secure()
param sqlWriterPassword string

@description('HMAC salt for author hashes, from the secret AUTHOR_HASH_SALT. Never rotated: a new salt makes every reviewer look new.')
@secure()
param authorHashSalt string

@description('Address the backup cost alert emails (ADR-017 option B), from the secret ALERT_EMAIL. Secure only to keep it out of the public repo and logs.')
@secure()
param alertEmail string

var tags = {
  project: 'patchpulse'
  managedBy: 'bicep'
}

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring'
  params: {
    location: location
    tags: tags
  }
}

module web 'modules/web.bicep' = {
  name: 'web'
  params: {
    location: staticWebAppLocation
    tags: tags
  }
}

module containerApps 'modules/containerapps.bicep' = {
  name: 'containerapps'
  params: {
    location: location
    tags: tags
    logAnalyticsWorkspaceName: monitoring.outputs.logAnalyticsWorkspaceName
    apiImage: apiImage
    allowedOrigin: 'https://${web.outputs.defaultHostname}'
  }
}

module sql 'modules/sql.bicep' = {
  name: 'sql'
  params: {
    location: location
    tags: tags
    adminPassword: sqlAdminPassword
    entraAdminLogin: sqlEntraAdminLogin
    entraAdminObjectId: sqlEntraAdminObjectId
    devIpAddress: devIpAddress
  }
}

module jobs 'modules/jobs.bicep' = {
  name: 'jobs'
  params: {
    location: location
    tags: tags
    environmentId: containerApps.outputs.environmentId
    jobsImage: jobsImage
    sqlServerFqdn: sql.outputs.serverFqdn
    sqlDatabaseName: sql.outputs.databaseName
    sqlAdminLogin: sql.outputs.adminLogin
    sqlAdminPassword: sqlAdminPassword
    sqlWriterPassword: sqlWriterPassword
    authorHashSalt: authorHashSalt
  }
}

module alerts 'modules/alerts.bicep' = {
  name: 'alerts'
  params: {
    tags: tags
    apiId: containerApps.outputs.apiId
    alertEmail: alertEmail
  }
}

output apiUrl string = 'https://${containerApps.outputs.apiFqdn}'
output staticWebAppHostname string = web.outputs.defaultHostname
output sqlServerFqdn string = sql.outputs.serverFqdn
output appInsightsName string = monitoring.outputs.appInsightsName
