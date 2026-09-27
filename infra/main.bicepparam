using 'main.bicep'

// Values come from the GitHub environment `production`; infra.yml exports them as env vars.
param apiImage = readEnvironmentVariable('API_IMAGE', 'ghcr.io/seif-douida/patchpulse-api:latest')
param sqlAdminPassword = readEnvironmentVariable('SQL_ADMIN_PASSWORD')
param sqlEntraAdminLogin = readEnvironmentVariable('SQL_ENTRA_ADMIN_LOGIN')
param sqlEntraAdminObjectId = readEnvironmentVariable('SQL_ENTRA_ADMIN_OBJECT_ID')
param devIpAddress = readEnvironmentVariable('DEV_IP_ADDRESS')
