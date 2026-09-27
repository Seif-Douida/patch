// Static Web App on the Free plan. It stays empty until Phase 1 (publish-site.yml) deploys the dashboard.
param location string
param tags object

resource site 'Microsoft.Web/staticSites@2024-11-01' = {
  name: 'pp-web'
  location: location
  tags: tags
  sku: {
    name: 'Free'
    tier: 'Free'
  }
  properties: {}
}

output defaultHostname string = site.properties.defaultHostname
