// Log Analytics workspace + workspace-based App Insights.
// The workspace daily cap is the ingestion guard for both, because workspace-based App Insights
// stores its data in this workspace: 0.15 GB/day x 31 days = 4.65 GB < 5 GB free per month.
param location string
param tags object

resource workspace 'Microsoft.OperationalInsights/workspaces@2025-07-01' = {
  name: 'log-patchpulse'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    workspaceCapping: {
      dailyQuotaGb: json('0.15')
    }
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-patchpulse'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: workspace.id
    IngestionMode: 'LogAnalytics'
  }
}

output logAnalyticsWorkspaceName string = workspace.name
output appInsightsName string = appInsights.name
