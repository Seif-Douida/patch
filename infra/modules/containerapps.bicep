// Container Apps environment (workload profiles v2, Consumption profile only) and the pp-api app.
// $0 guard values: Consumption profile only, 0.5 vCPU / 1 GiB, 0..1 replicas (ADR-017).
param location string
param tags object
param logAnalyticsWorkspaceName string
param apiImage string
param allowedOrigin string

@description('The free-tier Gemini key Ask answers with (Phase 5).')
@secure()
param geminiApiKey string

@description('The free Qdrant cluster: the agent\'s review search and Ask\'s counters (Phase 5). The URL is a secret too.')
@secure()
param qdrantUrl string

@secure()
param qdrantApiKey string

@description('Langfuse keys: Ask is traced at a sample (Phase 5 Task 18).')
@secure()
param langfusePublicKey string

@secure()
param langfuseSecretKey string

resource workspace 'Microsoft.OperationalInsights/workspaces@2025-07-01' existing = {
  name: logAnalyticsWorkspaceName
}

resource environment 'Microsoft.App/managedEnvironments@2025-07-01' = {
  name: 'cae-patchpulse'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspace.properties.customerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
    zoneRedundant: false
  }
}

resource api 'Microsoft.App/containerApps@2025-07-01' = {
  name: 'pp-api'
  location: location
  tags: tags
  properties: {
    environmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
        corsPolicy: {
          allowedOrigins: [
            allowedOrigin
          ]
          // Ask posts its question as JSON (Phase 5).
          allowedMethods: [
            'GET'
            'POST'
          ]
          allowedHeaders: [
            'Content-Type'
          ]
        }
      }
      secrets: [
        {
          name: 'gemini-api-key'
          value: geminiApiKey
        }
        {
          name: 'qdrant-url'
          value: qdrantUrl
        }
        {
          name: 'qdrant-api-key'
          value: qdrantApiKey
        }
        {
          name: 'langfuse-public-key'
          value: langfusePublicKey
        }
        {
          name: 'langfuse-secret-key'
          value: langfuseSecretKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          env: [
            {
              name: 'PP_GEMINI_API_KEY'
              secretRef: 'gemini-api-key'
            }
            {
              name: 'PP_QDRANT_URL'
              secretRef: 'qdrant-url'
            }
            {
              name: 'PP_QDRANT_API_KEY'
              secretRef: 'qdrant-api-key'
            }
            {
              // Phase 5 Task 18: the agent's traces (counts and ids, never review text).
              name: 'PP_LANGFUSE_PUBLIC_KEY'
              secretRef: 'langfuse-public-key'
            }
            {
              name: 'PP_LANGFUSE_SECRET_KEY'
              secretRef: 'langfuse-secret-key'
            }
            {
              // The EU region, where the Langfuse project is.
              name: 'PP_LANGFUSE_HOST'
              value: 'https://cloud.langfuse.com'
            }
            {
              name: 'PP_SITE_ORIGIN'
              value: allowedOrigin
            }
            {
              // Ask's questions a day (design D9): 15 until the compute per wake is measured.
              name: 'PP_ASK_DAILY_CAP'
              value: '15'
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        // A minute instead of the default five before scaling to zero (Phase 5, design D9): a
        // question that wakes pp-api costs about 2 replica-minutes instead of 6.
        cooldownPeriod: 60
      }
    }
  }
}

output apiFqdn string = api.properties.configuration.ingress.fqdn
output environmentId string = environment.id
output apiId string = api.id
