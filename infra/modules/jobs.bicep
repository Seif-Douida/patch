// Container Apps Jobs (ADR-009): pp-nightly runs the pipeline on a cron, pp-migrate applies
// migrations when deploy.yml starts it. Both run the jobs image in the pp-api environment, so their
// logs go to the same capped workspace.
// $0 guard values are literals (ADR-017): at most 1 vCPU / 2 GiB, 45 min, no retries, one replica,
// and one scheduled run a day. Worst case 31 x 2,700 s x 1 vCPU = 83.7k of the 180k vCPU-s grant.
param location string
param tags object
param environmentId string
param jobsImage string
param sqlServerFqdn string
param sqlDatabaseName string
param sqlAdminLogin string

@secure()
param sqlAdminPassword string

@secure()
param sqlWriterPassword string

@secure()
param authorHashSalt string

resource nightly 'Microsoft.App/jobs@2025-07-01' = {
  name: 'pp-nightly'
  location: location
  tags: tags
  properties: {
    environmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      triggerType: 'Schedule'
      scheduleTriggerConfig: {
        cronExpression: '0 3 * * *' // 03:00 UTC, after Steam's day has closed in every timezone we read
        parallelism: 1
        replicaCompletionCount: 1
      }
      replicaTimeout: 2700
      replicaRetryLimit: 0
      secrets: [
        {
          name: 'writer-password'
          value: sqlWriterPassword
        }
        {
          name: 'author-hash-salt'
          value: authorHashSalt
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'nightly'
          image: jobsImage
          args: [
            'nightly'
          ]
          env: [
            {
              name: 'PP_DB_HOST'
              value: sqlServerFqdn
            }
            {
              name: 'PP_DB_NAME'
              value: sqlDatabaseName
            }
            {
              name: 'PP_DB_USER'
              value: 'pp_writer'
            }
            {
              name: 'PP_DB_PASSWORD'
              secretRef: 'writer-password'
            }
            {
              name: 'PP_AUTHOR_HASH_SALT'
              secretRef: 'author-hash-salt'
            }
            {
              // Of the 45 min: 25 backfill, up to 5 for one Steam rate-limit pause (ingest/http.py),
              // and the rest for the incremental stages, dbt and the export.
              name: 'PP_BACKFILL_BUDGET_MINUTES'
              value: '25'
            }
          ]
          resources: {
            cpu: 1
            memory: '2Gi'
          }
        }
      ]
    }
  }
}

resource migrate 'Microsoft.App/jobs@2025-07-01' = {
  name: 'pp-migrate'
  location: location
  tags: tags
  properties: {
    environmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      triggerType: 'Manual'
      manualTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
      }
      replicaTimeout: 900
      replicaRetryLimit: 0
      secrets: [
        {
          name: 'admin-password'
          value: sqlAdminPassword
        }
        {
          name: 'writer-password'
          value: sqlWriterPassword
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'migrate'
          image: jobsImage
          args: [
            'migrate'
          ]
          env: [
            {
              name: 'PP_DB_HOST'
              value: sqlServerFqdn
            }
            {
              name: 'PP_DB_NAME'
              value: sqlDatabaseName
            }
            {
              name: 'PP_DB_USER'
              value: sqlAdminLogin
            }
            {
              name: 'PP_DB_PASSWORD'
              secretRef: 'admin-password'
            }
            {
              name: 'PP_WRITER_PASSWORD'
              secretRef: 'writer-password'
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
        }
      ]
    }
  }
}
