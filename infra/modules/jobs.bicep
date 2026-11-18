// Container Apps Jobs (ADR-009): pp-nightly runs the pipeline on a cron, pp-migrate applies
// migrations when deploy.yml starts it, pp-retrain trains a challenger when retrain.yml starts it
// (Phase 7). All run the jobs image in the pp-api environment, so their logs go to the same capped
// workspace.
// $0 guard values are literals (ADR-017): at most 1 vCPU / 2 GiB, 45 min, no retries, one replica,
// and at most one scheduled run a day. Worst case 31 x 2,700 s x 1 vCPU = 83.7k of the 180k vCPU-s
// grant.
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

@description('The GitHub App that publishes site-data and opens failure issues (ADR-018).')
param githubAppId string

@secure()
param githubAppPrivateKey string

@description('The free Qdrant cluster the index stage keeps in step (Phase 3, design D5). The URL is a secret too: with the key, it is the only door to the cluster.')
@secure()
param qdrantUrl string

@secure()
param qdrantApiKey string

@description('The free-tier Gemini key the reports stage answers with (Phase 5, design D8).')
@secure()
param geminiApiKey string

@description('Langfuse keys: every nightly report is traced (Phase 5 Task 18).')
@secure()
param langfusePublicKey string

@secure()
param langfuseSecretKey string

@description('DagsHub\'s MLflow, where pp-retrain logs its runs and registers its challenger (Phase 7, design D5).')
param mlflowTrackingUri string

param mlflowTrackingUsername string

@description('A DagsHub token with write access: pp-retrain only, never the nightly (design D5).')
@secure()
param dagshubToken string

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
        // 03:00 UTC, after Steam's day has closed in every timezone we read; every other day (odd
        // days of the month), so the free SQL allowance holds a month (ADR-017 amendment, 4 Nov 2026).
        cronExpression: '0 3 */2 * *'
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
        {
          name: 'github-app-key'
          value: githubAppPrivateKey
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
          name: 'gemini-api-key'
          value: geminiApiKey
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
          name: 'nightly'
          image: jobsImage
          args: [
            'nightly'
            '--publish'
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
              name: 'PP_GITHUB_REPOSITORY'
              value: 'Seif-Douida/patch'
            }
            {
              name: 'PP_GITHUB_APP_ID'
              value: githubAppId
            }
            {
              name: 'PP_GITHUB_APP_PRIVATE_KEY'
              secretRef: 'github-app-key'
            }
            {
              // Of the 45 min: 10 backfill, 5 scoring, 4 spot-check, 5 indexing, 2 effects,
              // 2 drift, 3 reports and 2 online evals leave 12 for the incremental stages, dbt,
              // the reconciliation and the export; the runner stops the budgeted stages 10 min
              // before the timeout whatever their budgets (NIGHT_RESERVE_S). Backfill had 25
              // until the refill was over (Phase 5) and 20 until Phase 7 (it ran 0 s on 17 Nov
              // 2026: every game's history is complete).
              name: 'PP_BACKFILL_BUDGET_MINUTES'
              value: '10'
            }
            {
              // Aspect scoring with the champion baked into the image (design D8).
              name: 'PP_SCORE_BUDGET_MINUTES'
              value: '5'
            }
            {
              // Phase 3 (design D5): the index stage syncs Qdrant with the encoder baked into the
              // image (PP_ENCODER_DIR); what's left waits a night. 0 turns it off.
              name: 'PP_INDEX_BUDGET_MINUTES'
              value: '5'
            }
            {
              // Phase 4 (design D10): the effects stage estimates the patches whose window ended
              // or whose inputs changed; what's left waits a night. 0 turns it off.
              name: 'PP_EFFECTS_BUDGET_MINUTES'
              value: '2'
            }
            {
              // Phase 5 (design D10): the reports stage writes the due reports, newest first;
              // what's left waits a night. 0 turns it off.
              name: 'PP_REPORTS_BUDGET_MINUTES'
              value: '3'
            }
            {
              // Phase 7 (design D3): the teacher spot-check, 10 requests to Gemma 4 31B on the
              // free tier (tokens-bound, ~3.5 min). 0 turns it off.
              name: 'PP_SPOTCHECK_BUDGET_MINUTES'
              value: '4'
            }
            {
              // Phase 7 (design D4): the online evals judge tonight's reports. 0 turns it off.
              name: 'PP_ONLINE_EVAL_BUDGET_MINUTES'
              value: '2'
            }
            {
              // Phase 7 (design D2): the weekly drift check, inside the night's wake (no
              // separate job: design D1). 0 turns it off.
              name: 'PP_MONITOR_BUDGET_MINUTES'
              value: '2'
            }
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

// Phase 7 (design D5): retrain.yml starts it to train a challenger S0, compare it with the
// champion baked into the image and register it on DagsHub. S0 trains in under a minute; the
// 45 min cap is cost-guard's, which counts its executions with the other jobs'.
resource retrain 'Microsoft.App/jobs@2025-07-01' = {
  name: 'pp-retrain'
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
      replicaTimeout: 2700
      replicaRetryLimit: 0
      secrets: [
        {
          name: 'writer-password'
          value: sqlWriterPassword
        }
        {
          name: 'dagshub-token'
          value: dagshubToken
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'retrain'
          image: jobsImage
          args: [
            'retrain'
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
              // The champion baked into the image: what the challenger is compared with.
              name: 'PP_MODEL_DIR'
              value: '/app/model'
            }
            {
              name: 'MLFLOW_TRACKING_URI'
              value: mlflowTrackingUri
            }
            {
              name: 'MLFLOW_TRACKING_USERNAME'
              value: mlflowTrackingUsername
            }
            {
              name: 'MLFLOW_TRACKING_PASSWORD'
              secretRef: 'dagshub-token'
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
