// Backup for cost-guard (ADR-017 option B). GitHub runs the 15-minute cost-guard schedule only
// every ~5 hours (measured 2026-10-05), so Azure itself watches cost-guard's burn-rate rule and
// emails Seif within minutes. It only notifies; stopping pp-api stays cost-guard's job.
// $0: one metric time series (the first 10 a month are free) and email only (free up to 1,000 a
// month; SMS and voice are billed). tests/infra/test_zero_cost_guards.py asserts both.
param tags object
param apiId string

@description('Where the alert is emailed. A secret only to keep it out of the public repo and logs.')
@secure()
param alertEmail string

resource owner 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: 'ag-patchpulse-owner'
  location: 'Global'
  tags: tags
  properties: {
    groupShortName: 'pp-owner'
    enabled: true
    emailReceivers: [
      {
        name: 'owner'
        emailAddress: alertEmail
        useCommonAlertSchema: true
      }
    ]
  }
}

resource burnRate 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'pp-api-replica-up-45-of-60-min'
  location: 'global'
  tags: tags
  properties: {
    description: 'pp-api has had a replica up for at least 45 of the last 60 minutes: a request flood or a stuck replica. cost-guard only runs every few hours, so run it now: GitHub, Actions, cost-guard, Run workflow, with dry_run unticked. It stops pp-api and opens an issue.'
    severity: 1
    enabled: true
    scopes: [
      apiId
    ]
    // pp-api runs at most 1 replica, so an hourly Average of 0.75 is 45 replica-minutes.
    windowSize: 'PT1H'
    evaluationFrequency: 'PT5M'
    autoMitigate: true
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'replica-up'
          criterionType: 'StaticThresholdCriterion'
          metricNamespace: 'Microsoft.App/containerApps'
          metricName: 'Replicas'
          timeAggregation: 'Average'
          operator: 'GreaterThanOrEqual'
          threshold: json('0.75')
        }
      ]
    }
    actions: [
      {
        actionGroupId: owner.id
      }
    ]
  }
}
