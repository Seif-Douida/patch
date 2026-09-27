<#
.SYNOPSIS
    One-time Azure bootstrap for PatchPulse (ADR-016). Run it locally, signed in as yourself.

.DESCRIPTION
    Creates what the GitHub deploy identity is deliberately not allowed to create:
      1. resource provider registrations (subscription scope, free),
      2. the resource group,
      3. an Entra app registration + service principal that GitHub Actions signs in as, trusted
         through an OIDC federated credential for the repo's `production` environment
         (no client secret exists anywhere),
      4. role assignments: Contributor on the resource group only, and Cost Management Reader on
         the subscription (read-only; used by cost-guard.yml).
    Every step checks before it creates, so the script is safe to re-run.

.EXAMPLE
    az login
    powershell -ExecutionPolicy Bypass -File .\infra\bootstrap\bootstrap.ps1 -SubscriptionId <id>
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $SubscriptionId,
    [string] $Location = 'northeurope',
    [string] $ResourceGroup = 'rg-patchpulse',
    [string] $GitHubRepo = 'Seif-Douida/patch',
    [string] $AppName = 'patchpulse-github-deploy'
)

$ErrorActionPreference = 'Stop'

# Windows PowerShell does not stop when a native command fails, so check every az exit code.
# Note: az is a .cmd file, so never put '|' or '&' inside an argument (cmd.exe would act on them).
function Invoke-Az {
    $output = & az @args
    if ($LASTEXITCODE -ne 0) { throw "az $($args -join ' ') failed (exit code $LASTEXITCODE)" }
    return $output
}

Invoke-Az account set --subscription $SubscriptionId
$tenantId = Invoke-Az account show --query tenantId -o tsv

Write-Host '[1/5] Registering resource providers (free, once per subscription)...'
$namespaces = @('Microsoft.App', 'Microsoft.OperationalInsights', 'Microsoft.Insights',
    'Microsoft.Sql', 'Microsoft.Web', 'Microsoft.CostManagement')
foreach ($namespace in $namespaces) {
    Invoke-Az provider register --namespace $namespace --wait | Out-Null
}

Write-Host "[2/5] Checking that $Location offers serverless SQL and Container Apps..."
$serverless = Invoke-Az sql db list-editions --location $Location --edition GeneralPurpose `
    --service-objective GP_S_Gen5_2 --available --query 'length(@)' -o tsv
if ([int]$serverless -lt 1) { throw "Serverless GP_S_Gen5_2 is not available in $Location." }
$displayName = Invoke-Az account list-locations --query "[?name=='$Location'].displayName" -o tsv
$appRegions = Invoke-Az provider show --namespace Microsoft.App `
    --query "resourceTypes[?resourceType=='managedEnvironments'].locations[]" -o tsv
if ($appRegions -notcontains $displayName) { throw "Container Apps is not available in $Location." }

Write-Host "[3/5] Resource group $ResourceGroup in $Location..."
Invoke-Az group create --name $ResourceGroup --location $Location `
    --tags project=patchpulse managedBy=bootstrap -o none
$resourceGroupId = Invoke-Az group show --name $ResourceGroup --query id -o tsv

Write-Host "[4/5] App registration $AppName, trusted by $GitHubRepo (environment: production)..."
$appId = Invoke-Az ad app list --display-name $AppName --query '[0].appId' -o tsv
if (-not $appId) { $appId = Invoke-Az ad app create --display-name $AppName --query appId -o tsv }
$principalId = Invoke-Az ad sp list --filter "appId eq '$appId'" --query '[0].id' -o tsv
if (-not $principalId) { $principalId = Invoke-Az ad sp create --id $appId --query id -o tsv }

$subject = "repo:${GitHubRepo}:environment:production"
$existing = Invoke-Az ad app federated-credential list --id $appId `
    --query "[?subject=='$subject'].name" -o tsv
if (-not $existing) {
    $credentialFile = New-TemporaryFile
    @{
        name      = 'github-production'
        issuer    = 'https://token.actions.githubusercontent.com'
        subject   = $subject
        audiences = @('api://AzureADTokenExchange')
    } | ConvertTo-Json | Set-Content -Path $credentialFile -Encoding ascii
    Invoke-Az ad app federated-credential create --id $appId --parameters "@$credentialFile" | Out-Null
    Remove-Item $credentialFile
}

Write-Host '[5/5] Role assignments...'
$assignments = @(
    @{ Role = 'Contributor'; Scope = $resourceGroupId },
    @{ Role = 'Cost Management Reader'; Scope = "/subscriptions/$SubscriptionId" }
)
foreach ($assignment in $assignments) {
    $count = Invoke-Az role assignment list --assignee $principalId --role $assignment.Role `
        --scope $assignment.Scope --query 'length(@)' -o tsv
    if ([int]$count -eq 0) {
        Invoke-Az role assignment create --assignee-object-id $principalId `
            --assignee-principal-type ServicePrincipal --role $assignment.Role `
            --scope $assignment.Scope -o none
    }
}

$me = (Invoke-Az ad signed-in-user show --query '{login:userPrincipalName, id:id}' -o json) -join "`n" |
    ConvertFrom-Json
$devIp = (Invoke-RestMethod -Uri 'https://api.ipify.org').Trim()

Write-Host ''
Write-Host "Done. Add these as VARIABLES of the GitHub environment 'production':"
[ordered]@{
    AZURE_CLIENT_ID       = $appId
    AZURE_TENANT_ID       = $tenantId
    AZURE_SUBSCRIPTION_ID = $SubscriptionId
} | Format-Table -AutoSize | Out-String | Write-Host
# Personal values go in SECRETS: the repo is public, and GitHub prints variables in run logs.
# (On a personal-account tenant the login embeds the sign-up email address.)
Write-Host "Add these as SECRETS of the GitHub environment 'production' (masked in logs):"
[ordered]@{
    SQL_ENTRA_ADMIN_LOGIN     = $me.login
    SQL_ENTRA_ADMIN_OBJECT_ID = $me.id
    DEV_IP_ADDRESS            = $devIp
} | Format-Table -AutoSize | Out-String | Write-Host
Write-Host 'Plus the SECRET SQL_ADMIN_PASSWORD (docs/runbooks/phase-0-setup.md, step 7).'
