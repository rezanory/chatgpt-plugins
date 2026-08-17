param(
    [int]$Port = 8000,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$gatewayPath = Join-Path $repoRoot "plugins/kaggle-gateway"
$accountsPath = Join-Path $gatewayPath "src/chatgpt_plugin_kaggle_gateway/accounts.json"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is required."
}
if (-not (Test-Path $accountsPath)) {
    throw "Gateway account registry was not found: $accountsPath"
}

# Global Kaggle variables are intentionally forbidden because KaggleApi.authenticate() can load
# them after per-account config and accidentally override the selected account.
foreach ($name in @("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")) {
    if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "Remove global $name before starting the multi-account gateway."
    }
}

$accounts = Get-Content -Raw -Encoding UTF8 $accountsPath | ConvertFrom-Json
$enabled = @($accounts | Where-Object { $_.enabled -eq $true })
$missing = @()
foreach ($account in $enabled) {
    foreach ($envName in @([string]$account.username_env, [string]$account.token_env)) {
        $value = [Environment]::GetEnvironmentVariable($envName)
        if ([string]::IsNullOrWhiteSpace($value)) {
            $missing += $envName
        }
    }
}

if ($missing.Count -gt 0) {
    $names = ($missing | Sort-Object -Unique) -join ", "
    throw "Missing gateway secret environment variables: $names. Values were not printed."
}

if (-not $SkipInstall) {
    python -m pip install --disable-pip-version-check -q -e $gatewayPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install chatgpt-plugin-kaggle-gateway."
    }
}

$env:CGP_GATEWAY_HOST = "127.0.0.1"
$env:CGP_GATEWAY_PORT = [string]$Port
Remove-Item Env:CGP_GATEWAY_ALLOW_UNAUTHENTICATED_REMOTE -ErrorAction SilentlyContinue

Write-Host "Starting Kaggle Direct Gateway on loopback only."
Write-Host "MCP endpoint: http://127.0.0.1:$Port/mcp"
Write-Host "Enabled accounts: $($enabled.Count). Credential values are not printed."

python -m chatgpt_plugin_kaggle_gateway.server
exit $LASTEXITCODE
