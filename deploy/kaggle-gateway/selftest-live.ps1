param(
    [string]$Search = "pneumonia-v6-2-2",
    [int]$PageSize = 20,
    [int]$MaxWorkers = 6,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$gatewayPath = Join-Path $repoRoot "plugins/kaggle-gateway"
$accountsPath = Join-Path $gatewayPath "src/chatgpt_plugin_kaggle_gateway/accounts.json"

function Set-ProcessSecretFromPrompt {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentName,
        [Parameter(Mandatory = $true)][string]$Prompt
    )

    $secure = Read-Host $Prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
        if ([string]::IsNullOrWhiteSpace($plain)) {
            throw "$EnvironmentName cannot be empty."
        }
        [Environment]::SetEnvironmentVariable($EnvironmentName, $plain, "Process")
    }
    finally {
        if ($null -ne $ptr -and $ptr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
        }
        $plain = $null
        $secure = $null
    }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is required."
}
if (-not (Test-Path $accountsPath)) {
    throw "Gateway account registry was not found: $accountsPath"
}
foreach ($name in @("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")) {
    if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "Remove global $name before running the multi-account direct self-test."
    }
}

$accounts = Get-Content -Raw -Encoding UTF8 $accountsPath | ConvertFrom-Json
$enabled = @($accounts | Where-Object { $_.enabled -eq $true })
$managed = New-Object System.Collections.Generic.List[string]

try {
    foreach ($account in $enabled) {
        $usernameEnv = [string]$account.username_env
        $tokenEnv = [string]$account.token_env
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($usernameEnv))) {
            $defaultUsername = [string]$account.owner_slug
            $entered = Read-Host "$($account.account_id) Kaggle username [$defaultUsername]"
            if ([string]::IsNullOrWhiteSpace($entered)) {
                $entered = $defaultUsername
            }
            [Environment]::SetEnvironmentVariable($usernameEnv, $entered, "Process")
            $managed.Add($usernameEnv)
        }
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($tokenEnv))) {
            Set-ProcessSecretFromPrompt -EnvironmentName $tokenEnv -Prompt "$($account.account_id) Kaggle API token"
            $managed.Add($tokenEnv)
        }
    }

    if (-not $SkipInstall) {
        python -m pip install --disable-pip-version-check -q -e $gatewayPath
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install chatgpt-plugin-kaggle-gateway."
        }
    }

    Write-Host "Running direct KaggleApi self-test for $($enabled.Count) enabled accounts."
    Write-Host "No Kaggle CLI, browser session, or GitHub Actions runtime is used."
    python -m chatgpt_plugin_kaggle_gateway.selftest `
        --search $Search `
        --page-size $PageSize `
        --max-workers $MaxWorkers
    exit $LASTEXITCODE
}
finally {
    foreach ($name in $managed) {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
}
