param(
    [Parameter(Mandatory = $true)]
    [string]$TunnelId,
    [string]$Profile = "kaggle-direct",
    [int]$Port = 8000,
    [switch]$SkipInstall,
    [switch]$SkipTunnelInit
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$accountsPath = Join-Path $repoRoot "plugins/kaggle-gateway/src/chatgpt_plugin_kaggle_gateway/accounts.json"
$gatewayScript = Join-Path $PSScriptRoot "start-private.ps1"
$tunnelScript = Join-Path $PSScriptRoot "start-tunnel.ps1"

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

function Wait-LocalPort {
    param([int]$TargetPort, [int]$Attempts = 30)
    for ($i = 0; $i -lt $Attempts; $i++) {
        $probe = Test-NetConnection -ComputerName 127.0.0.1 -Port $TargetPort -WarningAction SilentlyContinue
        if ($probe.TcpTestSucceeded) {
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "Gateway did not begin listening on 127.0.0.1:$TargetPort."
}

if (-not (Test-Path $accountsPath)) {
    throw "Account registry not found: $accountsPath"
}
if (-not (Test-Path $gatewayScript) -or -not (Test-Path $tunnelScript)) {
    throw "Gateway/tunnel launcher scripts are missing."
}

foreach ($name in @("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")) {
    if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "Remove global $name before starting the multi-account gateway."
    }
}

$accounts = Get-Content -Raw -Encoding UTF8 $accountsPath | ConvertFrom-Json
$enabled = @($accounts | Where-Object { $_.enabled -eq $true })
if ($enabled.Count -eq 0) {
    throw "No enabled Kaggle accounts are configured."
}

$managedEnvironmentNames = New-Object System.Collections.Generic.List[string]
$gatewayProcess = $null

try {
    foreach ($account in $enabled) {
        $usernameEnv = [string]$account.username_env
        $tokenEnv = [string]$account.token_env
        $username = [Environment]::GetEnvironmentVariable($usernameEnv)

        if ([string]::IsNullOrWhiteSpace($username)) {
            $defaultUsername = [string]$account.owner_slug
            $entered = Read-Host "$($account.account_id) Kaggle username [$defaultUsername]"
            if ([string]::IsNullOrWhiteSpace($entered)) {
                $entered = $defaultUsername
            }
            [Environment]::SetEnvironmentVariable($usernameEnv, $entered, "Process")
            $managedEnvironmentNames.Add($usernameEnv)
        }

        $token = [Environment]::GetEnvironmentVariable($tokenEnv)
        if ([string]::IsNullOrWhiteSpace($token)) {
            Set-ProcessSecretFromPrompt -EnvironmentName $tokenEnv -Prompt "$($account.account_id) Kaggle API token"
            $managedEnvironmentNames.Add($tokenEnv)
        }
    }

    if ([string]::IsNullOrWhiteSpace($env:CONTROL_PLANE_API_KEY)) {
        Set-ProcessSecretFromPrompt -EnvironmentName "CONTROL_PLANE_API_KEY" -Prompt "OpenAI Secure MCP Tunnel runtime API key"
        $managedEnvironmentNames.Add("CONTROL_PLANE_API_KEY")
    }

    $gatewayArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"' + $gatewayScript + '"'),
        "-Port", [string]$Port
    )
    if ($SkipInstall) {
        $gatewayArguments += "-SkipInstall"
    }

    Write-Host "Starting Kaggle Direct Gateway in a child process."
    Write-Host "Enabled accounts: $($enabled.Count). Secret values are never printed."
    $gatewayProcess = Start-Process powershell -ArgumentList $gatewayArguments -PassThru
    Wait-LocalPort -TargetPort $Port

    Write-Host "Gateway is listening on loopback. Starting Secure MCP Tunnel validation."
    $tunnelArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $tunnelScript,
        "-TunnelId", $TunnelId,
        "-Profile", $Profile,
        "-Port", [string]$Port
    )
    if ($SkipTunnelInit) {
        $tunnelArguments += "-SkipInit"
    }

    & powershell @tunnelArguments
    exit $LASTEXITCODE
}
finally {
    if ($null -ne $gatewayProcess -and -not $gatewayProcess.HasExited) {
        Stop-Process -Id $gatewayProcess.Id -Force -ErrorAction SilentlyContinue
    }
    foreach ($name in $managedEnvironmentNames) {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }
}
