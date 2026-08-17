param(
    [int]$Port = 8000,
    [switch]$SkipGatewayInstall,
    [switch]$GatewayAlreadyRunning
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$gatewayScript = Join-Path $PSScriptRoot "start-private.ps1"
$gatewayProcess = $null
$managedTunnelToken = $false

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

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "cloudflared is required. Install the current Cloudflare Tunnel client first."
}

if ([string]::IsNullOrWhiteSpace($env:TUNNEL_TOKEN)) {
    Set-ProcessSecretFromPrompt -EnvironmentName "TUNNEL_TOKEN" -Prompt "Cloudflare remotely-managed Tunnel token"
    $managedTunnelToken = $true
}

try {
    if (-not $GatewayAlreadyRunning) {
        $gatewayArguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", ('"' + $gatewayScript + '"'),
            "-Port", [string]$Port
        )
        if ($SkipGatewayInstall) {
            $gatewayArguments += "-SkipInstall"
        }

        Write-Host "Starting Kaggle Direct Gateway on loopback."
        $gatewayProcess = Start-Process powershell -ArgumentList $gatewayArguments -PassThru
    }

    Wait-LocalPort -TargetPort $Port
    Write-Host "Gateway is reachable on http://127.0.0.1:$Port/mcp"
    Write-Host "Starting remotely-managed Cloudflare Tunnel. Tunnel token is not printed."
    Write-Host "The Cloudflare published application route must point to http://localhost:$Port."

    & cloudflared tunnel --no-autoupdate run --token $env:TUNNEL_TOKEN
    exit $LASTEXITCODE
}
finally {
    if ($null -ne $gatewayProcess -and -not $gatewayProcess.HasExited) {
        Stop-Process -Id $gatewayProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($managedTunnelToken) {
        [Environment]::SetEnvironmentVariable("TUNNEL_TOKEN", $null, "Process")
    }
}
