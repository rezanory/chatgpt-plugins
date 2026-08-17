param(
    [Parameter(Mandatory = $true)]
    [string]$TunnelId,
    [string]$Profile = "kaggle-direct",
    [int]$Port = 8000,
    [switch]$SkipInit,
    [switch]$DoctorOnly
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command tunnel-client -ErrorAction SilentlyContinue)) {
    throw "tunnel-client is required. Install the current release from OpenAI Platform tunnel settings."
}

if ([string]::IsNullOrWhiteSpace($env:CONTROL_PLANE_API_KEY)) {
    throw "CONTROL_PLANE_API_KEY is required for tunnel-client. Its value must not be stored in Git."
}

if ($TunnelId -notmatch '^tunnel_[A-Za-z0-9]+$') {
    throw "TunnelId must use the tunnel_... form from OpenAI Platform tunnel settings."
}

$endpoint = "http://127.0.0.1:$Port/mcp"
$probe = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue
if (-not $probe.TcpTestSucceeded) {
    throw "Kaggle Gateway is not listening on 127.0.0.1:$Port. Start start-private.ps1 first."
}

if (-not $SkipInit) {
    Write-Host "Initializing Secure MCP Tunnel profile '$Profile'."
    & tunnel-client init `
        --sample sample_mcp_stdio_local `
        --profile $Profile `
        --tunnel-id $TunnelId `
        --mcp-server-url $endpoint
    if ($LASTEXITCODE -ne 0) {
        throw "tunnel-client init failed."
    }
}

Write-Host "Validating tunnel profile without printing the runtime API key."
& tunnel-client doctor --profile $Profile --explain
if ($LASTEXITCODE -ne 0) {
    throw "tunnel-client doctor failed."
}

if ($DoctorOnly) {
    Write-Host "Tunnel doctor PASS. Not starting the long-running tunnel because -DoctorOnly was set."
    exit 0
}

Write-Host "Starting Secure MCP Tunnel. Keep this process running while ChatGPT uses the Kaggle app."
Write-Host "Private MCP endpoint: $endpoint"
& tunnel-client run --profile $Profile
exit $LASTEXITCODE
