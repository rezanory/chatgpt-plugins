Set-StrictMode -Version Latest

function Get-GitHubOidcToken {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Audience,

        [ValidateRange(1, 8)]
        [int]$MaxAttempts = 5
    )

    if (-not $env:ACTIONS_ID_TOKEN_REQUEST_URL -or -not $env:ACTIONS_ID_TOKEN_REQUEST_TOKEN) {
        throw 'GitHub OIDC request context unavailable'
    }

    $sep = if ($env:ACTIONS_ID_TOKEN_REQUEST_URL.Contains('?')) { '&' } else { '?' }
    $uri = "$($env:ACTIONS_ID_TOKEN_REQUEST_URL)$($sep)audience=$Audience"
    $headers = @{ Authorization = "Bearer $env:ACTIONS_ID_TOKEN_REQUEST_TOKEN" }

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            $response = Invoke-RestMethod -Uri $uri -Headers $headers -TimeoutSec 30
            $token = [string]$response.value
            if ($token.Length -lt 100) {
                throw [System.InvalidOperationException]::new('GitHub OIDC token unavailable')
            }
            return $token
        }
        catch {
            $status = $null
            $response = $_.Exception.Response
            if ($null -ne $response -and $null -ne $response.StatusCode) {
                try { $status = [int]$response.StatusCode } catch { $status = $null }
            }

            $transient = (
                $null -eq $status -or
                $status -in @(408, 425, 429, 500, 502, 503, 504)
            )

            if (-not $transient -or $attempt -ge $MaxAttempts) {
                throw
            }

            $delay = [Math]::Min([Math]::Pow(2, $attempt - 1), 16)
            $statusLabel = if ($null -eq $status) { 'transport' } else { [string]$status }
            Write-Host "GITHUB_OIDC_TRANSIENT_RETRY audience=$Audience status=$statusLabel attempt=$attempt/$MaxAttempts sleep_seconds=$delay"
            Start-Sleep -Seconds $delay
        }
    }

    throw 'GitHub OIDC retry loop exhausted'
}
