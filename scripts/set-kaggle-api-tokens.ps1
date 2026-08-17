param(
    [string]$Repo = "rezanory/chatgpt-plugins",
    [string]$AccountsFile = "plugins/kaggle/config/accounts.json"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is required."
}

gh auth status | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated."
}

if (-not (Test-Path $AccountsFile)) {
    throw "Accounts file not found: $AccountsFile"
}

$accounts = Get-Content -Raw -Encoding UTF8 $AccountsFile | ConvertFrom-Json
$enabled = @($accounts | Where-Object { $_.enabled -eq $true })

Write-Host "Configuring KAGGLE_API_TOKEN for $($enabled.Count) enabled Kaggle account environments."
Write-Host "Tokens are read interactively, sent to gh over stdin, and are never written to disk by this script."

foreach ($account in $enabled) {
    $id = [string]$account.account_id
    $environment = [string]$account.secret_scope
    $owner = [string]$account.owner_slug

    Write-Host ""
    Write-Host "Account $id | owner=$owner | environment=$environment"

    gh api --method PUT "repos/$Repo/environments/$environment" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to ensure GitHub Environment exists: $environment"
    }

    $secure = Read-Host "Paste the NEW Kaggle API token from Settings -> API -> Generate New Token" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    $plain = $null
    try {
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        if ([string]::IsNullOrWhiteSpace($plain)) {
            throw "Empty token provided for $id"
        }
        $plain | gh secret set KAGGLE_API_TOKEN --repo $Repo --env $environment
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to set KAGGLE_API_TOKEN for $environment"
        }
        Write-Host "Configured $id successfully."
    }
    finally {
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
        $plain = $null
        $secure = $null
    }
}

Write-Host ""
Write-Host "Token bootstrap complete. Do not save token values in files or paste them into ChatGPT."
Write-Host "Next step: trigger a [KAGGLE-AUTH] Issue and require AUTH_OK before inventory or compute."
