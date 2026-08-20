param(
  [switch]$SkipGitHubCli,
  [switch]$SkipKaggle,
  [switch]$SkipCloudflare
)

$ErrorActionPreference = 'Stop'

function Add-GitHubPath([string]$PathValue) {
  if (-not (Test-Path $PathValue)) { throw "Path does not exist: $PathValue" }
  $env:PATH = "$PathValue;$env:PATH"
  if ($env:GITHUB_PATH) {
    [IO.File]::AppendAllText($env:GITHUB_PATH, "$PathValue`n", [Text.UTF8Encoding]::new($false))
  }
}

function Resolve-Python {
  $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
  if (-not $python) { $python = (Get-Command python -ErrorAction Stop).Source }
  return $python
}

function Add-PythonUserScripts([string]$PythonExe) {
  $scripts = (& $PythonExe -c "import sysconfig; print(sysconfig.get_path('scripts', scheme=sysconfig.get_preferred_scheme('user')))" | Select-Object -First 1).Trim()
  if (-not $scripts) { throw 'Unable to resolve Python user Scripts path' }
  if (-not (Test-Path $scripts)) {
    $userBase = (& $PythonExe -m site --user-base | Select-Object -First 1).Trim()
    $fallback = Get-ChildItem $userBase -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match '^Python\d+$' } |
      ForEach-Object { Join-Path $_.FullName 'Scripts' } |
      Where-Object { Test-Path $_ } |
      Select-Object -First 1
    if ($fallback) { $scripts = $fallback }
  }
  if (-not (Test-Path $scripts)) { throw "Python user Scripts path does not exist: $scripts" }
  Add-GitHubPath $scripts
  return $scripts
}

if (-not $SkipKaggle) {
  $python = Resolve-Python
  & $python -m pip install --disable-pip-version-check --upgrade kaggle kagglehub
  if ($LASTEXITCODE -ne 0) { throw 'Kaggle CLI/kagglehub bootstrap failed' }
  $pythonScripts = Add-PythonUserScripts $python
  & $python -c "import kagglehub; print('kagglehub', getattr(kagglehub, '__version__', 'unknown'))"
  if ($LASTEXITCODE -ne 0) { throw 'kagglehub import/version probe failed' }
  $kaggleCommand = Get-Command kaggle.exe -ErrorAction SilentlyContinue
  $kaggleExe = if ($kaggleCommand) { $kaggleCommand.Source } else { Join-Path $pythonScripts 'kaggle.exe' }
  if (-not (Test-Path $kaggleExe)) { throw "kaggle.exe not found after installation: $kaggleExe" }
  & $kaggleExe --version
  if ($LASTEXITCODE -ne 0) { throw 'Kaggle CLI version probe failed' }
}

if (-not $SkipCloudflare) {
  $worker = Join-Path $PSScriptRoot '..\deploy\cloudflare-worker-free'
  $worker = (Resolve-Path $worker).Path
  Push-Location $worker
  try {
    npm install --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'Wrangler bootstrap failed' }
    $nodeBin = Join-Path $worker 'node_modules\.bin'
    Add-GitHubPath $nodeBin
    wrangler --version
    if ($LASTEXITCODE -ne 0) { throw 'Wrangler version probe failed' }
  } finally {
    Pop-Location
  }
}

if (-not $SkipGitHubCli) {
  $existing = Get-Command gh.exe -ErrorAction SilentlyContinue
  if (-not $existing) {
    $headers = @{ 'User-Agent' = 'chatgpt-control-plane-v3-bootstrap' }
    if ($env:GITHUB_TOKEN) { $headers.Authorization = "Bearer $env:GITHUB_TOKEN" }
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/cli/cli/releases/latest' -Headers $headers -TimeoutSec 60
    $asset = @($release.assets | Where-Object { $_.name -match '^gh_[0-9.]+_windows_amd64\.zip$' }) | Select-Object -First 1
    if (-not $asset) { throw 'Unable to locate portable Windows amd64 GitHub CLI asset' }
    $root = Join-Path $env:RUNNER_TEMP 'control-plane-v3-gh'
    if (Test-Path $root) { Remove-Item -Recurse -Force $root }
    New-Item -ItemType Directory -Force $root | Out-Null
    $zip = Join-Path $root $asset.name
    Invoke-WebRequest -Uri $asset.browser_download_url -Headers $headers -OutFile $zip -TimeoutSec 180
    Expand-Archive -LiteralPath $zip -DestinationPath $root -Force
    $gh = Get-ChildItem $root -Recurse -File -Filter 'gh.exe' | Select-Object -First 1
    if (-not $gh) { throw 'gh.exe missing after portable GitHub CLI extraction' }
    Add-GitHubPath $gh.Directory.FullName
  }
  gh --version
  if ($LASTEXITCODE -ne 0) { throw 'GitHub CLI version probe failed' }
}

Write-Host 'CONTROL_PLANE_V3_BOOTSTRAP_PASS'
