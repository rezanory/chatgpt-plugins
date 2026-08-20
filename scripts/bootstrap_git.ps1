param()

$ErrorActionPreference = 'Stop'

function Add-GitHubPath([string]$PathValue) {
  if (-not (Test-Path $PathValue)) { throw "Path does not exist: $PathValue" }
  $env:PATH = "$PathValue;$env:PATH"
  if ($env:GITHUB_PATH) {
    [IO.File]::AppendAllText($env:GITHUB_PATH, "$PathValue`n", [Text.UTF8Encoding]::new($false))
  }
}

$existing = Get-Command git.exe -ErrorAction SilentlyContinue
if ($existing) {
  & $existing.Source --version
  if ($LASTEXITCODE -ne 0) { throw 'Existing Git version probe failed' }
  Write-Host 'CONTROL_PLANE_V3_GIT_READY=existing'
  exit 0
}

$headers = @{ 'User-Agent' = 'chatgpt-control-plane-v3-bootstrap' }
if ($env:GITHUB_TOKEN) { $headers.Authorization = "Bearer $env:GITHUB_TOKEN" }

$release = Invoke-RestMethod `
  -Uri 'https://api.github.com/repos/git-for-windows/git/releases/latest' `
  -Headers $headers `
  -TimeoutSec 60

$asset = @(
  $release.assets |
    Where-Object {
      $_.name -match '^MinGit-.*-64-bit\.zip$' -and $_.name -notmatch 'busybox'
    }
) | Select-Object -First 1

if (-not $asset) { throw 'Unable to locate MinGit 64-bit release asset' }

$root = Join-Path $env:RUNNER_TEMP 'control-plane-v3-mingit'
if (Test-Path $root) { Remove-Item -Recurse -Force $root }
New-Item -ItemType Directory -Force $root | Out-Null
$zip = Join-Path $root $asset.name

Invoke-WebRequest `
  -Uri $asset.browser_download_url `
  -Headers $headers `
  -OutFile $zip `
  -TimeoutSec 180

$extract = Join-Path $root 'git'
New-Item -ItemType Directory -Force $extract | Out-Null
Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force

$git = Get-ChildItem $extract -Recurse -File -Filter 'git.exe' |
  Where-Object { $_.FullName -match '[\\/](cmd|bin)[\\/]git\.exe$' } |
  Select-Object -First 1

if (-not $git) { throw 'git.exe missing after MinGit extraction' }
Add-GitHubPath $git.Directory.FullName

& $git.FullName --version
if ($LASTEXITCODE -ne 0) { throw 'Portable MinGit version probe failed' }
Write-Host "CONTROL_PLANE_V3_GIT_READY=$($git.FullName)"
