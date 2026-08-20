param()

$ErrorActionPreference = 'Stop'

function Add-GitHubPath([string]$PathValue) {
  if (-not (Test-Path $PathValue)) { throw "Path does not exist: $PathValue" }
  $env:PATH = "$PathValue;$env:PATH"
  if ($env:GITHUB_PATH) {
    [IO.File]::AppendAllText($env:GITHUB_PATH, "$PathValue`n", [Text.UTF8Encoding]::new($false))
  }
}

function Find-Git([string]$Root) {
  if (-not $Root -or -not (Test-Path $Root)) { return $null }
  return Get-ChildItem $Root -Recurse -File -Filter 'git.exe' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match '[\\/](cmd|bin)[\\/]git\.exe$' } |
    Select-Object -First 1
}

function Probe-Git([string]$PathValue, [string]$Source) {
  if (-not $PathValue -or -not (Test-Path $PathValue)) { return $false }
  & $PathValue --version
  if ($LASTEXITCODE -ne 0) { return $false }
  Add-GitHubPath ([IO.Path]::GetDirectoryName($PathValue))
  Write-Host "CONTROL_PLANE_V3_GIT_READY=${Source}:$PathValue"
  return $true
}

$existing = Get-Command git.exe -ErrorAction SilentlyContinue
if ($existing -and (Probe-Git $existing.Source 'existing')) { exit 0 }

$cacheBase = if ($env:RUNNER_TOOL_CACHE) {
  Join-Path $env:RUNNER_TOOL_CACHE 'control-plane-v3-mingit'
} else {
  Join-Path $env:RUNNER_TEMP 'control-plane-v3-mingit-cache'
}
$cached = Find-Git $cacheBase
if ($cached -and (Probe-Git $cached.FullName 'cache')) { exit 0 }

$legacyRoots = @(
  (Join-Path $env:RUNNER_TEMP 'control-plane-v3-mingit'),
  (Join-Path $env:RUNNER_TEMP 'control-plane-v3-mingit-cache')
)
foreach ($legacyRoot in $legacyRoots) {
  $legacyGit = Find-Git $legacyRoot
  if ($legacyGit -and (Probe-Git $legacyGit.FullName 'legacy-cache')) { exit 0 }
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

$stage = Join-Path $env:RUNNER_TEMP ("control-plane-v3-mingit-stage-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $stage | Out-Null
$zip = Join-Path $stage $asset.name

$downloaded = $false
$downloadErrors = @()
$targets = @(
  @{ Uri = [string]$asset.url; Api = $true },
  @{ Uri = [string]$asset.browser_download_url; Api = $false }
)
foreach ($target in $targets) {
  if ($downloaded) { break }
  for ($attempt = 1; $attempt -le 4; $attempt++) {
    try {
      $requestHeaders = @{}
      foreach ($key in $headers.Keys) { $requestHeaders[$key] = $headers[$key] }
      if ($target.Api) { $requestHeaders.Accept = 'application/octet-stream' }
      Invoke-WebRequest `
        -Uri $target.Uri `
        -Headers $requestHeaders `
        -OutFile $zip `
        -UseBasicParsing `
        -TimeoutSec 180
      if ((Test-Path $zip) -and (Get-Item $zip).Length -gt 1000000) {
        $downloaded = $true
        break
      }
      throw 'downloaded MinGit asset is unexpectedly small'
    } catch {
      $downloadErrors += "attempt=$attempt api=$($target.Api) error=$($_.Exception.Message)"
      if (Test-Path $zip) { Remove-Item -Force $zip -ErrorAction SilentlyContinue }
      Start-Sleep -Seconds ([Math]::Min(15, $attempt * 3))
    }
  }
}
if (-not $downloaded) {
  throw "Unable to download MinGit after retries: $($downloadErrors -join ' | ')"
}

$extract = Join-Path $stage 'git'
New-Item -ItemType Directory -Force $extract | Out-Null
Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
$git = Find-Git $extract
if (-not $git) { throw 'git.exe missing after MinGit extraction' }
& $git.FullName --version
if ($LASTEXITCODE -ne 0) { throw 'Portable MinGit version probe failed' }

if (Test-Path $cacheBase) { Remove-Item -Recurse -Force $cacheBase }
New-Item -ItemType Directory -Force (Split-Path $cacheBase -Parent) | Out-Null
Move-Item -Path $extract -Destination $cacheBase
$cached = Find-Git $cacheBase
if (-not $cached) { throw 'git.exe missing after persistent cache move' }
if (-not (Probe-Git $cached.FullName 'new-cache')) { throw 'Cached MinGit version probe failed' }

Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
