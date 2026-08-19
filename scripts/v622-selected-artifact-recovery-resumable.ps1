$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$uri = 'https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/selected-artifact-recovery'
$headers = @{ Authorization = "Bearer $env:V622_RECOVERY_TOKEN"; 'Content-Type' = 'application/json' }
$failurePath = Join-Path $env:RUNNER_TEMP 'v622-selected-recovery-failure.txt'

function Write-Failure([string]$message) {
  [System.IO.File]::WriteAllText($failurePath, $message, [System.Text.UTF8Encoding]::new($false))
}

function Get-SelectedRecoverySnapshot {
  $last = $null
  for ($attempt = 1; $attempt -le 12; $attempt++) {
    try {
      $candidate = Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -Body '{}' -TimeoutSec 90
      if ($candidate.project -ne 'PNEUMONIA V6.2.2' -or
          $candidate.stage -ne 'selected_backbone_checkpoint_config_recovery' -or
          $candidate.status -ne 'COMPLETE' -or
          $candidate.kernel_ref -ne 'trickermark/pneumonia-v6-2-2-backbone-m06-r224' -or
          [int]$candidate.target_count -ne 6) {
        throw 'stale selected recovery response'
      }
      if ($candidate.recipe_sha256 -ne '27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f') { throw 'Recipe hash mismatch' }
      if ($candidate.frozen_policy_sha256 -ne '7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861') { throw 'Policy hash mismatch' }
      if ($candidate.locked_test_used -ne $false -or $candidate.external_data_used -ne $false) { throw 'Locked/external data unexpectedly used' }
      return $candidate
    }
    catch {
      $last = $_.Exception.Message
      if ($attempt -lt 12) { Start-Sleep -Seconds 5 }
    }
  }
  throw "Selected recovery bridge did not converge: $last"
}

function Resolve-FreshTarget(
  [string]$candidateId,
  [string]$role,
  [string]$basename,
  [string]$sourceName
) {
  $fresh = Get-SelectedRecoverySnapshot
  $matches = @($fresh.targets | Where-Object {
    [string]$_.candidate_id -eq $candidateId -and
    [string]$_.role -eq $role -and
    [string]$_.basename -eq $basename -and
    [string]$_.source_file_name -eq $sourceName
  })
  if ($matches.Count -ne 1) {
    throw "Fresh recovery target must resolve exactly once for ${candidateId}|${role}; found $($matches.Count)"
  }
  return $matches[0]
}

try {
  $curl = (Get-Command curl.exe -ErrorAction Stop).Source
  Write-Host "Using resumable downloader: $curl"

  $initial = Get-SelectedRecoverySnapshot
  $plan = @()
  $seen = @{}

  foreach ($target in @($initial.targets)) {
    $candidateId = [string]$target.candidate_id
    $role = [string]$target.role
    $basename = [string]$target.basename
    $sourceName = [string]$target.source_file_name

    if (@('M06__convnext_tiny','M06__densenet121','M06__resnet50v2') -notcontains $candidateId) { throw "Unexpected candidate: $candidateId" }
    if (@('config','checkpoint') -notcontains $role) { throw "Unexpected artifact role: $role" }
    if ($role -eq 'config' -and $basename -ne 'config.json') { throw 'Unexpected config basename' }
    if ($role -eq 'checkpoint' -and $basename -ne 'final_selected.keras') { throw 'Unexpected checkpoint basename' }

    $key = "$candidateId|$role"
    if ($seen.ContainsKey($key)) { throw "Duplicate selected artifact: $key" }
    $seen[$key] = $true

    $plan += [pscustomobject]@{
      candidate_id = $candidateId
      role = $role
      basename = $basename
      source_file_name = $sourceName
      expected_sha256 = if ($role -eq 'config') { [string]$target.expected_sha256 } else { $null }
    }
  }

  if ($plan.Count -ne 6 -or $seen.Count -ne 6) { throw 'Selected artifact recovery cardinality mismatch' }

  $root = Join-Path $env:RUNNER_TEMP 'v622-selected-backbone-checkpoints-configs'
  New-Item -ItemType Directory -Force -Path $root | Out-Null
  $rows = @()

  foreach ($target in $plan) {
    $candidateId = [string]$target.candidate_id
    $role = [string]$target.role
    $basename = [string]$target.basename
    $sourceName = [string]$target.source_file_name
    $key = "$candidateId|$role"
    $dir = Join-Path $root $candidateId
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    $dest = Join-Path $dir $basename
    $part = "$dest.part"
    if (Test-Path $dest) { Remove-Item -Force $dest }

    $downloaded = $false
    $downloadLast = $null
    $maxAttempts = if ($role -eq 'checkpoint') { 10 } else { 5 }

    for ($downloadAttempt = 1; $downloadAttempt -le $maxAttempts; $downloadAttempt++) {
      try {
        $freshTarget = Resolve-FreshTarget $candidateId $role $basename $sourceName
        $downloadUrl = [string]$freshTarget.url
        if ([string]::IsNullOrWhiteSpace($downloadUrl)) { throw "Fresh recovery URL missing for $key" }

        $existingBytes = if (Test-Path $part) { [int64](Get-Item $part).Length } else { [int64]0 }
        if ($existingBytes -gt 0) {
          Write-Host "Resuming $key from byte $existingBytes (attempt $downloadAttempt/$maxAttempts; fresh URL)"
        }
        else {
          Write-Host "Downloading $key from byte 0 (attempt $downloadAttempt/$maxAttempts; fresh URL)"
        }

        $curlArgs = @(
          '--fail',
          '--location',
          '--silent',
          '--show-error',
          '--http1.1',
          '--connect-timeout', '30',
          '--max-time', '900',
          '--speed-time', '120',
          '--speed-limit', '1024',
          '--output', $part
        )
        if ($existingBytes -gt 0) {
          $curlArgs += @('--continue-at', '-')
        }
        $curlArgs += @($downloadUrl)

        & $curl @curlArgs
        $curlExit = $LASTEXITCODE

        if ($curlExit -eq 0) {
          if (-not (Test-Path $part)) { throw "Download missing after curl success: $key" }
          $bytes = [int64](Get-Item $part).Length
          if ($bytes -le 0) { throw "Downloaded empty artifact: $key" }
          Move-Item -Force $part $dest
          $downloaded = $true
          Write-Host "Completed $key ($bytes bytes)"
          break
        }

        # curl exit 33 means the endpoint rejected resume/range semantics. Reset only
        # that partial file and retry once more from byte zero with another fresh URL.
        if ($curlExit -eq 33 -and $existingBytes -gt 0) {
          $downloadLast = "curl exit 33: server rejected resume for $key; partial reset"
          Write-Warning $downloadLast
          Remove-Item -Force $part -ErrorAction SilentlyContinue
        }
        else {
          $partialBytes = if (Test-Path $part) { [int64](Get-Item $part).Length } else { [int64]0 }
          $downloadLast = "curl exit $curlExit for $key; partial_bytes=$partialBytes"
          Write-Warning $downloadLast
        }
      }
      catch {
        $partialBytes = if (Test-Path $part) { [int64](Get-Item $part).Length } else { [int64]0 }
        $downloadLast = "$($_.Exception.Message); partial_bytes=$partialBytes"
        Write-Warning "download attempt $downloadAttempt/$maxAttempts failed for ${key}: $downloadLast"
      }

      if ($downloadAttempt -lt $maxAttempts) {
        Start-Sleep -Seconds ([Math]::Min(30, 5 * $downloadAttempt))
      }
    }

    if (-not $downloaded) {
      throw "Selected artifact download failed after resumable retries for ${key}: $downloadLast"
    }

    $sha = (Get-FileHash -Algorithm SHA256 -Path $dest).Hash.ToLowerInvariant()
    $bytes = [int64](Get-Item $dest).Length
    if ($role -eq 'config') {
      $expected = ([string]$target.expected_sha256).ToLowerInvariant()
      if ($sha -ne $expected) { throw "Config SHA mismatch for $candidateId" }
    }

    $rows += [ordered]@{
      candidate_id = $candidateId
      role = $role
      source_file_name = $sourceName
      local_path = "$candidateId/$basename"
      bytes = $bytes
      sha256 = $sha
      expected_sha256 = if ($role -eq 'config') { [string]$target.expected_sha256 } else { $null }
    }
  }

  if ($rows.Count -ne 6) { throw 'Selected artifact recovery row count mismatch' }

  $manifest = [ordered]@{
    schema_version = 2
    project = 'PNEUMONIA V6.2.2'
    status = 'RECOVERED_SELECTED_ONLY'
    kernel_ref = 'trickermark/pneumonia-v6-2-2-backbone-m06-r224'
    recipe_sha256 = '27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f'
    frozen_policy_sha256 = '7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861'
    selected_candidate_ids = @('M06__convnext_tiny','M06__densenet121','M06__resnet50v2')
    artifacts = $rows
    selected_artifact_count = 6
    downloader = 'curl-http1.1-resumable-fresh-url'
    kaggle_compute_launched = $false
    canonical_m01_m12_rerun = $false
    hpo_or_confirmation_repeated = $false
    locked_test_used = $false
    external_data_used = $false
  }

  $manifestPath = Join-Path $root 'SELECTED_ARTIFACT_MANIFEST.json'
  [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 20), [System.Text.UTF8Encoding]::new($false))
  $manifestSha = (Get-FileHash -Algorithm SHA256 -Path $manifestPath).Hash.ToLowerInvariant()
  [System.IO.File]::WriteAllText((Join-Path $env:RUNNER_TEMP 'v622-selected-manifest-sha.txt'), $manifestSha, [System.Text.UTF8Encoding]::new($false))
  Write-Host "Recovered exactly six selected checkpoint/config artifacts; manifest_sha256=$manifestSha"
}
catch {
  Write-Failure $_.Exception.Message
  throw
}
