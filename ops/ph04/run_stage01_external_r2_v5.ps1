$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$source = Join-Path $PSScriptRoot 'stage01_external_r2_validate_v5.ps1'
if (-not (Test-Path -LiteralPath $source)) {
    throw "VALIDATOR_SOURCE_MISSING: $source"
}

$text = [System.IO.File]::ReadAllText($source)
$replacements = [ordered]@{
    'Fail "DOWNLOAD_TOO_SMALL:$Destination:$size"' = 'Fail "DOWNLOAD_TOO_SMALL:${Destination}:${size}"'
    'Fail "DOWNLOAD_SHA256_MISMATCH:$Destination:$actual"' = 'Fail "DOWNLOAD_SHA256_MISMATCH:${Destination}:${actual}"'
    "`$ComposerSha256 = '5EE7125F8A30A34D246CEFD0BC85B8A783B28F2AEC968994118512350D28027'" = "`$ComposerSha256 = '5EE7125F8A30A34D246CEFDC0BC85B8A783B28F2AEC968994118512350D28027'"
    'function Run([string]$Exe, [string[]]$Args) {' = 'function Run([string]$Exe, [string[]]$ArgumentList) {'
    '& $Exe @Args' = '& $Exe @ArgumentList'
    '& $python -c "import sys,struct; print(sys.version); raise SystemExit(0 if sys.version_info[:3] == (3,12,7) and struct.calcsize(''P'')*8 == 64 else 1)"' = '& $python -c "import sys,struct; print(sys.version); raise SystemExit(0 if sys.version_info[:3] == (3,12,7) and struct.calcsize(''P'')*8 == 64 else 1)" | Out-Host'
    '& $php -r ''echo PHP_VERSION, PHP_EOL; if (PHP_MAJOR_VERSION !== 8 || PHP_MINOR_VERSION !== 4 || PHP_INT_SIZE !== 8) { exit(1); }''' = '& $php -r ''echo PHP_VERSION, PHP_EOL; if (PHP_MAJOR_VERSION !== 8 || PHP_MINOR_VERSION !== 4 || PHP_INT_SIZE !== 8) { exit(1); }'' | Out-Host'
    '& $php $composer --version --no-ansi' = '& $php $composer --version --no-ansi | Out-Host'
}

foreach ($entry in $replacements.GetEnumerator()) {
    $count = ([regex]::Matches($text, [regex]::Escape($entry.Key))).Count
    Write-Host "PATCH_MATCH_COUNT=$count :: $($entry.Key)"
    if ($count -ne 1) {
        throw "VALIDATOR_PATCH_MATCH_COUNT_INVALID:$count"
    }
    $text = $text.Replace($entry.Key, $entry.Value)
}

$remaining = [regex]::Matches($text, '\$[A-Za-z_][A-Za-z0-9_]*:') |
    ForEach-Object { $_.Value } |
    Where-Object { $_ -notmatch '^\$env:$' } |
    Sort-Object -Unique
if ($remaining) {
    $remaining | ForEach-Object { Write-Host "SUSPICIOUS_INTERPOLATION=$_" }
    throw 'VALIDATOR_SUSPICIOUS_VARIABLE_COLON_REMAINS'
}

$temp = Join-Path $env:RUNNER_TEMP ("ph04-stage01-v5-fixed-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT.ps1")
[System.IO.File]::WriteAllText($temp, $text, [System.Text.UTF8Encoding]::new($false))

$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile($temp, [ref]$tokens, [ref]$errors) | Out-Null
if ($errors.Count -ne 0) {
    foreach ($error in $errors) {
        Write-Host "PARSER_ERROR=$($error.Message) @ $($error.Extent.Text)"
    }
    throw "VALIDATOR_PARSER_ERRORS:$($errors.Count)"
}
Write-Host 'VALIDATOR_PARSE=PASS'

try {
    & $temp
    if ($LASTEXITCODE -ne 0) {
        throw "VALIDATOR_EXIT_CODE:$LASTEXITCODE"
    }
}
finally {
    Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
}
