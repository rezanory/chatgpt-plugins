$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ExpectedRunner = 'LAPTOP-13QINEIF-v622-r2'
$ExpectedTargetSha = '0f8ccc2fac4cf8ac767c6f4011bfd9673a69c3db'
$ExpectedSourceHead = '361cf101dbd6f50600a3b9f59d33f3e94729a0a5'
$Source = 'C:\p4'
$PinnedSitePackages = 'C:\PH4-Tools\venvs\ph04-m3-stage01-361cf101\Lib\site-packages'
$Target = Join-Path $env:GITHUB_WORKSPACE 'target'

function Fail([string]$Message) { throw $Message }
function Step([string]$Name, [scriptblock]$Body) {
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Body
    Write-Host "PASS: $Name" -ForegroundColor Green
}
function Run([string]$Exe, [string[]]$Args) {
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) { Fail "$Exe failed with exit code $LASTEXITCODE" }
}
function Find-Git {
    $candidates = @(
        'C:\Program Files\Git\cmd\git.exe',
        'C:\Program Files\Git\bin\git.exe',
        'C:\Program Files (x86)\Git\cmd\git.exe'
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) { return $path }
    }
    $found = Get-ChildItem 'C:\Users\*\AppData\Local\Programs\Git\cmd\git.exe' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { return $found.FullName }
    Fail 'INFRA_GIT_NOT_FOUND'
}
function Test-PythonCandidate([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $old = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $Path -c "import sys; print('.'.join(map(str,sys.version_info[:3]))); raise SystemExit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "PYTHON_CANDIDATE_PASS=$Path :: $($output -join ' ')"
            return $true
        }
        Write-Host "PYTHON_CANDIDATE_REJECT=$Path :: $($output -join ' ')"
        return $false
    } finally { $ErrorActionPreference = $old }
}
function Find-Python312 {
    $candidates = New-Object System.Collections.Generic.List[string]
    Get-ChildItem 'C:\actions-runners\v622-r2\_work\_tool\Python\3.12*\x64\python.exe' -File -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | ForEach-Object { $candidates.Add($_.FullName) }
    foreach ($path in @(
        'C:\Program Files\Python312\python.exe',
        'C:\Program Files\Python\Python312\python.exe',
        'C:\Users\Radlina\AppData\Local\Programs\Python\Python312\python.exe'
    )) { if (-not $candidates.Contains($path)) { $candidates.Add($path) } }
    Get-ChildItem 'C:\PH4-Tools' -Filter 'python.exe' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName | ForEach-Object {
        if (-not $candidates.Contains($_.FullName)) { $candidates.Add($_.FullName) }
    }
    foreach ($candidate in $candidates) { if (Test-PythonCandidate $candidate) { return $candidate } }
    Fail 'NO_SERVICE_EXECUTABLE_PYTHON_312_FOUND'
}
function Find-PHP84 {
    $candidates = New-Object System.Collections.Generic.List[string]
    Get-ChildItem 'C:\PH4-Tools' -Filter 'php.exe' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName | ForEach-Object { $candidates.Add($_.FullName) }
    foreach ($path in @('C:\php\php.exe','C:\Program Files\PHP\php.exe')) { if (-not $candidates.Contains($path)) { $candidates.Add($path) } }
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $old = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $version = & $candidate -r 'echo PHP_VERSION;' 2>$null
            if ($LASTEXITCODE -eq 0 -and ([string]$version).StartsWith('8.4.')) { return $candidate }
        } finally { $ErrorActionPreference = $old }
    }
    Fail 'PINNED_PHP_84_NOT_FOUND'
}
function Find-Node22 {
    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($path in @('C:\Program Files\nodejs\node.exe','C:\Program Files (x86)\nodejs\node.exe')) { $candidates.Add($path) }
    Get-ChildItem 'C:\PH4-Tools' -Filter 'node.exe' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName | ForEach-Object { if (-not $candidates.Contains($_.FullName)) { $candidates.Add($_.FullName) } }
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $old = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $version = & $candidate --version 2>$null
            if ($LASTEXITCODE -eq 0 -and ([string]$version).StartsWith('v22.')) { return $candidate }
        } finally { $ErrorActionPreference = $old }
    }
    Fail 'PINNED_NODE_22_NOT_FOUND'
}
function Find-Composer([string]$Php) {
    $candidates = @(Get-ChildItem 'C:\PH4-Tools' -Filter 'composer.phar' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName)
    foreach ($candidate in $candidates) {
        $old = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $Php $candidate.FullName --version --no-ansi *> $null
            if ($LASTEXITCODE -eq 0) { return $candidate.FullName }
        } finally { $ErrorActionPreference = $old }
    }
    Fail 'PINNED_COMPOSER_PHAR_NOT_FOUND'
}

if ($env:RUNNER_NAME -ne $ExpectedRunner) {
    Write-Host "WRONG_RUNNER=$env:RUNNER_NAME"
    Fail "WRONG_RUNNER_NO_PROJECT_WORK_EXECUTED: expected $ExpectedRunner"
}
Write-Host "DEDICATED_R2_GATE=PASS"

$Git = Find-Git
$Python = Find-Python312
$Php = Find-PHP84
$Node = Find-Node22
$Composer = Find-Composer $Php

Step 'Pinned toolchain identity' {
    Run $Git @('--version')
    Run $Python @('-c', "import sys; print(sys.version); assert sys.version_info[:2] == (3,12)")
    Run $Php @('-r','echo PHP_VERSION, PHP_EOL;')
    Run $Node @('--version')
    Run $Php @($Composer,'--version','--no-ansi')
    Write-Host "COMPOSER_SHA256=$((Get-FileHash -LiteralPath $Composer -Algorithm SHA256).Hash)"
    if (-not (Test-Path -LiteralPath $PinnedSitePackages)) { Fail "PINNED_SITE_PACKAGES_MISSING: $PinnedSitePackages" }
    Write-Host "PINNED_SITE_PACKAGES=$PinnedSitePackages"
}

Step 'Read-only source cache precheck' {
    if (-not (Test-Path -LiteralPath $Source)) { Fail "SOURCE_CACHE_MISSING: $Source" }
    $head = ([string](& $Git -C $Source rev-parse HEAD)).Trim()
    if ($LASTEXITCODE -ne 0) { Fail 'SOURCE_CACHE_REV_PARSE_FAILED' }
    Write-Host "SOURCE_CACHE_HEAD=$head"
    if ($head -ne $ExpectedSourceHead) { Fail "SOURCE_CACHE_HEAD_MISMATCH: $head" }
    $status = @(& $Git -C $Source status --short)
    if ($LASTEXITCODE -ne 0) { Fail 'SOURCE_CACHE_STATUS_FAILED' }
    if ($status.Count -gt 0) { $status | ForEach-Object { Write-Host $_ }; Fail 'SOURCE_CACHE_NOT_CLEAN' }
    & $Git -C $Source cat-file -e "$ExpectedTargetSha^{commit}"
    if ($LASTEXITCODE -ne 0) { Fail "TARGET_COMMIT_MISSING: $ExpectedTargetSha" }
}

Step 'Create isolated exact-SHA local clone' {
    if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Recurse -Force }
    Run $Git @('clone','--no-checkout','--no-hardlinks',$Source,$Target)
    Run $Git @('-C',$Target,'checkout','--detach',$ExpectedTargetSha)
    $head = ([string](& $Git -C $Target rev-parse HEAD)).Trim()
    Write-Host "TARGET_HEAD=$head"
    if ($head -ne $ExpectedTargetSha) { Fail "TARGET_HEAD_MISMATCH: $head" }
    $status = @(& $Git -C $Target status --short)
    if ($status.Count -gt 0) { $status | ForEach-Object { Write-Host $_ }; Fail 'TARGET_DIRTY_AFTER_CHECKOUT' }
}

Step 'Reconfirm source cache untouched after clone' {
    $head = ([string](& $Git -C $Source rev-parse HEAD)).Trim()
    $status = @(& $Git -C $Source status --short)
    if ($head -ne $ExpectedSourceHead -or $status.Count -gt 0) { Fail 'SOURCE_CACHE_CHANGED_AFTER_CLONE' }
}

$Root = $Target
$Backend = Join-Path $Root 'aiChatProjectPyback'
$Plugin = Join-Path $Root 'aichat-WPplugin'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONPATH = "$Backend;$Root;$PinnedSitePackages"
$env:DJANGO_SETTINGS_MODULE = 'core.settings_test'
$env:DJANGO_ENV = 'test'
$env:DEBUG = 'True'
$env:TESTING = 'True'
$env:SECRET_KEY = 'ci-test-secret-key'
$env:WS_TOKEN_SECRET = 'ci-test-ws-token-secret'
$env:ADMIN_WS_SECRET = 'ci-test-admin-ws-secret'
$env:CHANNEL_ENCRYPTION_KEY = 'm7MQ4AiQFrXx5cYh3h7cZkzW1g9KJcJkt6dQX8JX7jM='
$env:WS_ALLOWED_ORIGINS = 'http://localhost'
$env:DATABASE_URL = 'sqlite:///test_db.sqlite3'
$env:USE_REDIS = 'false'
$env:OPENAI_API_KEY = 'ci-openai-key'
$env:CLAUDE_API_KEY = 'ci-claude-key'
$env:DEEPSEEK_API_KEY = 'ci-deepseek-key'
$env:ADMIN_API_SIGNING_SECRET = 'ci-admin-signing-secret'

Step 'Pinned Python dependency smoke' {
    Run $Python @('-c', "import django,pytest; print('django='+django.get_version()); print('pytest='+pytest.__version__)")
    Run $Python @('-m','pip','check')
}

Step 'UTF-8 and dependency supply-chain gates' {
    Push-Location $Root
    try { Run $Python @('scripts/scan_text_encoding.py','.') } finally { Pop-Location }
    Push-Location $Backend
    try { Run $Python @('..\scripts\validate_dependency_supply_chain.py') } finally { Pop-Location }
}

Step 'Compile + source/migration/architecture/composition gates' {
    Push-Location $Backend
    try {
        Run $Python @('-m','compileall','-q','.')
        Run $Python @('commerce/validate_source_layout.py','--require-reserved-paths')
        Run $Python @('commerce/validate_migration_contracts.py')
        Run $Python @('commerce/validate_architecture_boundaries.py')
        Run $Python @('commerce/compose_registry.py','--check')
    } finally { Pop-Location }
}

Step 'Lane/source manifest and backend-plugin contract gates' {
    Push-Location $Root
    try {
        Run $Python @('parallel_delivery/validate_lane_merge.py')
        Run $Python @('-m','pytest','-q','parallel_delivery/tests/test_lane_merge.py')
        Run $Python @('contracts/backend_plugin/v1/validate_contracts.py')
    } finally { Pop-Location }
    Push-Location $Backend
    try { Run $Python @('-m','pytest','-q','chat/tests/test_backend_plugin_contract_fixtures.py') } finally { Pop-Location }
}

Step 'Django check and migration drift' {
    Push-Location $Backend
    try {
        Run $Python @('manage.py','check')
        Run $Python @('manage.py','makemigrations','--check','--dry-run')
    } finally { Pop-Location }
}

Step 'Full backend pytest' {
    Push-Location $Backend
    try { Run $Python @('-m','pytest','-q') } finally { Pop-Location }
}

Step 'Release and deployment validation' {
    Push-Location $Root
    try {
        Run $Python @('-m','pytest','-q','scripts/tests/test_build_release_artifacts.py','scripts/tests/test_verify_release_attestation.py','scripts/tests/test_validate_dependency_supply_chain.py','scripts/tests/test_mansec_cluster_i_deployment.py')
    } finally { Pop-Location }
}

Step 'Composer install + metadata validation' {
    Push-Location $Plugin
    try {
        & $Php $Composer install --prefer-dist --no-progress --no-interaction --no-plugins --no-scripts
        if ($LASTEXITCODE -ne 0) {
            $sourceVendor = Join-Path $Source 'aichat-WPplugin\vendor'
            $sourceLock = Join-Path $Source 'aichat-WPplugin\composer.lock'
            $targetLock = Join-Path $Plugin 'composer.lock'
            if (-not (Test-Path -LiteralPath (Join-Path $sourceVendor 'autoload.php'))) { Fail 'COMPOSER_INSTALL_FAILED_AND_NO_SOURCE_VENDOR_FALLBACK' }
            if ((Get-FileHash $sourceLock -Algorithm SHA256).Hash -ne (Get-FileHash $targetLock -Algorithm SHA256).Hash) { Fail 'SOURCE_VENDOR_FALLBACK_LOCK_MISMATCH' }
            Write-Host 'COMPOSER_NETWORK_INSTALL_FAILED; USING READ_ONLY SAME-LOCK SOURCE VENDOR COPY'
            Copy-Item -LiteralPath $sourceVendor -Destination (Join-Path $Plugin 'vendor') -Recurse -Force
        }
        Run $Php @($Composer,'validate','--strict','--no-ansi')
    } finally { Pop-Location }
}

Step 'Plugin PHP lint' {
    $files = @(Get-ChildItem -LiteralPath $Plugin -Recurse -File -Filter '*.php' | Where-Object { $_.FullName -notmatch '[\\/]vendor[\\/]' })
    Write-Host "PHP_LINT_FILE_COUNT=$($files.Count)"
    foreach ($file in $files) { Run $Php @('-l',$file.FullName) }
}

Step 'Plugin tests + shared contract fixture' {
    Push-Location $Plugin
    try {
        Run $Php @($Composer,'test','--','--no-coverage')
        Run $Php @('vendor/bin/phpunit','tests/Unit/BackendPluginSharedContractFixturesTest.php')
    } finally { Pop-Location }
}

Step 'All JavaScript state and boundary selectors' {
    $tests = @(Get-ChildItem -LiteralPath (Join-Path $Plugin 'tests\js') -File -Filter '*.test.js' | Sort-Object Name)
    Write-Host "JS_TEST_FILE_COUNT=$($tests.Count)"
    foreach ($test in $tests) { Write-Host "JS_TEST=$($test.Name)"; Run $Node @($test.FullName) }
}

Step 'Final exact-SHA and source-cache hygiene' {
    $targetHead = ([string](& $Git -C $Target rev-parse HEAD)).Trim()
    if ($targetHead -ne $ExpectedTargetSha) { Fail "FINAL_TARGET_HEAD_MISMATCH: $targetHead" }
    Run $Git @('-C',$Target,'diff','--check')
    $tracked = @(& $Git -C $Target status --short --untracked-files=no)
    if ($tracked.Count -gt 0) { $tracked | ForEach-Object { Write-Host $_ }; Fail 'VALIDATION_MODIFIED_TRACKED_TARGET_FILES' }
    $sourceHead = ([string](& $Git -C $Source rev-parse HEAD)).Trim()
    $sourceStatus = @(& $Git -C $Source status --short)
    if ($sourceHead -ne $ExpectedSourceHead -or $sourceStatus.Count -gt 0) { Fail 'READ_ONLY_SOURCE_CACHE_CHANGED_DURING_VALIDATION' }
    Write-Host 'SOURCE_CACHE_FINAL_HYGIENE=PASS'
    Write-Host "VALIDATED_SHA=$targetHead"
}

Write-Host ""
Write-Host 'PH04_STAGE01_EXTERNAL_R2_CANONICAL_CI=PASS' -ForegroundColor Green
