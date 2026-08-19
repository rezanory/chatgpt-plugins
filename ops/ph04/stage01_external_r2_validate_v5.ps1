$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ExpectedRunner = 'LAPTOP-13QINEIF-v622-r2'
$ExpectedTargetSha = '0f8ccc2fac4cf8ac767c6f4011bfd9673a69c3db'
$ExpectedSourceHead = '361cf101dbd6f50600a3b9f59d33f3e94729a0a5'
$Source = 'C:\p4'
$PinnedSitePackages = 'C:\PH4-Tools\venvs\ph04-m3-stage01-361cf101\Lib\site-packages'
$Target = Join-Path $env:GITHUB_WORKSPACE 'target'
$PortableRoot = Join-Path $env:RUNNER_TEMP ("ph04-toolchain-" + $env:GITHUB_RUN_ID + '-' + $env:GITHUB_RUN_ATTEMPT)

$PythonUrl = 'https://api.nuget.org/v3-flatcontainer/python/3.12.7/python.3.12.7.nupkg'
$PythonSha256 = '149DD298E0B7A82250CA019471770FFF079874088A4E8501CA20922D7DF3A6AC'
$PhpUrl = 'https://downloads.php.net/~windows/releases/archives/php-8.4.24-nts-Win32-vs17-x64.zip'
$PhpSha256 = '86470A30CBBAEAFB259E727DFA5CD336F2F3F0A462CD6F8E3EAC00FDBDED13CB'
$ComposerUrl = 'https://getcomposer.org/download/2.10.2/composer.phar'
$ComposerSha256 = '5EE7125F8A30A34D246CEFD0BC85B8A783B28F2AEC968994118512350D28027'
$NodeUrl = 'https://nodejs.org/dist/v22.16.0/node-v22.16.0-win-x64.zip'
$NodeSha256 = '21C2D9735C80B8F86DAB19305AA6A9F6F59BBC808F68DE3EEF09D5832E3BF BBD'.Replace(' ','')

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
    foreach ($path in @(
        'C:\Program Files\Git\cmd\git.exe',
        'C:\Program Files\Git\bin\git.exe',
        'C:\Program Files (x86)\Git\cmd\git.exe'
    )) {
        if (Test-Path -LiteralPath $path -ErrorAction SilentlyContinue) { return $path }
    }
    Fail 'INFRA_GIT_NOT_FOUND'
}
function Download-Verified([string]$Url, [string]$Destination, [string]$ExpectedSha256, [int64]$MinimumBytes) {
    Write-Host "DOWNLOAD_URL=$Url"
    Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing -TimeoutSec 180
    if (-not (Test-Path -LiteralPath $Destination)) { Fail "DOWNLOAD_MISSING:$Destination" }
    $size = (Get-Item -LiteralPath $Destination).Length
    if ($size -lt $MinimumBytes) { Fail "DOWNLOAD_TOO_SMALL:$Destination:$size" }
    $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToUpperInvariant()
    Write-Host "DOWNLOAD_SIZE=$size"
    Write-Host "DOWNLOAD_SHA256=$actual"
    if ($actual -ne $ExpectedSha256.ToUpperInvariant()) { Fail "DOWNLOAD_SHA256_MISMATCH:$Destination:$actual" }
}
function Expand-Zip([string]$Archive, [string]$Destination) {
    if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination -Recurse -Force }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
}
function Provision-Toolchain {
    if (Test-Path -LiteralPath $PortableRoot) { Remove-Item -LiteralPath $PortableRoot -Recurse -Force }
    New-Item -ItemType Directory -Path $PortableRoot -Force | Out-Null

    $pyNupkg = Join-Path $PortableRoot 'python.3.12.7.nupkg'
    $pyZip = Join-Path $PortableRoot 'python.3.12.7.zip'
    $pyExtract = Join-Path $PortableRoot 'python'
    Download-Verified $PythonUrl $pyNupkg $PythonSha256 10000000
    Copy-Item -LiteralPath $pyNupkg -Destination $pyZip -Force
    Expand-Zip $pyZip $pyExtract
    $python = Join-Path $pyExtract 'tools\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { Fail 'PORTABLE_PYTHON_EXE_MISSING' }
    & $python -c "import sys,struct; print(sys.version); raise SystemExit(0 if sys.version_info[:3] == (3,12,7) and struct.calcsize('P')*8 == 64 else 1)"
    if ($LASTEXITCODE -ne 0) { Fail 'PORTABLE_PYTHON_IDENTITY_MISMATCH' }

    $phpZip = Join-Path $PortableRoot 'php-8.4.24-nts-win-x64.zip'
    $phpExtract = Join-Path $PortableRoot 'php'
    Download-Verified $PhpUrl $phpZip $PhpSha256 30000000
    Expand-Zip $phpZip $phpExtract
    $php = Join-Path $phpExtract 'php.exe'
    if (-not (Test-Path -LiteralPath $php)) { Fail 'PORTABLE_PHP_EXE_MISSING' }
    $phpIni = Join-Path $phpExtract 'php.ini'
    @(
        '[PHP]',
        'extension_dir="ext"',
        'date.timezone="UTC"',
        'memory_limit=512M'
    ) | Set-Content -LiteralPath $phpIni -Encoding ASCII
    foreach ($ext in @('mbstring','curl','openssl','zip','fileinfo','intl')) {
        $dll = Join-Path $phpExtract ("ext\php_" + $ext + '.dll')
        if (Test-Path -LiteralPath $dll) { Add-Content -LiteralPath $phpIni -Value ("extension=" + $ext) -Encoding ASCII }
    }
    $env:PHPRC = $phpExtract
    & $php -r 'echo PHP_VERSION, PHP_EOL; if (PHP_MAJOR_VERSION !== 8 || PHP_MINOR_VERSION !== 4 || PHP_INT_SIZE !== 8) { exit(1); }'
    if ($LASTEXITCODE -ne 0) { Fail 'PORTABLE_PHP_IDENTITY_MISMATCH' }
    & $php -m | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail 'PORTABLE_PHP_MODULE_LIST_FAILED' }

    $composer = Join-Path $PortableRoot 'composer.phar'
    Download-Verified $ComposerUrl $composer $ComposerSha256 1000000
    & $php $composer --version --no-ansi
    if ($LASTEXITCODE -ne 0) { Fail 'PORTABLE_COMPOSER_IDENTITY_FAILED' }

    $nodeZip = Join-Path $PortableRoot 'node-v22.16.0-win-x64.zip'
    $nodeExtract = Join-Path $PortableRoot 'node'
    Download-Verified $NodeUrl $nodeZip $NodeSha256 20000000
    Expand-Zip $nodeZip $nodeExtract
    $node = Join-Path $nodeExtract 'node-v22.16.0-win-x64\node.exe'
    if (-not (Test-Path -LiteralPath $node)) { Fail 'PORTABLE_NODE_EXE_MISSING' }
    $nodeVersion = & $node --version
    if ($LASTEXITCODE -ne 0 -or ([string]$nodeVersion).Trim() -ne 'v22.16.0') { Fail "PORTABLE_NODE_IDENTITY_MISMATCH:$nodeVersion" }

    return @($python,$php,$composer,$node)
}

if ($env:RUNNER_NAME -ne $ExpectedRunner) { Fail "WRONG_RUNNER_NO_PROJECT_WORK_EXECUTED:$env:RUNNER_NAME" }
Write-Host 'DEDICATED_R2_GATE=PASS'

$Git = Find-Git
$toolchain = Provision-Toolchain
$Python = $toolchain[0]
$Php = $toolchain[1]
$Composer = $toolchain[2]
$Node = $toolchain[3]

Step 'Toolchain identity' {
    Run $Git @('--version')
    Run $Python @('-c', "import sys; print(sys.version); assert sys.version_info[:3] == (3,12,7)")
    Run $Php @('-r','echo PHP_VERSION, PHP_EOL;')
    Run $Php @($Composer,'--version','--no-ansi')
    Run $Node @('--version')
    if (-not (Test-Path -LiteralPath $PinnedSitePackages)) { Fail "PINNED_SITE_PACKAGES_MISSING:$PinnedSitePackages" }
    Write-Host "PINNED_SITE_PACKAGES=$PinnedSitePackages"
}

Step 'Read-only source cache precheck' {
    if (-not (Test-Path -LiteralPath $Source)) { Fail "SOURCE_CACHE_MISSING:$Source" }
    $head = ([string](& $Git -C $Source rev-parse HEAD)).Trim()
    if ($LASTEXITCODE -ne 0) { Fail 'SOURCE_CACHE_REV_PARSE_FAILED' }
    Write-Host "SOURCE_CACHE_HEAD=$head"
    if ($head -ne $ExpectedSourceHead) { Fail "SOURCE_CACHE_HEAD_MISMATCH:$head" }
    $status = @(& $Git -C $Source status --short)
    if ($LASTEXITCODE -ne 0) { Fail 'SOURCE_CACHE_STATUS_FAILED' }
    if ($status.Count -gt 0) { $status | ForEach-Object { Write-Host $_ }; Fail 'SOURCE_CACHE_NOT_CLEAN' }
    & $Git -C $Source cat-file -e "$ExpectedTargetSha^{commit}"
    if ($LASTEXITCODE -ne 0) { Fail "TARGET_COMMIT_MISSING:$ExpectedTargetSha" }
}

Step 'Create isolated exact-SHA local clone' {
    if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Recurse -Force }
    Run $Git @('clone','--no-checkout','--no-hardlinks',$Source,$Target)
    Run $Git @('-C',$Target,'checkout','--detach',$ExpectedTargetSha)
    $head = ([string](& $Git -C $Target rev-parse HEAD)).Trim()
    Write-Host "TARGET_HEAD=$head"
    if ($head -ne $ExpectedTargetSha) { Fail "TARGET_HEAD_MISMATCH:$head" }
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

Step 'Pinned dependency environment smoke' {
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
    try { Run $Python @('-m','pytest','-q','scripts/tests/test_build_release_artifacts.py','scripts/tests/test_verify_release_attestation.py','scripts/tests/test_validate_dependency_supply_chain.py','scripts/tests/test_mansec_cluster_i_deployment.py') } finally { Pop-Location }
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
            $targetVendor = Join-Path $Plugin 'vendor'
            if (Test-Path -LiteralPath $targetVendor) { Remove-Item -LiteralPath $targetVendor -Recurse -Force }
            Copy-Item -LiteralPath $sourceVendor -Destination $targetVendor -Recurse -Force
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
    if ($targetHead -ne $ExpectedTargetSha) { Fail "FINAL_TARGET_HEAD_MISMATCH:$targetHead" }
    Run $Git @('-C',$Target,'diff','--check')
    $tracked = @(& $Git -C $Target status --short --untracked-files=no)
    if ($tracked.Count -gt 0) { $tracked | ForEach-Object { Write-Host $_ }; Fail 'VALIDATION_MODIFIED_TRACKED_TARGET_FILES' }
    $sourceHead = ([string](& $Git -C $Source rev-parse HEAD)).Trim()
    $sourceStatus = @(& $Git -C $Source status --short)
    if ($sourceHead -ne $ExpectedSourceHead -or $sourceStatus.Count -gt 0) { Fail 'READ_ONLY_SOURCE_CACHE_CHANGED_DURING_VALIDATION' }
    Write-Host 'SOURCE_CACHE_FINAL_HYGIENE=PASS'
    Write-Host "VALIDATED_SHA=$targetHead"
}

Write-Host ''
Write-Host 'PH04_STAGE01_EXTERNAL_R2_CANONICAL_CI=PASS' -ForegroundColor Green
