# Implementation Status — V0.1 Initial

## Implemented

- Monorepo contract packages.
- Strict `chatgpt.compute.job/v1` model with immutable source commit.
- GitHub Issue protocol for Job / Run / Status records.
- Trusted account registry -> GitHub Environment mapping.
- Trusted argv-only execution profiles.
- Source repository allowlist.
- Parallel matrix planning with unique account IDs per batch.
- Private source dataset packaging.
- Private Kaggle kernel generation and submission.
- Kaggle CLI pinned to `2.2.4` in active workflows.
- On-demand `/kaggle status` workflow.
- Selected output download for terminal tasks.
- Log sanitization/redaction.
- Deterministic failure taxonomy and fingerprint.
- SHA-256 terminal evidence manifest.
- Security gate to reject `shell=True`, `eval`, `exec`, `os.system`, raw Issue-body shell
  interpolation, credential-like account config, and legacy command profiles.

## Validation performed in this build

- 21 Python tests: PASS.
- Python compileall: PASS.
- Security gate: PASS.
- GitHub workflow YAML parse: PASS.
- Integration simulation: dispatch -> run record: PASS.
- Integration simulation: failed status -> sanitized failure -> hash manifest: PASS.

## Not validated yet

A real Kaggle API call has not been executed because no Kaggle credential is present in this
working environment. The first live validation should be a minimal CPU profile on one configured
account, followed by a two-account parallel batch.

## Required before first live run

1. Create `rezanory/chatgpt-plugins` (private).
2. Push this source.
3. Configure real public `owner_slug` values and enable accounts in `accounts.json`.
4. Create matching GitHub Environments and add Kaggle tokens.
5. Optionally add a read-only source token for private target repositories.
6. Open the first `[KAGGLE-JOB]` Issue from ChatGPT.
