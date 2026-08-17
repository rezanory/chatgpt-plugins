# chatgpt-plugins

V0.1 monorepo for operating external compute from a normal ChatGPT conversation without using
ChatGPT Work or Codex Cloud as the execution runner.

The first implemented provider is **Kaggle**.

## V0.1 control path

```text
ChatGPT
  |
  | existing GitHub integration (write-capable)
  v
GitHub Issue in chatgpt-plugins
  |
  | issue opened
  v
GitHub Actions dispatcher (short lived)
  |
  | trusted account -> GitHub Environment -> KAGGLE_API_TOKEN
  v
Kaggle private source dataset + private kernel
  |
  v
Kaggle compute
```

Status is explicitly requested from the same ChatGPT conversation:

```text
ChatGPT -> GitHub issue comment: /kaggle status
        -> short status workflow
        -> Kaggle kernels status/output
        -> sanitize + hash evidence
        -> GitHub issue comment + Actions artifact
        -> ChatGPT reads the result through GitHub
```

This is a **relay**, not a bypass of ChatGPT's custom-MCP write restrictions. When direct
write-capable custom MCP is available on the target ChatGPT plan, only the transport should be
replaced; job, provider, provenance, failure, and repair contracts stay stable.

## Authentication policy

V0.1 uses **only** Kaggle's current non-interactive API-token flow:

```text
Kaggle Settings -> API -> Generate New Token
                  |
                  v
GitHub Environment secret: KAGGLE_API_TOKEN
```

Browser sessions, cookies, repeated interactive logins, OAuth browser flows, and legacy
`KAGGLE_USERNAME`/`KAGGLE_KEY` credentials are not part of the V0.1 runtime path.

## What V0.1 supports

- Immutable GitHub commit SHA as the source of every new run.
- One or many authorized Kaggle accounts configured as GitHub Environments.
- Parallel batch submission: one Issue may contain up to 16 tasks, each mapped to a distinct
  Kaggle account.
- Trusted execution profiles only; Issue content cannot provide arbitrary shell commands.
- `subprocess.run(..., shell=False)` in the Kaggle bootstrap.
- Private source dataset and private Kaggle kernel.
- Dependency-free `python-smoke` profile for bridge validation.
- On-demand status collection with `/kaggle status`.
- `[KAGGLE-AUTH]` Issue workflow for harmless token validation across enabled accounts.
- `[KAGGLE-INVENTORY] <search>` Issue workflow for discovering existing kernels through API
  tokens without submitting new compute.
- Sanitized logs, deterministic failure classification, failure fingerprinting.
- Hashed evidence manifests stored as GitHub Actions artifacts for terminal runs.
- Repair contracts with hard attempt limits; no autonomous workflow edits to source.

## Existing-run recovery

The inventory path exists specifically for workloads that were already running before this control
plane was installed. It never creates a new kernel:

```text
ChatGPT
  -> [KAGGLE-INVENTORY] pneumonia-v6-2-2
  -> enabled account environments
  -> kaggle kernels list -m -s pneumonia-v6-2-2
  -> sanitized inventory comments
  -> identify existing owner/kernel refs
  -> query status/output through API
```

This allows previously interrupted multi-account workloads to be discovered and assessed before a
new submission is considered.

## Deliberate V0.1 limits

- Cross-Issue distributed account leasing is not claimed to be atomic. For reliable parallel
  multi-account runs, place parallel tasks in **one batch Issue** with unique account IDs.
- GitHub Actions dispatches and checks; it does not wait for an hours-long Kaggle run.
- No high-churn runtime state is committed to Git.
- No multi-provider router yet.
- No arbitrary dynamic notebook generation.
- No unattended repair daemon.
- No provider quota values are hard-coded. Provider/account limits are operator configuration.

## Repository layout

```text
packages/
  core/             versioned job/provider contracts, scheduler primitives, provenance types
  github-bridge/    GitHub Issue protocol + optional future workflow-dispatch adapter
  repair-engine/    deterministic failure classifier, fingerprint, repair policy/contracts
plugins/
  kaggle/           Kaggle provider implementation and GitHub Actions-side commands
.github/workflows/
  kaggle-auth-check.yml
  kaggle-inventory.yml
  kaggle-issue-dispatch.yml
  kaggle-issue-status.yml
  ci.yml
docs/
  architecture-v0.1.md
  security-v0.1.md
  setup-v0.1.md
```

## Fast setup

See [`docs/setup-v0.1.md`](docs/setup-v0.1.md). The essential steps are:

1. Create a private GitHub repository named `chatgpt-plugins` and push this source.
2. Configure `plugins/kaggle/config/accounts.json` with public account metadata only.
3. Create one GitHub Environment per account (`kaggle-01`, `kaggle-02`, ...).
4. Generate a current Kaggle API token for each account and put it in that environment as
   `KAGGLE_API_TOKEN`. Never commit or paste tokens into ChatGPT.
5. Open a `[KAGGLE-AUTH]` Issue and require `AUTH_OK` before any new compute is submitted.
6. If recovering existing work, open `[KAGGLE-INVENTORY] <search-term>` before creating new jobs.
7. If target source repositories are private, add a narrowly scoped read-only
   `SOURCE_GITHUB_TOKEN` repository secret.
8. Open a `[KAGGLE-JOB]` Issue only after account authentication is healthy.
9. Ask ChatGPT to check active jobs; ChatGPT can comment `/kaggle status` through GitHub.

## Development

With dependencies available:

```bash
uv sync --all-packages --dev
uv run pytest
uv run ruff check .
```

The repository has no requirement for a long-running service in V0.1.

## Safety boundary

Multiple accounts are supported only as accounts the operator is authorized to use and must not
be used to evade Kaggle quotas, restrictions, or terms. The scheduler/configuration model is
quota-aware by design, but V0.1 intentionally does not guess or hard-code Kaggle quota numbers.
