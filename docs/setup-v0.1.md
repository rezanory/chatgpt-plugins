# V0.1 Setup

## 1. Control repository

Create a private repository named `chatgpt-plugins` and push this monorepo to its default branch.
Enable GitHub Issues and GitHub Actions.

## 2. Kaggle account registry

Edit `plugins/kaggle/config/accounts.json`.

Example for two authorized accounts:

```json
[
  {
    "account_id": "kg-01",
    "owner_slug": "kaggle_username_1",
    "secret_scope": "kaggle-01",
    "enabled": true,
    "max_parallel": 1,
    "capabilities": ["cpu", "gpu"],
    "default_accelerator": "NvidiaTeslaT4"
  },
  {
    "account_id": "kg-02",
    "owner_slug": "kaggle_username_2",
    "secret_scope": "kaggle-02",
    "enabled": true,
    "max_parallel": 1,
    "capabilities": ["cpu", "gpu"]
  }
]
```

No credential belongs in this file.

## 3. GitHub Environments and API tokens

Create environments whose names exactly match `secret_scope`:

- `kaggle-01`
- `kaggle-02`
- ...

For each Kaggle account:

1. Open Kaggle **Settings -> API**.
2. Use **Generate New Token**.
3. Store that token only in the matching GitHub Environment as:

```text
KAGGLE_API_TOKEN
```

V0.1 is token-only. Do not configure browser cookies, cached sessions, interactive OAuth login,
`KAGGLE_USERNAME`, `KAGGLE_KEY`, or a persisted `kaggle.json` as runtime dependencies.

Never expose token values to ChatGPT, Issues, source files, artifacts, or logs.

## 4. Validate every token before compute

Open a GitHub Issue with a title starting exactly:

```text
[KAGGLE-AUTH]
```

The auth workflow performs a harmless authenticated API read for every enabled account and posts
only sanitized `AUTH_OK` / `AUTH_FAILED` results.

Do not submit new compute until the intended accounts report `AUTH_OK`.

## 5. Recover existing Kaggle work before creating new runs

If workloads already existed before this control plane was installed, discover them first with:

```text
[KAGGLE-INVENTORY] <optional-search-term>
```

Example:

```text
[KAGGLE-INVENTORY] pneumonia-v6-2-2
```

The inventory workflow uses each enabled account's `KAGGLE_API_TOKEN` and calls the official
`kaggle kernels list -m` API path. It does **not** create datasets, kernels, or GPU runs.

Use the returned `owner/kernel-slug` references to assess existing run status/output before deciding
whether a new submission is necessary.

## 6. Private source repositories

The control repository's normal `GITHUB_TOKEN` is repository-scoped. If new jobs need to checkout
a different **private** repository, add `SOURCE_GITHUB_TOKEN` as a control-repository secret. Use a
fine-grained token with read-only Contents access limited to the intended source repositories.

Public target repositories do not need a broad token.

## 7. Execution profiles

Profiles live in `plugins/kaggle/config/profiles.json` and are trusted configuration.

A profile uses argv, not shell strings. `python-smoke` is dependency-free and is the preferred
first end-to-end submission after tokens are validated.

Do not put secrets in a profile.

## 8. Open a new job Issue from ChatGPT

Example two-account parallel batch body (the Issue title is `[KAGGLE-JOB] Example parallel validation`):

````markdown
<!-- chatgpt-plugins-job:v1 -->
```json
{
  "schema": "chatgpt.compute.job/v1",
  "job_id": "example-001",
  "source": {
    "repository": "owner/project",
    "commit": "0123456789abcdef0123456789abcdef01234567"
  },
  "tasks": [
    {
      "task_id": "tests-a",
      "profile": "python-tests",
      "account_id": "kg-01"
    },
    {
      "task_id": "tests-b",
      "profile": "python-tests",
      "account_id": "kg-02"
    }
  ],
  "repair_policy": {
    "enabled": true,
    "max_attempts": 2
  }
}
```
````

The `issues: opened` workflow validates the envelope before the selected Kaggle environment secret
is loaded.

## 9. Check status from ChatGPT

Post exactly:

```text
/kaggle status
```

The status workflow resolves machine-readable run records already present on the Issue and loads
only the corresponding account environment.

For terminal tasks it uploads sanitized evidence as a GitHub Actions artifact and posts the
failure category/fingerprint (when applicable) to the Issue.

## 10. Repair

A failed source-level task is analyzed in ChatGPT. Any repair must target a new commit. Re-run by
opening a new job Issue that references the new commit. Do not rewrite or delete the old run record.

Authentication, quota, network, and provider failures never trigger source-code repair.
