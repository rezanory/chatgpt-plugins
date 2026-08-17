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

## 3. GitHub Environments

Create environments whose names exactly match `secret_scope`:

- `kaggle-01`
- `kaggle-02`
- ...

Inside each environment set either:

- `KAGGLE_API_TOKEN` (current token flow), or
- legacy `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Do not expose these values to ChatGPT or commit them.

## 4. Private source repositories

The control repository's normal `GITHUB_TOKEN` is repository-scoped. If jobs need to checkout a
different **private** repository, add `SOURCE_GITHUB_TOKEN` as a control-repository secret. Use a
fine-grained token with read-only Contents access limited to the intended source repositories.

Public target repositories do not need a broad token.

## 5. Execution profiles

Profiles live in `plugins/kaggle/config/profiles.json` and are trusted configuration.

A profile uses argv, not shell strings:

```json
{
  "python-tests": {
    "capabilities": ["cpu"],
    "internet": false,
    "steps": [
      {"argv": ["python", "-m", "pytest", "-q"], "cwd": "."}
    ]
  }
}
```

Do not put secrets in a profile.

## 6. Open a job Issue from ChatGPT

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

The `issues: opened` workflow validates the envelope before any Kaggle environment secret is
loaded.

## 7. Check status from ChatGPT

Post exactly:

```text
/kaggle status
```

The status workflow resolves machine-readable run records already present on the Issue and loads
only the corresponding account environment.

For terminal tasks it uploads sanitized evidence as a GitHub Actions artifact and posts the
failure category/fingerprint (when applicable) to the Issue.

## 8. Repair

A failed source-level task is analyzed in ChatGPT. Any repair must target a new commit. Re-run by
opening a new job Issue that references the new commit. Do not rewrite or delete the old run record.
