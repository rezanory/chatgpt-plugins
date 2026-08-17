# Existing Kaggle Run Recovery — V0.1

This path is for Kaggle kernels that already existed before `chatgpt-plugins` took over orchestration.
It is intentionally **read-only**: it never submits, restarts, cancels, edits, or deletes a Kaggle run.

## Preconditions

1. Each enabled account has a current Kaggle token in its matching GitHub Environment as
   `KAGGLE_API_TOKEN`.
2. A `[KAGGLE-AUTH]` check reports `AUTH_OK` for every account that will be queried.
3. `plugins/kaggle/config/accounts.json` contains the correct public Kaggle owner slug.

## Step 1 — discover existing kernels

Open an Issue such as:

```text
[KAGGLE-INVENTORY] pneumonia-v6-2-2
```

For each enabled account the control plane calls the official token-authenticated equivalent of:

```text
kaggle kernels list -m -s pneumonia-v6-2-2
```

No compute is submitted. The sanitized output is posted back to the Issue so ChatGPT can identify
exact `owner/kernel-slug` references.

## Step 2 — recover status and selected outputs

After the exact kernel refs are known, open an Issue whose title starts:

```text
[KAGGLE-RECOVER]
```

Example envelope:

````markdown
<!-- chatgpt-plugins-recovery:v1 -->
```json
{
  "schema": "chatgpt.compute.recovery/v1",
  "recovery_id": "pneumonia-v62-hpo-recovery",
  "artifact_names": [
    "KAGGLE_EXECUTION_V62_2"
  ],
  "runs": [
    {
      "task_id": "s01",
      "account_id": "kg-01",
      "kernel_ref": "azadka/exact-kernel-slug-from-inventory"
    },
    {
      "task_id": "s02",
      "account_id": "kg-02",
      "kernel_ref": "radlinaradlina/exact-kernel-slug-from-inventory"
    }
  ]
}
```
````

The real six-shard recovery may contain up to 16 runs, but each task and account ID must be unique.

## Security and ownership checks

The planner rejects a run when:

- the account is disabled or unknown;
- the kernel ref is malformed;
- the kernel owner does not match the owner registered for that account;
- a task/account is duplicated;
- an artifact name contains unsafe characters.

Only the GitHub Environment belonging to that account is loaded. A token from one account is not
used to inspect another account's kernel.

## Evidence

For terminal kernels, only the requested allow-listed output names are downloaded. The evidence
bundle contains a `manifest.json` with SHA-256 for every downloaded file. The control plane can
therefore compare a known expected fingerprint/hash against retrieved evidence without trusting a
filename or raw log message.

For the current pneumonia recovery, the known fingerprint is:

```text
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

Do not assume this hash is a filename. First discover actual output filenames, then compare the
manifest hashes/content evidence to the expected fingerprint.

## Rule before new compute

Existing HPO/confirmation/merge/training work must be inventoried and assessed first. A new Kaggle
submission is a separate decision and is not performed by the recovery workflow.
