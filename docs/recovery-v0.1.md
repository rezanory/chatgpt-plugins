# Existing Kaggle Run Recovery — V0.1 Direct API

This path is for Kaggle kernels that already existed before the direct gateway took over
orchestration. It is intentionally **read-only**: it never submits, restarts, cancels, edits, or
deletes a Kaggle run.

## Preconditions

1. Each enabled logical account has a working username/token pair in the gateway process secret
   environment.
2. The gateway authenticates each account with the proven `KaggleApi` sequence.
3. `kaggle_auth_check_all(max_workers=6)` reports `auth_ok=true` for intended accounts.
4. `plugins/kaggle-gateway/.../accounts.json` contains the correct public owner slug.

## Step 1 — discover existing kernels in parallel

From ChatGPT call:

```text
kaggle_kernels_inventory_all(
  search="pneumonia-v6-2-2",
  page_size=20,
  max_workers=6
)
```

For each enabled account the gateway directly calls that account's isolated:

```python
api.kernels_list(
    search="pneumonia-v6-2-2",
    page_size=20,
    mine=True,
    sort_by="dateRun",
)
```

No CLI, browser session, GitHub Action, or new compute is involved.

## Step 2 — resolve exact owner/kernel refs

Inventory results are grouped by logical account. For each candidate, retain the exact
`owner/kernel-slug`. The gateway will reject later kernel-specific operations if the owner does not
match the selected account's configured `owner_slug`.

## Step 3 — read status and logs

For each exact existing run:

```text
kaggle_kernel_status(account_id, kernel_ref)
kaggle_kernel_logs(account_id, kernel_ref)
```

These map directly to:

```python
api.kernels_status(kernel_ref)
api.kernels_logs(kernel_ref)
```

Returned logs are bounded and all configured gateway credentials are redacted before they can reach
ChatGPT.

## Step 4 — selected output recovery

The internal `KaggleApiPool.kernels_output()` method directly supports:

```python
api.kernels_output(
    kernel_ref,
    path=output_path,
    file_pattern=allowlisted_pattern,
    force=False,
    quiet=True,
)
```

Output download is not yet exposed as a public read tool until the gateway deployment has a defined
private artifact directory/storage policy. When enabled, it must keep the same direct-API path and
must never route through CLI or GitHub Actions.

## Known evidence target

For the current pneumonia recovery, the expected fingerprint is:

```text
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

and a known artifact identifier is:

```text
KAGGLE_EXECUTION_V62_2
```

Do not assume the fingerprint is a filename. First inventory exact runs/files, then compare actual
retrieved evidence/hash data against the expected fingerprint.

## Rule before new compute

Existing HPO/confirmation/merge/training work must be inventoried and assessed first. New Kaggle
submission is a separate later write decision and is not part of read-only recovery.
