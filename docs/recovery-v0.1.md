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
`owner/kernel-slug`. The gateway rejects later kernel-specific operations if the owner does not
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

## Step 4 — selected output recovery and hashing

The public read-only tool is:

```text
kaggle_kernel_output_manifest(
  account_id,
  kernel_ref,
  artifact_names,
  expected_fingerprint="..."
)
```

It calls the direct API path:

```python
api.kernels_output(
    kernel_ref,
    path=temporary_output_path,
    file_pattern=escaped_allowlist_pattern,
    force=False,
    quiet=True,
)
```

The tool does **not** return raw downloaded files to ChatGPT. It:

1. accepts only safe literal artifact-name fragments;
2. escapes them before constructing Kaggle's filename-regex filter;
3. downloads only matching existing outputs into a fresh private temporary directory;
4. rejects symlinks/path escapes and bounds file count/total bytes;
5. computes SHA-256 for every recovered file;
6. optionally scans small files for the expected fingerprint value;
7. returns only the manifest and exact fingerprint-hit paths;
8. deletes the temporary local output before the MCP tool returns.

This keeps artifact recovery read-only while allowing integrity comparison from ChatGPT.

## Known evidence target

For the current pneumonia recovery, the expected fingerprint is:

```text
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

and a known artifact identifier is:

```text
KAGGLE_EXECUTION_V62_2
```

Example:

```text
kaggle_kernel_output_manifest(
  account_id="kg-01",
  kernel_ref="<exact-owner>/<exact-kernel>",
  artifact_names=["KAGGLE_EXECUTION_V62_2", "fingerprint"],
  expected_fingerprint="fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838"
)
```

Do not assume the fingerprint is a filename. Inventory exact runs first, then compare actual
manifest hashes/content hits against the expected fingerprint.

## Rule before new compute

Existing HPO/confirmation/merge/training work must be inventoried and assessed first. New Kaggle
submission is a separate later write decision and is not part of read-only recovery.
