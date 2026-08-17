# Existing Kaggle Run Recovery — V0.1 Direct API

This path is for Kaggle kernels that already existed before the Cloudflare direct gateway took over
orchestration. Recovery is intentionally **read-only**: it does not submit, restart, cancel, edit, or
delete Kaggle runs.

## Runtime topology

The direct Worker has six execution accounts:

```text
kg-02  radlinaradlina
kg-03  rezanory
kg-04  reyhanehazad
kg-05  trickermark
kg-06  msdenis
kg-07  nisabulutmark
```

A seventh Kaggle account is modeled separately as the read-only Master:

```text
master  azadka
```

All provider tokens and the private MCP path token live only in Cloudflare Worker Secrets.

## Step 1 — authenticate

From ChatGPT/MCP:

```text
kaggle_auth_check_all(max_workers=6)
kaggle_master_auth_check()
```

The six-account call validates only the execution pool. Master is checked separately and never joins
the parallel execution pool.

## Step 2 — discover existing kernels

```text
kaggle_kernels_inventory_all(
  search="pneumonia-v6-2-2",
  page_size=20,
  max_workers=6
)

kaggle_master_kernels_list(
  search="pneumonia-v6-2-2",
  page_size=20
)
```

The active Worker mirrors Kaggle's HTTPS `kernels.KernelsApiService` transport directly. No Kaggle
CLI, browser session, local PC, Render service, or container runtime is required.

## Step 3 — resolve exact refs and read status

For execution accounts:

```text
kaggle_kernel_status(account_id, kernel_ref)
kaggle_kernel_logs(account_id, kernel_ref)
```

For Master:

```text
kaggle_master_kernel_status(kernel_ref)
kaggle_master_kernel_logs(kernel_ref)
```

The gateway rejects kernel-specific operations when the owner in `owner/kernel-slug` does not match
the selected logical account. Long logs return a bounded tail so end-of-run evidence is retained.

## Step 4 — selected output evidence

Execution-account tool:

```text
kaggle_kernel_output_manifest(
  account_id,
  kernel_ref,
  artifact_names,
  expected_fingerprint="..."
)
```

Master tool:

```text
kaggle_master_kernel_output_manifest(
  kernel_ref,
  artifact_names,
  expected_fingerprint="..."
)
```

The Worker lists bounded output filenames, downloads only explicitly selected small files, computes
SHA-256, optionally scans selected files for an expected fingerprint, and returns metadata rather
than unrestricted raw output.

## Proven pneumonia recovery

The expected source fingerprint is:

```text
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

Recovered training shards:

```text
W01  azadka       / pneumonia-v6-2-2-train-w01-m01-r224   Master
W02  radlinaradlina / pneumonia-v6-2-2-train-w02-m01-r320
W03  reyhanehazad / pneumonia-v6-2-2-train-w03-m01-r384
W04  trickermark  / pneumonia-v6-2-2-train-w04-m02-r224
W05  msdenis      / pneumonia-v6-2-2-train-w05-m02-r320
W06  nisabulutmark / pneumonia-v6-2-2-train-w06-m02-r384
```

Live checks returned `COMPLETE` for W01-W06 and found the expected source fingerprint in each
relevant bounded log tail. The literal identifier `KAGGLE_EXECUTION_V62_2` was not present in the
checked log tails, so it is not used as the recovery acceptance signal.

`kg-03 / rezanory` is a valid execution account but its latest search result belongs to an older
`pneumonia-v62-deep-*` sequence rather than this W01-W06 training shard set.

## Rule before new compute

Existing HPO, confirmation, merge, and training results should be classified first. New Kaggle
submission is a separate write decision. Production remains:

```text
CGP_WRITE_ENABLED=0
```
