# Implementation Status — V0.1

## Canonical runtime

```text
READ / RECOVERY
ChatGPT Pro -> custom MCP -> Cloudflare Workers Free -> direct Kaggle HTTPS API -> Kaggle

WRITE / EXECUTION (implemented but frozen off)
ChatGPT -> connected GitHub app -> [KAGGLE-RUN] Issue -> signed GitHub webhook
        -> Cloudflare Workers Free -> direct Kaggle HTTPS API -> Kaggle
        -> Issue receipt/failure -> ChatGPT
```

The user's PC is not a runtime. Render, Cloudflare Containers, Kaggle CLI, browser sessions, and
operational Kaggle GitHub Actions are not part of production.

## Live deployment

The zero-cost Worker is deployed and healthy:

```text
Worker:  chatgpt-kaggle-gateway
URL:     https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev
Health:  /healthz PASS
Deploy:  cloudflare-worker-free-deploy run #3, SUCCESS
Run ID:  32069572874
Version: afec9485-3674-4611-8820-526dcd07a55c
```

Later deployments, including documentation/config hardening, also completed successfully.

## Direct Kaggle transport

The active Worker mirrors the official Kaggle SDK transport:

```text
POST https://api.kaggle.com/v1/kernels.KernelsApiService/<Method>
Authorization: Basic base64(username:api-key)
Content-Type: application/json
```

Enabled logical accounts:

```text
kg-01 -> azadka
kg-02 -> radlinaradlina
kg-04 -> reyhanehazad
kg-05 -> trickermark
kg-06 -> msdenis
kg-07 -> nisabulutmark
```

`kg-03` remains disabled pending canonical owner-slug resolution.

## Read surface

```text
kaggle_accounts
kaggle_auth_check
kaggle_auth_check_all
kaggle_kernels_list
kaggle_kernels_inventory_all
kaggle_kernel_status
kaggle_kernel_logs
kaggle_kernel_output_manifest
```

The Worker package has passed TypeScript validation and `wrangler deploy --dry-run` and has been
published successfully.

## Write surface

`rerun_existing` is implemented through the signed GitHub webhook control plane but is deliberately
frozen off with `CGP_WRITE_ENABLED=0` until existing-run recovery is complete. The Worker validates
the repository, actor, strict control schema, account-owner match, HMAC signature, and persistent
GitHub `job_id` claim marker before any write. The provider write is `GetKernel` followed by
`SaveKernel` with `SAVE_AND_RUN_ALL`.

## Current activation blocker

The Worker itself is live. Kaggle connectivity is not yet live because runtime secrets have not yet
been configured directly on the Cloudflare Worker. Required for read/recovery:

```text
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_TOKEN
CGP_MCP_PATH_TOKEN
```

`CGP_MCP_PATH_TOKEN` must be a random URL-safe 32–128 character value using only letters, numbers,
`_`, and `-`. It is used directly as `/mcp/<token>` and is entered only in Cloudflare and the
ChatGPT custom-app endpoint. These values must never be placed in GitHub Actions, commits, Issues,
or conversation text.

## Recovery sequence

After the seven read/recovery secrets are configured and the custom MCP is connected:

```text
1. kaggle_auth_check_all(max_workers=6)
2. kaggle_kernels_inventory_all(search="pneumonia-v6-2-2", page_size=20, max_workers=6)
3. resolve the six existing shard kernels
4. status + logs
5. selected output manifest
6. verify KAGGLE_EXECUTION_V62_2
7. verify fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

No new Kaggle compute is allowed before those existing runs are classified.
