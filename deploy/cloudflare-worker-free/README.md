# Cloudflare Workers Free — Kaggle Gateway

This is the active zero-cost V0.1 production runtime.

## Runtime

```text
ChatGPT -> MCP -> Cloudflare Worker Free -> direct Kaggle HTTPS API
ChatGPT -> GitHub Issue -> signed webhook -> Worker Free -> direct Kaggle HTTPS API
```

There is no Cloudflare Container, Render service, Kaggle CLI, or user-PC runtime.

## Deployment

The Worker is deployed from GitHub by `.github/workflows/cloudflare-worker-free-deploy.yml` using
only `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` from the `cloudflare-production`
environment. Kaggle credentials are never passed through GitHub Actions.

## Required Worker secrets for read/recovery

Set these directly on the deployed Worker:

```text
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_TOKEN
CGP_MCP_PATH_SECRET
```

Usernames/owner slugs are public registry metadata compiled into `src/kaggle.ts`.

The MCP endpoint is:

```text
https://chatgpt-kaggle-gateway.<workers-dev-subdomain>.workers.dev/mcp/<sha256(CGP_MCP_PATH_SECRET)>
```

The secret itself must not appear in Git, ChatGPT, Issues, or Actions logs.

## Optional later write bridge

Write execution is frozen off in `wrangler.jsonc` with:

```text
CGP_WRITE_ENABLED=0
```

After existing-run recovery is complete, enabling `rerun_existing` additionally requires Worker
Secrets:

```text
CGP_GITHUB_WEBHOOK_SECRET
CGP_GITHUB_TOKEN
```

and a GitHub repository webhook pointing to:

```text
https://chatgpt-kaggle-gateway.<workers-dev-subdomain>.workers.dev/github/webhook
```

with `Issues` events only and the same webhook secret.

## Kaggle API transport

The Worker mirrors the official Kaggle SDK transport:

```text
POST https://api.kaggle.com/v1/kernels.KernelsApiService/<Method>
Authorization: Basic base64(username:api-key)
Content-Type: application/json
```

Read methods used in V0.1:

```text
ListKernels
GetKernelSessionStatus
ListKernelSessionOutput
GetKernel
```

The disabled write path uses `SaveKernel` with `SAVE_AND_RUN_ALL` only after strict Issue/webhook,
account-owner, and idempotency checks.
