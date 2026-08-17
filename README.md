# chatgpt-plugins

V0.1 monorepo for operating external compute from a normal ChatGPT conversation without using the
user's PC as a runtime and without requiring paid hosting.

The first provider is **Kaggle**. The active production boundary is a plain **Cloudflare Worker on
Workers Free**, with no Container, VM, Render service, Kaggle CLI, or operational Kaggle GitHub
Action.

## Goal: operate Kaggle from this ChatGPT conversation

```text
READ / RECOVERY
ChatGPT
  -> custom MCP
  -> Cloudflare Worker Free
  -> direct Kaggle HTTPS API
  -> Kaggle

WRITE / EXECUTION
ChatGPT
  -> connected GitHub app creates a [KAGGLE-RUN] Issue
  -> signed GitHub repository webhook
  -> Cloudflare Worker Free
  -> direct Kaggle HTTPS API
  -> Kaggle
  -> claim/receipt/failure comment on the same Issue
  -> ChatGPT reads the result through the GitHub connector
```

GitHub Actions only validate and deploy the Worker source. They never receive Kaggle credentials
and never authenticate to or execute Kaggle.

## Direct Kaggle HTTP contract

The active Worker mirrors Kaggle's official SDK transport instead of running the Python package.
Production requests are JSON `POST`s to:

```text
https://api.kaggle.com/v1/kernels.KernelsApiService/<Method>
```

Each authorized logical account uses its own legacy Kaggle username/API-key pair through HTTP Basic
Auth. Credential values live only as Cloudflare Worker Secrets.

Enabled accounts:

```text
kg-01 -> azadka
kg-02 -> radlinaradlina
kg-04 -> reyhanehazad
kg-05 -> trickermark
kg-06 -> msdenis
kg-07 -> nisabulutmark
```

`kg-03` remains disabled until its canonical owner slug is resolved.

## MCP read/recovery surface

- `kaggle_accounts`
- `kaggle_auth_check`
- `kaggle_auth_check_all`
- `kaggle_kernels_list`
- `kaggle_kernels_inventory_all`
- `kaggle_kernel_status`
- `kaggle_kernel_logs`
- `kaggle_kernel_output_manifest`

The MCP endpoint is not exposed at a predictable `/mcp` URL. A URL-safe 32–128 character Worker
Secret named `CGP_MCP_PATH_TOKEN` becomes the private capability path `/mcp/<token>`. The same token
is entered only in Cloudflare and the ChatGPT custom-app endpoint; it must never be pasted into the
conversation, Git, Issues, or Actions logs. If the token is absent or malformed, MCP is disabled.

## Recovery before write

No new Kaggle compute should be submitted before the existing pneumonia runs are classified:

```text
1. kaggle_auth_check_all(max_workers=6)
2. kaggle_kernels_inventory_all(search="pneumonia-v6-2-2")
3. resolve the existing six shard kernels
4. status + logs
5. selected output manifest
6. verify KAGGLE_EXECUTION_V62_2
7. verify fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

For that reason `CGP_WRITE_ENABLED` is canonically frozen to `0` in `wrangler.jsonc`.

## Narrow write bridge

The code contains one V0.1 write action, `rerun_existing`, but it remains disabled until recovery is
complete. A valid control Issue must begin with `[KAGGLE-RUN]`, contain the strict v1 control marker
and JSON envelope, arrive through a valid GitHub webhook HMAC, come from the configured repository
and actor, and match the selected account's owner slug.

When later enabled, the Worker performs `GetKernel` followed by `SaveKernel` with
`kernelExecutionType: SAVE_AND_RUN_ALL`, preserving source and execution metadata. A claim comment
is persisted before submission so webhook redelivery cannot silently run the same `job_id` twice.

## Active deployment package

```text
deploy/cloudflare-worker-free/
  package.json
  wrangler.jsonc
  tsconfig.json
  src/index.ts
  src/kaggle.ts
  src/github.ts
```

`wrangler.jsonc` has `workers_dev: true`, disables preview URLs, and deliberately contains no active
Containers, Durable Objects, queues, or paid runtime bindings. Its historical migration only deletes
the abandoned `KaggleGatewayContainer` class.

Live deployment is performed by:

```text
.github/workflows/cloudflare-worker-free-deploy.yml
```

using only the existing Cloudflare deployment credentials:

```text
CLOUDFLARE_API_TOKEN
CLOUDFLARE_ACCOUNT_ID
```

Those are deployment credentials, not Kaggle credentials.

## Cloudflare runtime secrets

Before live Kaggle authentication, configure these Worker Secrets directly in Cloudflare:

```text
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_TOKEN
CGP_MCP_PATH_TOKEN
```

For the later write bridge also configure:

```text
CGP_GITHUB_WEBHOOK_SECRET
CGP_GITHUB_TOKEN
```

Never paste secret values into ChatGPT, Git commits, Issues, or Actions logs.

## Security boundaries

- user PC is not a runtime;
- no paid hosting dependency;
- no Kaggle credential is committed to Git;
- GitHub Actions never receive Kaggle runtime credentials;
- Kaggle CLI/subprocess runtime paths are forbidden;
- Render and Cloudflare Containers are forbidden by the security gate;
- the active Worker uses direct HTTPS only;
- kernel owner must match the selected logical account;
- write remains disabled until existing-run recovery is complete;
- webhook writes require HMAC, repository and actor validation plus durable GitHub claim markers;
- small output hashing is bounded by file count and byte budgets.

## Repository layout

```text
packages/                       provider-neutral contracts and support code
plugins/kaggle-gateway/         Python reference/test implementation retained for parity
plugins/kaggle/                 legacy migration/reference code
deploy/cloudflare-worker-free/  ACTIVE zero-cost production runtime
.github/workflows/ci.yml        source/package validation only
.github/workflows/cloudflare-worker-free-deploy.yml
```

Multiple accounts are supported only when the operator is authorized to use them. They must not be
rotated to evade Kaggle restrictions, quotas, or terms.
