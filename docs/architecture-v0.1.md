# Architecture Freeze — V0.1 Kaggle Gateway

## Decision

Production is remote, zero-cost, and does not use the user's PC:

```text
READ
ChatGPT Pro -> custom MCP -> Cloudflare Worker Free -> direct Kaggle HTTPS API -> Kaggle

WRITE (frozen until recovery completes)
ChatGPT -> GitHub connector -> signed control Issue/webhook -> Cloudflare Worker Free
        -> direct Kaggle HTTPS API -> Kaggle -> GitHub receipt -> ChatGPT
```

GitHub Actions are source validation/deployment only. They never receive Kaggle runtime credentials
and never make operational Kaggle calls.

## Active production package

```text
deploy/cloudflare-worker-free/
```

The Worker is plain Workers Free. There are no active Container, Durable Object, queue, VM, Render,
or paid runtime bindings. A historical `deleted_classes` migration remains only to record cleanup of
the abandoned `KaggleGatewayContainer` Durable Object class.

Live Worker:

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev
```

## Kaggle protocol boundary

The active Worker mirrors the official Kaggle SDK HTTP contract:

```text
POST https://api.kaggle.com/v1/kernels.KernelsApiService/<Method>
JSON camelCase request bodies
legacy username/API-key HTTP Basic Auth per account
```

The public account registry maps logical IDs to owner slugs; only API keys are runtime secrets.
Every kernel-specific operation requires the `owner/slug` to match the selected account.

## Read/recovery boundary

The MCP surface is deliberately read-only:

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

A random URL-safe `CGP_MCP_PATH_TOKEN` is used directly as `/mcp/<token>`. Without a valid 32–128
character token, no MCP route is exposed. The token is stored only as a Cloudflare Worker Secret and
in the ChatGPT custom-app endpoint configuration, not in source or conversation text.

Output recovery is bounded by allowlisted names, file-count and byte budgets, HTTPS host allowlists,
SHA-256 hashing, and bounded fingerprint scanning.

## Write/control boundary

V0.1 contains one narrow write action, `rerun_existing`, but the canonical configuration keeps it
disabled with `CGP_WRITE_ENABLED=0` until existing work is recovered.

When later enabled, GitHub webhook requests require:

- valid `X-Hub-Signature-256` HMAC;
- exact repository match;
- allowlisted Issue actor;
- `[KAGGLE-RUN]` title and strict v1 JSON schema;
- selected account/owner match;
- persistent GitHub claim marker for duplicate prevention.

The write provider path is `GetKernel` followed by `SaveKernel` with
`kernelExecutionType=SAVE_AND_RUN_ALL`, preserving source and relevant metadata.

## Secret boundary

Read/recovery Worker Secrets:

```text
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_TOKEN
CGP_MCP_PATH_TOKEN
```

Later write control additionally requires `CGP_GITHUB_WEBHOOK_SECRET` and a repository-scoped
`CGP_GITHUB_TOKEN` for the persistent Issue journal. None of these values belong in GitHub Actions,
commits, Issues, or conversation text.

## Existing-run recovery gate

Before enabling any write:

```text
auth six accounts
-> inventory pneumonia-v6-2-2
-> resolve exact existing shard refs
-> status/logs
-> selected output manifests
-> KAGGLE_EXECUTION_V62_2
-> fingerprint fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

## Explicit exclusions

- user-PC runtime or tunnels;
- Render;
- paid Cloudflare Containers;
- Cloudflare Zero Trust/Access dependency for V0.1 MCP activation;
- Kaggle CLI/subprocess execution;
- operational Kaggle GitHub Actions;
- browser/cookie authentication;
- automatic quota-evasion account rotation.
