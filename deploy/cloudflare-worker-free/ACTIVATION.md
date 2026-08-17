# Zero-cost activation runbook — Cloudflare Workers Free

This runbook activates read/recovery access through one Cloudflare Worker using six execution
accounts plus one separate Master account. Kaggle writes remain disabled until recovery evidence is
classified.

## Live Worker

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev
```

Public readiness endpoint:

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/healthz
```

Healthy production state:

```json
{
  "service": "chatgpt-kaggle-gateway",
  "transport": "cloudflare-workers-free",
  "status": "ready",
  "mcp_configured": true,
  "kaggle_tokens_configured": 6,
  "write_enabled": false
}
```

## Kaggle account topology

The Worker execution pool contains exactly six accounts:

```text
kg-02  radlinaradlina
kg-03  rezanory
kg-04  reyhanehazad
kg-05  trickermark
kg-06  msdenis
kg-07  nisabulutmark
```

`azadka` is the separate Master account and is not part of the six-account parallel Worker pool.

## Cloudflare runtime secrets

The Worker execution bindings are:

```text
CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG03_TOKEN
CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_TOKEN
CGP_KAGGLE_MASTER_TOKEN
CGP_MCP_PATH_TOKEN
```

KGAT tokens are sent to the direct Kaggle API using Bearer authentication. Legacy credentials remain
supported only as a compatibility fallback in the client.

`CGP_MCP_PATH_TOKEN` is the private capability path used for MCP and protected recovery endpoints.
The Worker validates its format and does not expose its value from `/healthz`.

## Readiness gate

Before recovery checks proceed, `/healthz` must report:

```text
status=ready
mcp_configured=true
kaggle_tokens_configured=6
write_enabled=false
```

The protected auth probe must also report these six exact account/owner pairs:

```text
kg-02 -> radlinaradlina
kg-03 -> rezanory
kg-04 -> reyhanehazad
kg-05 -> trickermark
kg-06 -> msdenis
kg-07 -> nisabulutmark
```

The Master probe is checked separately and must authenticate as the `azadka` account.

## ChatGPT MCP endpoint

The endpoint is the live Worker base URL plus the private path token:

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/mcp/<CGP_MCP_PATH_TOKEN>
```

## Live recovery sequence

Read/recovery checks are performed in this order:

```text
1. kaggle_auth_check_all(max_workers=6)
2. kaggle_kernels_inventory_all(search="pneumonia-v6-2-2", page_size=20, max_workers=6)
3. resolve exact latest matching kernel ref for each worker account
4. kaggle_kernel_status(...)
5. kaggle_kernel_logs(...), retaining the tail of long logs
6. kaggle_kernel_output_manifest(...), including bounded available output filenames
7. check KAGGLE_EXECUTION_V62_2
8. check fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

Live evidence is published in GitHub Issue #17 by the deployment/recovery workflows without
including provider credentials.

## Write remains frozen

Canonical production config remains:

```text
CGP_WRITE_ENABLED=0
```

Do not enable the signed write bridge until the existing runs and evidence have been classified.
