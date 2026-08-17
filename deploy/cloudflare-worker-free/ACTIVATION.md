# Zero-cost activation runbook — Cloudflare Workers Free

Production read/recovery access runs through one Cloudflare Worker using six execution accounts plus
one separate read-only Master account. Kaggle writes remain disabled until recovery evidence is
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
  "master_configured": true,
  "write_enabled": false
}
```

## Kaggle account topology

The execution pool contains exactly six accounts:

```text
kg-02  radlinaradlina
kg-03  rezanory
kg-04  reyhanehazad
kg-05  trickermark
kg-06  msdenis
kg-07  nisabulutmark
```

The separate Master account is:

```text
master  azadka  read-only
```

Master is not part of six-account parallel execution and the Worker refuses `rerunExisting` for the
Master role.

## Cloudflare runtime secrets

These bindings exist only as Cloudflare Worker Secrets:

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

Normal code deployments do not receive or rewrite these values. KGAT tokens are sent to the direct
Kaggle API using Bearer authentication; legacy username/key credentials remain only as a compatibility
fallback in the client.

## Readiness gate

`/healthz` must report:

```text
status=ready
mcp_configured=true
kaggle_tokens_configured=6
master_configured=true
write_enabled=false
```

The Worker tools then verify the exact identities:

```text
kaggle_auth_check_all()       -> kg-02..kg-07
kaggle_master_auth_check()    -> master / azadka
```

## ChatGPT MCP endpoint

The private endpoint is the live Worker base URL plus the Cloudflare-only capability token:

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/mcp/<CGP_MCP_PATH_TOKEN>
```

## Proven recovery state

The live recovery completed successfully through the direct HTTPS path. TRAIN W01 through W06 were
read without new compute, their latest sessions reported `COMPLETE`, and the expected source
fingerprint was found in the bounded log tails:

```text
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

W01 is owned by Master `azadka`; W02-W06 are owned by execution accounts. The literal
`KAGGLE_EXECUTION_V62_2` was not present in the checked log tails, so the fingerprint is the confirmed
integrity signal for this recovered batch.

## Write remains frozen

Canonical production config remains:

```text
CGP_WRITE_ENABLED=0
```

Do not enable the signed write bridge until the recovered runs are classified for the next action.
