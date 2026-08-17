# Zero-cost activation runbook — Cloudflare Workers Free

This runbook activates only read/recovery access. Do not enable Kaggle writes until the existing
runs have been inventoried and classified.

## Live Worker

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev
```

Public readiness endpoint:

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/healthz
```

Expected before secrets are configured:

```json
{
  "service": "chatgpt-kaggle-gateway",
  "transport": "cloudflare-workers-free",
  "status": "ready",
  "mcp_configured": false,
  "kaggle_tokens_configured": 0,
  "write_enabled": false
}
```

## Add Worker secrets directly in Cloudflare

Open Cloudflare Dashboard, then:

```text
Workers & Pages
  -> chatgpt-kaggle-gateway
  -> Settings
  -> Variables and Secrets
  -> Add
  -> Type: Secret
```

Add exactly these seven read/recovery secrets:

```text
CGP_KAGGLE_KG01_TOKEN   # azadka
CGP_KAGGLE_KG02_TOKEN   # radlinaradlina
CGP_KAGGLE_KG04_TOKEN   # reyhanehazad
CGP_KAGGLE_KG05_TOKEN   # trickermark
CGP_KAGGLE_KG06_TOKEN   # msdenis
CGP_KAGGLE_KG07_TOKEN   # nisabulutmark
CGP_MCP_PATH_TOKEN      # private capability path token
```

Never put these values in GitHub Secrets, Git, Issues, Actions logs, or ChatGPT.

`CGP_MCP_PATH_TOKEN` must be a newly generated random URL-safe secret of 32–128 characters containing
only letters, digits, `_`, and `-`. Keep it private. The Worker validates this format and does not
expose the value from `/healthz`.

After all seven entries are present, select `Deploy` in the Cloudflare Variables and Secrets UI.
Code redeploys preserve existing Worker secrets.

## Readiness gate

After the secrets deployment, `/healthz` must report:

```text
status=ready
mcp_configured=true
kaggle_tokens_configured=6
write_enabled=false
```

Do not proceed to ChatGPT MCP setup if the count is not exactly 6.

## ChatGPT MCP endpoint

The endpoint is the live Worker base URL plus the private path token:

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/mcp/<CGP_MCP_PATH_TOKEN>
```

Enter this endpoint directly in ChatGPT's custom MCP/app configuration. Do not paste the endpoint
with its private capability token into a chat message or GitHub Issue.

## First live recovery sequence

Once the MCP tools appear in ChatGPT, run only read/recovery operations first:

```text
1. kaggle_auth_check_all(max_workers=6)
2. kaggle_kernels_inventory_all(search="pneumonia-v6-2-2", page_size=20, max_workers=6)
3. resolve exact existing shard kernel refs
4. kaggle_kernel_status(...)
5. kaggle_kernel_logs(...)
6. kaggle_kernel_output_manifest(...)
7. verify KAGGLE_EXECUTION_V62_2
8. verify fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

## Write remains frozen

Canonical production config remains:

```text
CGP_WRITE_ENABLED=0
```

Do not add the later write-bridge credentials and do not flip this flag until the existing runs are
classified. The later bridge requires a signed GitHub webhook plus `CGP_GITHUB_WEBHOOK_SECRET` and
`CGP_GITHUB_TOKEN` and supports only the narrow `rerun_existing` command in V0.1.
