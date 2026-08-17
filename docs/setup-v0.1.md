# V0.1 Kaggle Gateway Setup — Workers Free

## 1. Deployment state

The Worker is already deployed from `main`:

```text
https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev
```

`GET /healthz` has passed live validation. No paid runtime or user-PC service is required.

## 2. Active source

```text
deploy/cloudflare-worker-free/
.github/workflows/cloudflare-worker-free-deploy.yml
```

The deployment workflow uses only `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` from the
`cloudflare-production` GitHub environment. It does not receive Kaggle credentials.

## 3. Configure read/recovery secrets directly in Cloudflare

Open the deployed Worker `chatgpt-kaggle-gateway` in Cloudflare and add these values as **Secrets**,
not plain text variables:

```text
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_TOKEN
CGP_MCP_PATH_SECRET
```

The six public usernames are already compiled into the Worker registry. Do not add Kaggle secrets to
GitHub Actions, commits, Issues, or ChatGPT.

## 4. MCP path

The MCP route is derived as:

```text
/mcp/<sha256(CGP_MCP_PATH_SECRET)>
```

The root `/mcp` route is intentionally unavailable. The capability path must be treated as private.

## 5. Connect ChatGPT

After the secrets are configured, connect the Worker HTTPS endpoint plus the private MCP path as the
custom MCP server and scan tools. Expected surface:

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

## 6. First live calls

```text
kaggle_auth_check_all(max_workers=6)
kaggle_kernels_inventory_all(search="pneumonia-v6-2-2", page_size=20, max_workers=6)
```

Then resolve exact owner/kernel references and inspect status, logs and selected outputs.

Evidence target:

```text
KAGGLE_EXECUTION_V62_2
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

`CGP_WRITE_ENABLED` remains `0`; do not rerun anything before recovery is complete.

## 7. Later write bridge

Only after recovery, configure Worker Secrets:

```text
CGP_GITHUB_WEBHOOK_SECRET
CGP_GITHUB_TOKEN
```

and a repository webhook:

```text
Payload URL: https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/github/webhook
Content-Type: application/json
Secret: same CGP_GITHUB_WEBHOOK_SECRET
Events: Issues only
```

Then and only then can the canonical config be changed to `CGP_WRITE_ENABLED=1`. V0.1 permits only
`rerun_existing`; owner checks and persistent claim markers remain mandatory.
