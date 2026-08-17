# V0.1 Remote Kaggle Gateway Setup

## 1. Repository

Keep `rezanory/chatgpt-plugins` private. GitHub is source control and source-quality CI only.
GitHub Actions must never authenticate to or operate Kaggle.

## 2. Active production components

```text
plugins/kaggle-gateway/         Python direct KaggleApi + MCP service
deploy/cloudflare-container/    Cloudflare Worker + Container deployment
```

The user's PC is not part of the production runtime.

## 3. Account registry

Public/non-secret account metadata is stored in:

```text
plugins/kaggle-gateway/src/chatgpt_plugin_kaggle_gateway/accounts.json
```

Each entry maps a logical account to Cloudflare runtime secret-reference names. Never place real
credential values in this JSON.

Enabled accounts are currently:

```text
kg-01
kg-02
kg-04
kg-05
kg-06
kg-07
```

`kg-03` remains disabled until its canonical Kaggle owner slug is resolved.

## 4. One-time Cloudflare project connection

Import the GitHub repository into Cloudflare Workers Builds and use:

```text
Repository:        rezanory/chatgpt-plugins
Production branch: main
Root directory:   deploy/cloudflare-container
Deploy command:   npx wrangler deploy
Worker name:      chatgpt-kaggle-gateway
```

Cloudflare becomes the production runtime/deployment owner. Subsequent source changes are delivered
from GitHub without requiring the user's PC.

The deployment package is already CI-validated with:

```text
wrangler types
TypeScript check
wrangler deploy --dry-run
Docker image build
```

## 5. Configure Kaggle credentials as Cloudflare Secrets

In the Cloudflare Worker configuration, add these as **Secret** values:

```text
CGP_KAGGLE_KG01_USERNAME
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_USERNAME
CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG04_USERNAME
CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_USERNAME
CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_USERNAME
CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_USERNAME
CGP_KAGGLE_KG07_TOKEN
```

Do not commit them, place them in Issues, or paste them into ChatGPT.

Do not add process-global `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, or `KAGGLE_KEY`; the Python gateway
rejects those variables because they can override isolated account config.

## 6. Direct Kaggle authentication behavior

Inside the remote Cloudflare Container, each account follows exactly:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

Every account receives a dedicated temporary config directory and `KaggleApi` client.

## 7. Configure Cloudflare Access for MCP

Protect the Worker MCP endpoint with Cloudflare Access Managed OAuth.

Configure:

```text
TEAM_DOMAIN = https://<team-name>.cloudflareaccess.com
POLICY_AUD  = <Access application audience tag>
```

The Worker requires and cryptographically validates `Cf-Access-Jwt-Assertion` before forwarding any
`/mcp` request to the Container. The Access JWT and cookie are stripped before forwarding.

Production endpoints:

```text
GET https://<worker-host>/healthz
MCP https://<worker-host>/mcp
```

`/healthz` is generic and does not touch Kaggle credentials.

## 8. Connect ChatGPT

Create the custom MCP app using the deployed Cloudflare URL:

```text
Endpoint:       https://<worker-host>/mcp
Authentication: OAuth
```

After OAuth, scan tools. Expected initial surface:

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

## 9. First live verification

Run:

```text
kaggle_auth_check_all(max_workers=6)
```

All intended accounts must return `auth_ok=true` before recovery proceeds.

## 10. Recover existing work before new compute

```text
kaggle_kernels_inventory_all(
  search="pneumonia-v6-2-2",
  page_size=20,
  max_workers=6
)
```

For each exact existing `owner/kernel`:

```text
kaggle_kernel_status(account_id, kernel_ref)
kaggle_kernel_logs(account_id, kernel_ref)
kaggle_kernel_output_manifest(...)
```

Known evidence target:

```text
artifact: KAGGLE_EXECUTION_V62_2
fingerprint: fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

Do not submit/restart new compute until the existing six runs are classified.

## 11. CI boundary

`.github/workflows/ci.yml` may validate code and the Cloudflare deployment package. It contains no
Kaggle credential and makes no operational Kaggle API call.

## 12. After recovery

Direct write/submission tools may later be added to `KaggleApiPool`, but they must remain direct
Python API calls, be explicitly classified as writes, and use bounded retry/repair policy.
