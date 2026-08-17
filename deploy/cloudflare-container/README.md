# Cloudflare Remote Runtime — Kaggle Direct Gateway

This is the **active production deployment path** for V0.1.

```text
GitHub: rezanory/chatgpt-plugins
  -> Cloudflare Workers Builds (Git integration)
  -> Cloudflare Worker + Access Managed OAuth
  -> Cloudflare Container
  -> Python Kaggle Direct Gateway
  -> one isolated KaggleApi per enabled account
  -> Kaggle API
```

No user PC, Kaggle CLI, browser session, or GitHub Actions runtime is required for Kaggle operations.

## 1. Import the GitHub repository into Cloudflare

In Cloudflare Dashboard:

```text
Workers & Pages
  -> Create application
  -> Import a repository
  -> GitHub
  -> rezanory/chatgpt-plugins
```

Use these build settings:

```text
Worker name:   chatgpt-kaggle-gateway
Production branch: main
Root directory: deploy/cloudflare-container
Build command:  (leave empty; dependency install is automatic)
Deploy command: npx wrangler deploy
```

The Worker name must match `name` in `wrangler.jsonc`.

`wrangler.jsonc` sets `image_build_context` to the repository root so the Docker build can copy
`plugins/kaggle-gateway` into the Python image while the Worker project itself remains isolated in
this monorepo directory.

## 2. Configure runtime Kaggle secrets in Cloudflare

Under the Worker:

```text
Settings -> Variables and Secrets
```

Add the following as **Secret** values, not plaintext variables:

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

These values are passed at runtime from the Worker secret environment into the Container. They are
never present in Git or the Docker image.

Do not configure global `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, or `KAGGLE_KEY`. The Python gateway
intentionally rejects those global overrides to preserve per-account isolation.

## 3. Protect the MCP endpoint with Cloudflare Access Managed OAuth

The Worker deliberately refuses `/mcp` unless Cloudflare Access is configured.

Protect the Worker hostname with Cloudflare Access and create an allow policy for the intended user.
For a single-user deployment, an email allow policy is sufficient.

Enable **Managed OAuth** for the Access application so MCP clients can complete a standard OAuth
flow instead of receiving a browser-only redirect.

Then obtain:

```text
TEAM_DOMAIN = https://<team-name>.cloudflareaccess.com
POLICY_AUD  = <Access application audience tag>
```

Add both to the Worker under **Variables and Secrets**. They are configuration values, not Kaggle
credentials.

The Worker validates every `Cf-Access-Jwt-Assertion` cryptographically against the team-domain
JWKS, expected issuer, and expected audience before forwarding a request to the Python Container.

Do not disable this validation in production.

## 4. Expected remote endpoints

After deployment:

```text
GET  https://<worker-host>/healthz
MCP  https://<worker-host>/mcp
```

`/healthz` returns only generic service readiness and never touches Kaggle credentials.

`/mcp` is protected by Cloudflare Access and proxies to a single Cloudflare Container instance.

## 5. Connect ChatGPT Pro

ChatGPT Pro supports custom MCP apps with read/fetch permissions in Developer Mode.

Create a custom app using:

```text
MCP endpoint: https://<worker-host>/mcp
Authentication: OAuth
```

Complete the Cloudflare Access OAuth login and run **Scan Tools**.

Expected read-only tools:

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

## 6. First live sequence

Do not submit new compute first.

```text
1. kaggle_auth_check_all(max_workers=6)
2. kaggle_kernels_inventory_all(
     search="pneumonia-v6-2-2",
     page_size=20,
     max_workers=6
   )
3. Resolve the exact existing kernel refs for the six shards.
4. kaggle_kernel_status(account_id, kernel_ref)
5. kaggle_kernel_logs(account_id, kernel_ref)
6. kaggle_kernel_output_manifest(
     account_id,
     kernel_ref,
     ["KAGGLE_EXECUTION_V62_2", "fingerprint"],
     "fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838"
   )
```

Only after existing work is inventoried and classified should write/submission tools be introduced.

## 7. Continuous deployment

Cloudflare Workers Builds is connected directly to GitHub. A push to the configured production
branch triggers Cloudflare's own build/deploy pipeline. GitHub Actions remains a source-quality CI
only and never carries Kaggle runtime credentials.

## Security invariants

- Kaggle operations use `KaggleApi()` directly inside the Python Container.
- One isolated KaggleApi/config file per account.
- No Kaggle CLI runtime.
- No GitHub Actions Kaggle runtime.
- No credential values committed to Git.
- Cloudflare Access Managed OAuth protects `/mcp`.
- Worker validates the Access JWT before proxying to the Container.
- The Access JWT/cookie is stripped before the request reaches Python.
- Container has outbound Internet access only because Kaggle API access requires it.
