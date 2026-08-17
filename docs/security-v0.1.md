# Security Model — V0.1 Kaggle Gateway

## Trust boundaries

1. **ChatGPT conversation** — requests work but never contains Kaggle credentials.
2. **Custom MCP capability path** — protects read/recovery access to the Worker.
3. **Cloudflare Worker Free** — active public runtime and direct Kaggle HTTPS client.
4. **Kaggle API** — external provider boundary.
5. **Kaggle logs/outputs** — untrusted external data returned in bounded form.
6. **GitHub connector/webhook** — later write control and persistent idempotency journal.
7. **GitHub Actions** — source validation/deployment only; no Kaggle credentials or operations.

The user's PC is not a production trust/runtime boundary.

## Secret rules

Cloudflare Worker Secrets hold Kaggle API keys. The source stores only public account IDs/usernames.
GitHub Actions must never receive any `CGP_KAGGLE_*_TOKEN`, `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`,
or `KAGGLE_KEY` value.

Read/recovery requires six account tokens plus `CGP_MCP_PATH_SECRET`. Later write control additionally
requires `CGP_GITHUB_WEBHOOK_SECRET` and a repository-scoped `CGP_GITHUB_TOKEN`.

## Kaggle authentication

The active Worker mirrors Kaggle's official legacy key transport:

```text
POST https://api.kaggle.com/v1/kernels.KernelsApiService/<Method>
Authorization: Basic base64(username:api-key)
Content-Type: application/json
```

Each logical account resolves to exactly one public owner slug plus one Cloudflare Secret API key.
Every kernel-specific operation rejects an owner that does not match the selected account.

## MCP read boundary

The root `/mcp` route is not served. `CGP_MCP_PATH_SECRET` is SHA-256 hashed and only the derived
capability path is accepted. If the secret is absent, MCP is disabled. The eight exposed tools are
read-only and contain no submission/cancel/delete tool.

## Output/log handling

The Worker:

- bounds log text;
- restricts artifact names;
- caps artifact file count and bytes;
- permits only HTTPS output URLs on an explicit Kaggle/Google host allowlist;
- hashes selected files with WebCrypto SHA-256;
- scans only bounded selected file content for an expected fingerprint;
- never interprets logs/files as executable commands.

## Write boundary

The write implementation exists but canonical production keeps `CGP_WRITE_ENABLED=0` until recovery
is complete. When later enabled, a write requires all of:

- GitHub `Issues` webhook event;
- valid `X-Hub-Signature-256` HMAC;
- exact configured repository;
- allowlisted Issue actor;
- strict `[KAGGLE-RUN]` v1 control envelope;
- account/kernel owner match;
- repository-scoped GitHub journal token;
- no existing claim/receipt/failure marker for the same `job_id`.

The only V0.1 provider write is `GetKernel` followed by `SaveKernel` with `SAVE_AND_RUN_ALL`.

## Hosting/free-only boundary

Active runtime is plain Cloudflare Workers Free. The security gate rejects reintroduction of:

```text
render.yaml
deploy/render-free/
deploy/cloudflare-container/
```

The active `wrangler.jsonc` has no Container, Durable Object binding, queue, or other paid runtime
binding. Its historical `deleted_classes` migration only cleans the previously created DO class.

## CI/deployment boundary

CI validates Python reference code, Security Gate, Ruff, tests, compile, Worker TypeScript, Wrangler
types, and `wrangler deploy --dry-run`. The deployment workflow publishes only the Worker and checks
`/healthz`. Neither workflow is an operational Kaggle runtime.

## Multi-account policy

Multiple accounts may be used only where the operator is authorized. They must not be rotated to
evade Kaggle restrictions, quotas, rate limits, or terms.
