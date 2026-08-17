# Architecture Freeze — V0.1 Remote Kaggle Gateway

## Decision

The production runtime is fully remote. The user's PC is not part of the architecture.

```text
GitHub: rezanory/chatgpt-plugins
  -> Cloudflare build/deploy
  -> Cloudflare Worker + Access/OAuth boundary
  -> Cloudflare Container
  -> Python Kaggle Direct Gateway
  -> one isolated KaggleApi client per logical account
  -> Kaggle API
```

GitHub is source control and source-quality CI. It is not a Kaggle execution/authentication relay.

Explicitly excluded from the active runtime:

- user-PC hosting or local tunnels;
- Kaggle CLI;
- GitHub Actions as Kaggle dispatcher/status runtime;
- browser sessions/cookies;
- repeated interactive login;
- shared global Kaggle credentials/client/config.

## Kaggle authentication contract

Every enabled account uses the operator-validated direct Python sequence:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

The final list call is the authenticated harmless readiness probe.

## Parallel account isolation

Each account receives an independent client slot:

```text
account_id
  -> dedicated KaggleApi instance
  -> dedicated temporary config directory
  -> dedicated temporary kaggle.json
  -> account creation lock
  -> account API-call lock
```

Different accounts can initialize and operate concurrently. Calls within one account are serialized
when needed. Temporary config is removed at shutdown; POSIX permissions are restricted to `0700`
for the directory and `0600` for the credential file.

Global Kaggle auth variables are rejected because they can override isolated config:

```text
KAGGLE_API_TOKEN
KAGGLE_USERNAME
KAGGLE_KEY
```

## Cloudflare runtime boundary

Production deployment lives in `deploy/cloudflare-container/`.

The Cloudflare Worker is the public MCP boundary. It:

1. exposes a generic `/healthz` endpoint;
2. accepts `/mcp` only after Cloudflare Access authentication;
3. cryptographically validates `Cf-Access-Jwt-Assertion` using the Access team JWKS;
4. verifies issuer and `POLICY_AUD` audience;
5. strips the Access JWT and cookies before forwarding;
6. proxies only authenticated MCP traffic to the Python Container.

The Container runs the unchanged Python gateway. It is allowed outbound Internet access only so it
can call Kaggle APIs.

`CGP_GATEWAY_TRUSTED_PROXY=cloudflare-container` permits the Python MCP server to bind to
`0.0.0.0:8080` only inside this approved Container boundary.

## Secret boundary

Real account credentials are Cloudflare Secrets, not GitHub secrets and never repository content.
The Worker passes them to the Container at runtime:

```text
CGP_KAGGLE_KG01_USERNAME / CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_USERNAME / CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG04_USERNAME / CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_USERNAME / CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_USERNAME / CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_USERNAME / CGP_KAGGLE_KG07_TOKEN
```

`kg-03` remains disabled until its canonical Kaggle owner slug is resolved.

## Read-only recovery surface

The initial MCP surface is deliberately read-only:

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

No read-recovery tool creates, restarts, modifies, cancels, or deletes a Kaggle run.

Recovery sequence:

```text
auth all accounts
  -> inventory existing pneumonia-v6-2-2 kernels
  -> resolve owner/kernel per shard
  -> read status/logs
  -> hash selected existing outputs
  -> compare expected fingerprint
  -> classify existing work before any new compute
```

Known evidence target:

```text
artifact: KAGGLE_EXECUTION_V62_2
fingerprint: fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

## Ownership and output boundaries

Every kernel-specific call validates:

```text
kernel_ref.owner == configured owner_slug for account_id
```

Kaggle responses are external/untrusted data. The gateway converts SDK objects to JSON-safe values,
bounds large logs, redacts configured credentials, validates output paths/symlinks, hashes selected
artifacts, and deletes temporary output downloads after producing the manifest.

## GitHub / CI boundary

`.github/workflows/ci.yml` may validate source and the Cloudflare deployment package. It must never
receive Kaggle runtime credentials or perform operational Kaggle calls.

CI gates include:

- Python security gate;
- Ruff;
- Python tests/compile;
- `wrangler types`;
- TypeScript validation;
- `wrangler deploy --dry-run`;
- Docker image build from the production Container Dockerfile.

The security gate rejects operational `.github/workflows/kaggle-*.yml` files and subprocess/CLI use
inside the active Python gateway.

## Deployment model

The intended steady state is GitHub-connected Cloudflare deployment from `main`. Cloudflare holds
runtime secrets and Access configuration. The user's PC is not required after one-time account-side
Cloudflare configuration.

## Deferred after live read-only recovery

- direct write/submission MCP tools;
- bounded retry/repair integration;
- durable scheduler/quota/health state;
- additional compute providers.
