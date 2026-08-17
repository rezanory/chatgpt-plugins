# Security Model — V0.1 Remote Kaggle Gateway

## Trust boundaries

1. **ChatGPT conversation** — requests work but never receives Kaggle credentials.
2. **Cloudflare Access/OAuth** — authenticates the MCP client/user.
3. **Cloudflare Worker** — validates the Access JWT and is the public `/mcp` boundary.
4. **Cloudflare Container** — private Python runtime for the direct Kaggle gateway.
5. **Per-account KaggleApi slot** — isolated client/config/lock.
6. **Kaggle API** — external provider boundary.
7. **Kaggle outputs/logs** — external/untrusted data sanitized before ChatGPT.
8. **GitHub repository/CI** — source and validation only; no Kaggle runtime credentials.

The user's PC is not a production trust or runtime boundary.

## Secret rules

- Kaggle credential values are never committed to Git.
- Production Kaggle credentials are Cloudflare Secrets.
- GitHub Actions does not receive Kaggle runtime credentials.
- The account registry stores only public metadata and secret-reference names.
- Process-global `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` are rejected.
- Every account gets a separate temporary config directory/file before `set_config_value()`.
- POSIX temporary directory/file permissions are `0700`/`0600`.
- Temporary credential files are removed at shutdown and on failed client construction.

## Kaggle authentication isolation

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

No shared operational `KaggleApi` instance or shared `~/.kaggle/kaggle.json` is used.
Authentication failure from one account is converted to an account-scoped failure rather than
terminating the whole gateway.

## Cloudflare Access boundary

The Worker accepts `/mcp` only after validating `Cf-Access-Jwt-Assertion` using:

- the configured Cloudflare Access team-domain JWKS;
- the expected issuer;
- the configured `POLICY_AUD` audience.

Before proxying to the Container, the Worker removes:

```text
Cf-Access-Jwt-Assertion
Cookie
```

The Python Container therefore never needs to interpret Cloudflare user/session credentials.

The security gate fails if required `/mcp` JWT-validation/stripping fragments disappear from the
production Worker source.

## Container boundary

The Python MCP service may bind non-loopback in production only when
`CGP_GATEWAY_TRUSTED_PROXY=cloudflare-container` is supplied by the production image.

The Container is single-instance in V0.1 (`max_instances=1`) so account-local client/config state is
not silently split across replicas before durable scheduler state exists.

Outbound Internet access is enabled because direct Kaggle API access requires it.

## Direct-runtime invariant

The active Python gateway must not execute the Kaggle CLI or spawn subprocesses. CI enforces:

- no subprocess use inside `plugins/kaggle-gateway/src`;
- no operational `.github/workflows/kaggle-*.yml` runtime;
- no hard-coded Kaggle credential values in the Cloudflare deployment package.

GitHub Actions may build/test the deployment package but must not authenticate to Kaggle.

## Account/owner boundary

Every kernel-specific operation checks:

```text
kernel_ref.owner == registry[account_id].owner_slug
```

This prevents accidental credential crossover between configured accounts.

## External output/log handling

Before Kaggle data reaches ChatGPT, the gateway:

- bounds response/log size;
- redacts configured usernames/tokens;
- converts SDK objects to JSON-safe data;
- validates selected artifact names and paths;
- rejects symlink/path escape during output recovery;
- hashes recovered artifacts with SHA-256;
- scans only bounded text for the expected fingerprint;
- deletes temporary downloaded outputs after producing the manifest.

No Kaggle log, filename, or output is interpreted as code or a command.

## Read-only MCP surface

The initial operational surface contains account/auth/inventory/status/log/output-manifest reads
only. MCP annotations mark them read-only/idempotent, but security relies on the implementation:
there is no submit/restart/cancel/delete path exposed through these tools.

## GitHub / build boundary

CI currently validates:

```text
Security Gate
Ruff
Python tests
Python compile
wrangler types
TypeScript
wrangler deploy --dry-run
Docker image build
```

The deploy dry-run receives no Kaggle credentials and makes no operational Kaggle call.

## Repair/write boundary

No autonomous write/repair path is exposed in V0.1 recovery. Later write tools must:

- call direct `KaggleApi` methods, never Kaggle CLI;
- be explicitly classified as writes;
- keep historical evidence immutable;
- never repair source for auth/quota/policy failures;
- use bounded retry, fingerprint, and no-progress stopping rules.

## Provider policy

Multi-account support is only for accounts the operator is authorized to use. The scheduler must
not rotate accounts to evade Kaggle restrictions, quotas, or terms.
