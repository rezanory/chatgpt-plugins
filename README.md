# chatgpt-plugins

V0.1 monorepo for operating external compute from a normal ChatGPT conversation without using the
user's PC as a runtime.

The first provider is **Kaggle**. The active production runtime is a direct Python `KaggleApi`
gateway hosted remotely in Cloudflare Containers and exposed through a Cloudflare Worker/MCP
boundary.

## Active production path

```text
GitHub: rezanory/chatgpt-plugins
  -> Cloudflare build/deploy
  -> Cloudflare Worker + Access/OAuth boundary
  -> Cloudflare Container
  -> Python Kaggle Direct Gateway
  -> one isolated KaggleApi instance per logical account
  -> Kaggle API
```

The user's PC is **not** part of the production runtime. GitHub is source control and source-quality
CI; Kaggle authentication/execution is never performed by GitHub Actions.

There is no Kaggle CLI, browser session, cookie login, or repeated interactive login in the active
runtime.

## Proven Kaggle authentication sequence

Every enabled account gets its own `KaggleApi()` object and follows the operator-validated flow:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

A successful `kernels_list(page_size=1)` is the harmless authenticated readiness probe.

## Multi-account isolation

`set_config_value()` writes account configuration to a file, so the gateway never shares the
default `~/.kaggle/kaggle.json` between accounts. Each logical account receives:

- one dedicated `KaggleApi` instance;
- one private temporary config directory;
- one private temporary `kaggle.json`;
- one per-account creation lock;
- one per-account API-call lock.

Different accounts can authenticate and operate in parallel. Calls within one account are
serialized where needed. Temporary config directories are deleted when the gateway shuts down.

## Cloudflare credential model

Real Kaggle credentials live only as **Cloudflare Worker Secrets** and are passed at runtime to the
Python Container:

```text
CGP_KAGGLE_KG01_USERNAME
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_USERNAME
CGP_KAGGLE_KG02_TOKEN
...
```

The repository stores only the names of those secret references and public account metadata.

Do not configure process-global `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, or `KAGGLE_KEY`; the gateway
rejects them to prevent cross-account credential override.

## Current read/recovery MCP surface

- `kaggle_accounts`
- `kaggle_auth_check`
- `kaggle_auth_check_all`
- `kaggle_kernels_list`
- `kaggle_kernels_inventory_all`
- `kaggle_kernel_status`
- `kaggle_kernel_logs`
- `kaggle_kernel_output_manifest`

These tools use direct `KaggleApi` calls. They do not submit, restart, cancel, edit, or delete a
Kaggle run.

## Existing-run recovery first

Before new compute, recover the existing pneumonia work:

```text
ChatGPT
  -> kaggle_auth_check_all(max_workers=6)
  -> kaggle_kernels_inventory_all(search="pneumonia-v6-2-2")
  -> resolve exact owner/kernel refs per account
  -> kaggle_kernel_status(...)
  -> kaggle_kernel_logs(...)
  -> kaggle_kernel_output_manifest(...)
```

Known evidence target:

```text
artifact: KAGGLE_EXECUTION_V62_2
fingerprint: fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

No new Kaggle compute should be created before those existing runs are classified.

## Cloudflare deployment package

Production deployment source is under:

```text
deploy/cloudflare-container/
```

It contains:

- `Dockerfile` — Python Kaggle Gateway image;
- `src/index.ts` — authenticated Worker proxy;
- `wrangler.jsonc` — Container/Durable Object configuration;
- `package.json` — pinned Wrangler/Container SDK tooling;
- `README.md` — one-time Cloudflare/GitHub/Access setup.

CI validates both the Python application and Cloudflare deployment package. The Cloudflare gate
runs `wrangler types`, TypeScript validation, `wrangler deploy --dry-run`, and an actual local Docker
image build without deploying or receiving Kaggle credentials.

## Security boundaries

- No Kaggle credential value is committed to Git.
- GitHub Actions never receives Kaggle runtime credentials.
- Active Python gateway source contains no subprocess/Kaggle-CLI runtime path.
- Operational `.github/workflows/kaggle-*.yml` files are forbidden by the security gate.
- Cloudflare Worker protects `/mcp` and validates the Cloudflare Access JWT before proxying.
- Access JWT/cookies are stripped before the request reaches the Python Container.
- A kernel `owner/slug` must match the selected logical account's configured owner.
- Authentication failure from one account cannot terminate the other account clients.
- Credential values are redacted from returned errors/logs.

## Repository layout

```text
packages/
  core/                   contracts and scheduler primitives
  github-bridge/          source/control integrations for the wider monorepo
  repair-engine/          classifier/fingerprint/repair policy
plugins/
  kaggle-gateway/         ACTIVE direct multi-account KaggleApi + MCP runtime
  kaggle/                 legacy relay code retained temporarily for migration/reference
deploy/
  cloudflare-container/   ACTIVE production deployment package
.github/workflows/
  ci.yml                  source/deploy-package validation only
```

## Safety boundary

Multiple accounts are supported only when the operator is authorized to use them. The gateway must
not rotate accounts to evade Kaggle restrictions, quotas, or terms.
