# chatgpt-plugins

V0.1 monorepo for operating external compute from a normal ChatGPT conversation without using the
user's PC as a runtime and without requiring a paid hosting service.

The first provider is **Kaggle**. The active V0.1 runtime is a direct Python `KaggleApi` gateway on
a **Render Free** web service.

## Goal: operate Kaggle from this ChatGPT conversation

The user should be able to say, for example:

```text
Run this existing Kaggle shard again on kg-01.
```

and ChatGPT performs the operation from the conversation.

Because ChatGPT Pro custom MCP is read/fetch-only, V0.1 separates read and write control without
weakening the Kaggle authentication path:

```text
READ / RECOVERY
ChatGPT
  -> custom MCP (read-only)
  -> Render Free Kaggle Gateway
  -> direct KaggleApi()
  -> Kaggle

WRITE / EXECUTION
ChatGPT
  -> built-in GitHub connector creates a signed-control Issue
  -> GitHub repository webhook
  -> Render Free Kaggle Gateway
  -> direct KaggleApi()
  -> Kaggle
  -> receipt/failure comment on the same GitHub Issue
  -> ChatGPT reads the result through the GitHub connector
```

**No Kaggle CLI and no GitHub Actions are used to authenticate to or execute Kaggle.** GitHub
Actions in this repository are source-quality CI only.

The user's PC is not part of either runtime path.

## Proven Kaggle authentication sequence

Every enabled account gets its own isolated `KaggleApi()` object and follows the operator-validated
flow:

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

## Current accounts

Enabled V0.1 logical accounts:

```text
kg-01 -> azadka
kg-02 -> radlinaradlina
kg-04 -> reyhanehazad
kg-05 -> trickermark
kg-06 -> msdenis
kg-07 -> nisabulutmark
```

`kg-03` remains disabled until its exact Kaggle owner slug is resolved.

## Current MCP read/recovery surface

- `kaggle_accounts`
- `kaggle_auth_check`
- `kaggle_auth_check_all`
- `kaggle_kernels_list`
- `kaggle_kernels_inventory_all`
- `kaggle_kernel_status`
- `kaggle_kernel_logs`
- `kaggle_kernel_output_manifest`

These use direct `KaggleApi` calls and do not create new compute.

## V0.1 write surface

The first deliberately narrow write action is:

```text
rerun_existing
```

It is initiated by a GitHub Issue whose title starts with:

```text
[KAGGLE-RUN]
```

and whose body contains:

```text
<!-- chatgpt-plugins-kaggle-control:v1 -->
```

followed by a strict JSON payload such as:

```json
{
  "schema": "chatgpt.kaggle.control/v1",
  "action": "rerun_existing",
  "job_id": "rerun-s01-001",
  "account_id": "kg-01",
  "kernel_ref": "azadka/example-kernel"
}
```

The gateway validates the GitHub webhook HMAC, repository, issue actor, schema, account ID, and
kernel owner before execution. It then calls the Python API directly:

```text
api.kernels_pull(..., metadata=True)
api.kernels_push(...)
```

A persistent claim marker is written to the Issue before submission so webhook redelivery cannot
silently run the same `job_id` twice. The final submission receipt or safe failure is written back
to the same Issue.

## Existing-run recovery first

Before creating or rerunning compute, recover the existing pneumonia work:

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

Do not submit new Kaggle compute before those existing runs are classified.

## Free remote deployment

The repository contains a Render Blueprint:

```text
render.yaml
```

and Docker runtime:

```text
deploy/render-free/Dockerfile
```

The Blueprint explicitly sets:

```text
plan: free
```

No paid Render or Cloudflare Container plan is part of V0.1.

### One-time Render configuration

Create the Blueprint from this private GitHub repository and provide only secret values that are
marked `sync: false` in `render.yaml`:

- six Kaggle API tokens (`kg-01`, `02`, `04`, `05`, `06`, `07`);
- `CGP_GITHUB_WEBHOOK_SECRET` — a random webhook secret;
- `CGP_GITHUB_TOKEN` — a fine-grained GitHub token restricted to
  `rezanory/chatgpt-plugins` with Issues read/write access.

Kaggle usernames are public metadata and are already supplied by the Blueprint.

`CGP_MCP_PATH_SECRET` is generated by Render. At startup, the service logs the derived private MCP
path as:

```text
CGP_MCP_ENDPOINT_PATH=/mcp/<sha256>
```

The complete custom MCP endpoint is the service's HTTPS Render URL plus that path.

### One-time GitHub webhook configuration

In `rezanory/chatgpt-plugins` configure one repository webhook:

```text
Payload URL: https://<render-service>.onrender.com/github/webhook
Content type: application/json
Secret: same value as CGP_GITHUB_WEBHOOK_SECRET
Events: Issues only
```

After this, ChatGPT can trigger `rerun_existing` by creating a control Issue through the connected
GitHub app. There is no Actions job in that execution path.

## Security boundaries

- No Kaggle credential value is committed to Git.
- GitHub Actions never receives Kaggle runtime credentials.
- Active Python gateway source contains no subprocess/Kaggle-CLI runtime path.
- Operational `.github/workflows/kaggle-*.yml` files are forbidden by the security gate.
- Paid Cloudflare Container runtime is forbidden by the security gate.
- Production Blueprint is pinned to `plan: free`.
- Write webhook requires a valid GitHub HMAC signature.
- Control events are limited to the configured repository and approved GitHub actors.
- V0.1 only permits the explicit `rerun_existing` write action.
- A kernel `owner/slug` must match the selected logical account's configured owner.
- Duplicate `job_id` webhook deliveries do not trigger another run.
- Credential values are redacted from returned errors/logs.

## Repository layout

```text
packages/
  core/                   contracts and scheduler primitives
  github-bridge/          source/control integrations for the wider monorepo
  repair-engine/          classifier/fingerprint/repair policy
plugins/
  kaggle-gateway/         ACTIVE direct multi-account KaggleApi + MCP/webhook runtime
  kaggle/                 legacy relay code retained temporarily for migration/reference
deploy/
  render-free/            ACTIVE free production Docker package
.github/workflows/
  ci.yml                  source and Docker validation only; never Kaggle runtime execution
render.yaml                Render Free Blueprint
```

## Safety boundary

Multiple accounts are supported only when the operator is authorized to use them. The gateway must
not rotate accounts to evade Kaggle restrictions, quotas, or terms.
