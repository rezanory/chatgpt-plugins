# V0.1 Direct Kaggle Gateway Setup

## 1. Repository

Keep `rezanory/chatgpt-plugins` private. GitHub is used for source control and CI only.
Do not configure an Actions workflow to authenticate to or operate Kaggle.

## 2. Active component

The active runtime is:

```text
plugins/kaggle-gateway/
```

The older `plugins/kaggle/` relay code is retained temporarily for migration/reference and is not
an operational runtime.

## 3. Account registry

Public, non-secret gateway metadata is stored in:

```text
plugins/kaggle-gateway/src/chatgpt_plugin_kaggle_gateway/accounts.json
```

Each entry maps a logical account to secret-reference environment variable names:

```json
{
  "account_id": "kg-01",
  "owner_slug": "example-owner",
  "username_env": "CGP_KAGGLE_KG01_USERNAME",
  "token_env": "CGP_KAGGLE_KG01_TOKEN",
  "enabled": true
}
```

Never put a username/token value in this JSON file.

## 4. Runtime credential injection

Provide the working credentials to the **gateway process** through a secure deployment secret
mechanism. For example, the gateway process for `kg-01` must receive:

```text
CGP_KAGGLE_KG01_USERNAME=<username>
CGP_KAGGLE_KG01_TOKEN=<token>
```

Repeat for enabled accounts.

Do not set these process-global Kaggle variables in the gateway:

```text
KAGGLE_API_TOKEN
KAGGLE_USERNAME
KAGGLE_KEY
```

They are deliberately rejected because they can override per-account configuration during Kaggle
authentication.

Do not commit secrets, put them in GitHub Issues, or paste them into ChatGPT.

## 5. Direct authentication behavior

The gateway creates one isolated `KaggleApi()` instance per account and performs exactly:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

Before calling `set_config_value`, the gateway gives the instance its own temporary config path so
parallel accounts do not overwrite a shared `~/.kaggle/kaggle.json`.

## 6. Install development/runtime dependencies

From the repository root:

```bash
uv sync --all-packages --dev
```

Validation:

```bash
uv run python scripts/security_gate.py
uv run ruff check .
uv run pytest
uv run python -m compileall -q packages plugins
```

## 7. Start the gateway privately

The gateway defaults to loopback:

```bash
uv run --package chatgpt-plugin-kaggle-gateway kaggle-gateway
```

Default endpoint host/port:

```text
127.0.0.1:8000
```

The process refuses unauthenticated non-loopback binding unless an explicit development override is
set. Do not use that override for production. An online ChatGPT connection must expose the gateway
through authenticated MCP transport/tunneling or an authenticated reverse proxy.

## 8. Validate all accounts from MCP

First call:

```text
kaggle_auth_check_all(max_workers=6)
```

Expected shape:

```json
[
  {"account_id": "kg-01", "auth_ok": true},
  {"account_id": "kg-02", "auth_ok": true}
]
```

A failure is contained to its account and does not terminate the gateway or the other accounts.

## 9. Recover existing work before new compute

Use the parallel direct-API inventory tool first:

```text
kaggle_kernels_inventory_all(
  search="pneumonia-v6-2-2",
  page_size=20,
  max_workers=6
)
```

For every discovered `owner/kernel`:

```text
kaggle_kernel_status(account_id, kernel_ref)
kaggle_kernel_logs(account_id, kernel_ref)
```

This stage is read-only. Do not create or restart compute until the existing runs are classified.

## 10. CI boundary

`.github/workflows/ci.yml` validates source only. It must never receive Kaggle credentials.
The security gate fails if an operational `.github/workflows/kaggle-*.yml` workflow is added or if
the direct gateway source introduces subprocess/CLI execution.

## 11. Next stage after recovery

After the existing runs are understood, direct write tools can be added to the same `KaggleApiPool`
for controlled submission/retry. They must remain direct Python API calls and require explicit
write semantics; they must not be mislabeled as read-only MCP tools.
