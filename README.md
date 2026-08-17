# chatgpt-plugins

V0.1 monorepo for operating external compute from a normal ChatGPT conversation without using
ChatGPT Work or Codex Cloud as the execution runner.

The first provider is **Kaggle**. The active Kaggle runtime is a direct Python API gateway.

## Active runtime path

```text
ChatGPT
  |
  | MCP / ChatGPT App connection
  v
Kaggle Direct Gateway
  |
  | select account_id
  v
Dedicated KaggleApi instance for that account
  |
  | direct Python API calls
  v
Kaggle
```

**GitHub is not in the Kaggle runtime path.** GitHub stores source and runs repository CI only.
There is no Kaggle CLI, GitHub Actions relay, browser session, cookie login, or repeated interactive
login in the active runtime architecture.

## Proven authentication sequence

Every enabled account gets its own `KaggleApi()` object. Authentication intentionally follows the
same direct method already validated against the operator's accounts:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

A successful `kernels_list(page_size=1)` is the harmless authentication/readiness probe.

## Multi-account isolation

`set_config_value()` writes account configuration to a file. Sharing the default
`~/.kaggle/kaggle.json` between several clients would create an overwrite race. The gateway keeps
the proven sequence above but gives each logical account:

- one dedicated `KaggleApi` instance;
- one private temporary config directory;
- one private temporary `kaggle.json`;
- one per-account creation lock;
- one per-account API-call lock.

Different accounts authenticate and operate in parallel. Calls within the same account are
serialized where needed. Temporary config directories are removed when the gateway shuts down.
On POSIX systems the directory is restricted to `0700` and the config file to `0600`.

## Credential model

Credentials are injected into the **gateway process secret environment**, never committed to Git.
The repository contains only secret-reference variable names such as:

```text
CGP_KAGGLE_KG01_USERNAME
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_USERNAME
CGP_KAGGLE_KG02_TOKEN
...
```

Do not set process-global `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, or `KAGGLE_KEY` in the
multi-account gateway. The gateway rejects those variables so they cannot override isolated
per-account configuration during authentication.

## Current read/recovery tools

The MCP gateway currently exposes the safe first-stage tools:

- `kaggle_accounts`
- `kaggle_auth_check`
- `kaggle_auth_check_all`
- `kaggle_kernels_list`
- `kaggle_kernels_inventory_all`
- `kaggle_kernel_status`
- `kaggle_kernel_logs`

These tools use direct `KaggleApi` calls. They do not submit, restart, or cancel a kernel.

## Existing-run recovery first

Before creating any new compute, existing multi-account work is recovered read-only:

```text
ChatGPT
  -> kaggle_auth_check_all()
  -> kaggle_kernels_inventory_all(search="pneumonia-v6-2-2")
  -> identify owner/kernel refs per account
  -> kaggle_kernel_status(account_id, kernel_ref)
  -> kaggle_kernel_logs(account_id, kernel_ref)
  -> decide whether an existing run completed, failed, or needs later resubmission
```

No new Kaggle compute is created during this sequence.

## Security boundaries

- The active gateway runtime contains no subprocess/CLI path.
- CI fails if `plugins/kaggle-gateway/src` imports or invokes `subprocess`.
- CI fails if a `kaggle-*.yml` operational GitHub Actions workflow is reintroduced.
- A kernel `owner/slug` must match the selected account's configured owner.
- Authentication failure from one account is contained to that account; it cannot terminate the
  whole multi-account gateway.
- Tokens and configured usernames are redacted from returned errors and logs.
- Account configuration files in Git contain secret references only, never secret values.
- The MCP server defaults to loopback and refuses unauthenticated non-loopback binding unless an
  explicit development override is supplied. Production exposure must use authenticated transport.

## Repository layout

```text
packages/
  core/             job/provider contracts and scheduler primitives
  github-bridge/    source/control integrations for the wider monorepo
  repair-engine/    failure classifier, fingerprint, repair policy/contracts
plugins/
  kaggle-gateway/   ACTIVE direct multi-account KaggleApi + MCP runtime
  kaggle/           legacy relay implementation retained temporarily for migration/reference
.github/workflows/
  ci.yml            source validation only; never a Kaggle runtime
```

## Development

```bash
uv sync --all-packages --dev
uv run pytest
uv run ruff check .
uv run python scripts/security_gate.py
```

## Safety boundary

Multiple accounts are supported only when the operator is authorized to use them. The gateway must
not be used to evade Kaggle quotas, restrictions, or terms. Provider quota behavior is not
hard-coded or bypassed.
