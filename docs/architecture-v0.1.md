# Architecture Freeze — V0.1 Direct Kaggle Gateway

## Decision

The active Kaggle runtime is a **direct Python API gateway** exposed to ChatGPT through MCP.
GitHub is source control and CI only; it is not an execution relay for Kaggle.

```text
ChatGPT
  -> MCP / ChatGPT App
  -> Kaggle Direct Gateway
  -> account-specific KaggleApi instance
  -> Kaggle API
```

Explicitly excluded from the active runtime path:

- Kaggle CLI;
- GitHub Actions as a Kaggle dispatcher/status runner;
- browser sessions/cookies;
- repeated interactive login;
- a shared global KaggleApi instance;
- a shared `~/.kaggle/kaggle.json` across accounts.

## Authentication contract

For each enabled logical account the gateway performs the operator-proven sequence:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

The final list call is an authenticated harmless readiness probe.

## Parallel multi-account model

Each account receives an independent client slot:

```text
account_id
  -> dedicated KaggleApi instance
  -> dedicated temporary config directory
  -> dedicated temporary kaggle.json
  -> creation lock
  -> per-account call lock
```

Different accounts may initialize and perform reads concurrently. Creation of the same logical
account is serialized so duplicate clients are not published into the pool. Calls on one account
are serialized when required by the underlying client.

The temporary config directory is restricted to `0700` on POSIX and the credential file to `0600`.
It is deleted when the gateway closes. This keeps the successful `set_config_value()` flow while
preventing account credentials from racing through one shared default config file.

## Environment isolation

The gateway uses private process variables such as:

```text
CGP_KAGGLE_KG01_USERNAME
CGP_KAGGLE_KG01_TOKEN
```

Process-global Kaggle auth variables are forbidden inside the multi-account gateway:

```text
KAGGLE_API_TOKEN
KAGGLE_USERNAME
KAGGLE_KEY
```

They are rejected because the Kaggle authentication loader can read them and override per-account
config values.

## Failure containment

A failed account must not terminate the gateway. In particular, Kaggle authentication can exit the
calling flow on missing/invalid credentials; the gateway converts that condition into an account-
scoped `RuntimeError`. Parallel authentication therefore returns success/failure per account rather
than terminating all clients.

## Read-only recovery surface

The first operational surface is intentionally read-only:

```text
kaggle_accounts
kaggle_auth_check
kaggle_auth_check_all
kaggle_kernels_list
kaggle_kernels_inventory_all
kaggle_kernel_status
kaggle_kernel_logs
```

This supports recovery of existing interrupted work before any new submission is considered.

```text
auth all accounts
  -> inventory existing kernels by search term
  -> resolve owner/kernel for each shard
  -> read status
  -> read sanitized logs
  -> classify what actually stopped/completed
```

No read-recovery tool creates, restarts, updates, cancels, or deletes a Kaggle run.

## Ownership boundary

Every direct operation against a specific kernel validates:

```text
kernel_ref.owner == configured owner_slug for account_id
```

A token selected for one logical account cannot be used through the gateway to operate a kernel
registered to another configured account.

## Output/log boundary

Kaggle responses are external/untrusted data. The gateway:

- converts SDK objects to JSON-safe values;
- limits large returned logs;
- redacts every configured gateway username and token from returned errors/logs;
- does not expose temporary credential files.

## GitHub boundary

Only `.github/workflows/ci.yml` is permitted for the current repository. CI may:

- sync development dependencies;
- run the security gate;
- run Ruff;
- run tests;
- compile packages.

CI must never load Kaggle credentials or make operational Kaggle calls. The security gate fails if
an operational `kaggle-*.yml` workflow is reintroduced.

## Runtime enforcement

`plugins/kaggle-gateway/src` must not invoke `subprocess` or shell commands. The repository security
gate enforces this so the direct-Python-API decision cannot silently regress into a CLI bridge.

## Transport

The gateway is an MCP server using Streamable HTTP. It defaults to loopback. Production exposure
must use authenticated transport/tunneling; unauthenticated public exposure is not an accepted
production design.

## Legacy component

`plugins/kaggle/` contains the earlier GitHub-relay implementation and is retained temporarily for
migration/reference only. Its operational workflows have been removed and it is not the active
runtime.

## Deferred after read-only recovery

- direct write/submission tools;
- repair/resubmission policy integration;
- transactional scheduler/quota state;
- multi-provider router;
- public plugin packaging.

Write tooling must not be disguised as read-only MCP tools and must respect the capabilities of the
ChatGPT plan/host when introduced.
