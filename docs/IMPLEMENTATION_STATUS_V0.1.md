# Implementation Status — V0.1 Direct Kaggle Gateway

## Active architecture

```text
ChatGPT
  -> custom MCP / Secure MCP Tunnel
  -> Kaggle Direct Gateway
  -> one isolated KaggleApi instance per logical account
  -> Kaggle API
```

GitHub remains source control and CI only. Operational Kaggle GitHub Actions and the Kaggle CLI are
not part of the active runtime.

## Implemented

- Modular `chatgpt-plugins` monorepo.
- Active `plugins/kaggle-gateway` component.
- MCP Python SDK v2 / `MCPServer` Streamable HTTP server.
- One dedicated `KaggleApi()` instance per account.
- Operator-proven authentication sequence:
  - `set_config_value(CONFIG_NAME_USER, username)`
  - `set_config_value(CONFIG_NAME_KEY, token)`
  - `authenticate()`
  - `kernels_list(page_size=1)` readiness probe.
- Private temporary config directory/file per account to prevent shared `kaggle.json` races.
- POSIX `0700` temp directory and `0600` credential file permissions.
- One-time `KaggleApi` class import before parallel account authentication.
- Per-account creation lock and API-call lock.
- Parallel authentication/readiness across enabled accounts.
- Account-scoped containment of Kaggle `SystemExit` authentication failures.
- Rejection of process-global `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` overrides.
- Owner/account validation for kernel-specific operations.
- Credential redaction from returned errors and logs.
- Direct read-only MCP tools:
  - `kaggle_accounts`
  - `kaggle_auth_check`
  - `kaggle_auth_check_all`
  - `kaggle_kernels_list`
  - `kaggle_kernels_inventory_all`
  - `kaggle_kernel_status`
  - `kaggle_kernel_logs`
  - `kaggle_kernel_output_manifest`
- Read-only output recovery with:
  - allow-listed artifact names;
  - escaped file-pattern regex;
  - temporary output directory;
  - path/symlink checks;
  - file-count and total-size limits;
  - SHA-256 manifest;
  - expected fingerprint scan;
  - automatic local cleanup.
- Security gate that rejects subprocess/CLI usage in active gateway source.
- Security gate that rejects operational `.github/workflows/kaggle-*.yml` workflows.
- Only `.github/workflows/ci.yml` remains; CI has no Kaggle credential/runtime role.
- Windows private launcher under `deploy/kaggle-gateway/start-private.ps1`.
- Secret-reference template under `deploy/kaggle-gateway/env.example`.
- Private deployment documentation oriented around Secure MCP Tunnel.

## Current account registry

Enabled logical accounts:

```text
kg-01
kg-02
kg-04
kg-05
kg-06
kg-07
```

`kg-03` remains disabled until its canonical Kaggle owner slug is corrected.

No credential value is stored in the repository registry.

## Validation baseline

Verified CI run on commit:

```text
e2914b482c1ef544bbcbde61abe66e9ccf859fcd
```

Result:

```text
Package installation: PASS
Security gate:        PASS
Ruff:                 PASS
Pytest:               35 PASS
Compileall:           PASS
CI conclusion:        SUCCESS
```

The successful CI installs the actual editable packages, including:

```text
kaggle==2.2.4
mcp==2.0.0
chatgpt-plugin-kaggle-gateway
```

## External operational evidence

Outside this repository CI, the operator has already reconnected the active Kaggle accounts using
the exact direct `KaggleApi` sequence implemented by the gateway and reported `auth_ok` for all six
active accounts.

That operational evidence is the basis for preserving this authentication method rather than using
CLI/browser-session authentication.

## Next operational milestone

Run the direct gateway in the same private environment that already has working six-account
credentials, then connect it to ChatGPT through Secure MCP Tunnel.

First live calls from ChatGPT must remain read-only:

```text
1. kaggle_auth_check_all(max_workers=6)
2. kaggle_kernels_inventory_all(search="pneumonia-v6-2-2", page_size=20, max_workers=6)
3. resolve exact existing owner/kernel refs
4. kaggle_kernel_status(...)
5. kaggle_kernel_logs(...)
6. kaggle_kernel_output_manifest(...)
```

Known recovery targets:

```text
artifact identifier:
KAGGLE_EXECUTION_V62_2

expected fingerprint:
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

No new Kaggle compute should be submitted before the six existing runs are inventoried and
classified.

## Deferred after read-only live recovery

- direct write/submission tools exposed through MCP;
- controlled resubmission/retry;
- repair-loop integration with GitHub source changes;
- durable quota/health scheduler state;
- additional providers in the monorepo.

On ChatGPT Pro, the immediate target is the read/fetch surface. Full custom-MCP write/modify support
is currently a later plan/host capability rather than a requirement for V0.1 read-only recovery.
