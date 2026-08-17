# Security Model — V0.1 Direct Kaggle Gateway

## Trust boundaries

1. **ChatGPT / user conversation** — requests work but never receives Kaggle credentials.
2. **MCP transport** — connects ChatGPT to the gateway; production exposure must be authenticated.
3. **Kaggle Direct Gateway** — long-running trusted service that owns account selection and secret
   injection.
4. **Per-account KaggleApi slot** — isolated Python API object, temporary config, and call lock.
5. **Kaggle API** — external provider boundary.
6. **Kaggle outputs/logs** — external/untrusted data sanitized before presentation to ChatGPT.
7. **GitHub repository/CI** — source and validation only; no Kaggle runtime credentials.

## Secret rules

- Kaggle credential values are never committed to Git.
- The active account registry stores `account_id`, public owner slug, enabled state, and names of
  secret environment variables only.
- Gateway credentials use `CGP_KAGGLE_<ID>_USERNAME` and `CGP_KAGGLE_<ID>_TOKEN` style variables.
- Process-global `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` are rejected by the
  multi-account gateway because they can override account-isolated configuration.
- Every account gets a separate temporary config directory/file before `set_config_value()` is
  called.
- On POSIX, the directory is `0700` and the config file is `0600`.
- Temporary credential files are deleted on gateway shutdown and on failed client construction.

## Authentication isolation

The direct account boundary is:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

No shared module-level KaggleApi instance is used for operational calls. Different accounts have
separate instance-local config paths so parallel authentication cannot overwrite one shared
`~/.kaggle/kaggle.json`.

If Kaggle authentication terminates with `SystemExit`, the gateway converts it into an account-
scoped failure. One bad account therefore cannot terminate the whole service or the other accounts.

## Direct-runtime invariant

The active gateway source must not execute the Kaggle CLI or spawn subprocesses. CI enforces:

- no `subprocess` import/use inside `plugins/kaggle-gateway/src`;
- no operational `.github/workflows/kaggle-*.yml` files.

GitHub Actions are allowed only for repository validation.

## Account/owner boundary

Every kernel-specific operation checks that:

```text
kernel_ref.owner == registry[account_id].owner_slug
```

This prevents accidental credential crossover between configured accounts.

## External output/log handling

Kaggle responses and logs are untrusted. Before returned text reaches ChatGPT, the gateway:

- bounds response size;
- replaces every configured gateway token with `[REDACTED]`;
- also redacts configured gateway usernames from returned errors/logs;
- converts SDK objects to JSON-safe data rather than executing or evaluating their representation.

No Kaggle log or filename is interpreted as a command.

## MCP read surface

The initial operational surface is read-only: account listing/auth readiness, kernel inventory,
status, and logs. MCP `ToolAnnotations` mark these tools read-only/idempotent hints, but security does
not rely on annotations alone; the functions themselves contain no write/submit/cancel path.

## Server exposure

The server defaults to loopback. It refuses non-loopback binding unless an explicit development
override is supplied. That override is not a production authentication mechanism. Production use
must place the MCP endpoint behind authenticated transport/tunneling or equivalent access control.

## Repair/write boundary

No autonomous write or repair path is exposed in the initial direct read/recovery surface. When
write tools are added later:

- they must use direct `KaggleApi` methods, never CLI/Actions relay;
- they must be explicitly classified as write operations;
- historical execution evidence must remain immutable;
- authentication/quota/policy failures must never trigger source modification;
- retry/repair attempts must remain bounded.

## Provider policy

Multi-account support is for accounts the operator is authorized to use. The gateway must not
rotate accounts to evade provider restrictions or quota limits.
