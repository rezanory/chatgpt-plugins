# Kaggle Direct Gateway — Private Deployment

The V0.1 deployment target is the same private machine/environment where the authorized Kaggle
accounts are known to authenticate successfully through `KaggleApi`.

## Runtime topology

```text
ChatGPT
  -> Secure MCP Tunnel
  -> http://127.0.0.1:8000/mcp
  -> Kaggle Direct Gateway
  -> one isolated KaggleApi instance per account
  -> Kaggle API
```

GitHub Actions and the Kaggle CLI are not in this runtime path.

OpenAI's current ChatGPT MCP guidance states that a local/private MCP server is not connected
directly from ChatGPT; use **Secure MCP Tunnel** so the private server does not have to be exposed
to the public Internet:

https://help.openai.com/en/articles/12584461-developer-mode-and-mcp-apps-in-chatgpt

Do not replace the tunnel with an unauthenticated public port.

## 1. Inject account credentials into the gateway process

The active registry is packaged with the gateway and refers to environment variable names only.
For the six currently enabled logical accounts provide the matching username/token pairs:

```text
CGP_KAGGLE_KG01_USERNAME
CGP_KAGGLE_KG01_TOKEN
CGP_KAGGLE_KG02_USERNAME
CGP_KAGGLE_KG02_TOKEN
CGP_KAGGLE_KG04_USERNAME
CGP_KAGGLE_KG04_TOKEN
CGP_KAGGLE_KG05_USERNAME
CGP_KAGGLE_KG05_TOKEN
CGP_KAGGLE_KG06_USERNAME
CGP_KAGGLE_KG06_TOKEN
CGP_KAGGLE_KG07_USERNAME
CGP_KAGGLE_KG07_TOKEN
```

Use the same working username/token values that already pass the direct sequence:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

Do not store real values in `env.example`, Git, ChatGPT, or logs.

## 2. Start on Windows

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\start-private.ps1
```

The launcher:

- verifies all enabled account secret variables are present without printing values;
- rejects global `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` overrides;
- installs the gateway package when needed;
- forces loopback binding;
- starts Streamable HTTP MCP at `http://127.0.0.1:8000/mcp`.

To skip the editable install after it has already been installed:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\start-private.ps1 -SkipInstall
```

## 3. Connect through ChatGPT

On ChatGPT Pro, enable Developer Mode and connect the private MCP through Secure MCP Tunnel. The
initial gateway exposes read/fetch-style tools only, which matches the current Pro MCP capability.

Initial tools:

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

## 4. First operational sequence

Do not launch new compute first. Recover the existing six-account pneumonia work:

```text
1. kaggle_auth_check_all(max_workers=6)
2. kaggle_kernels_inventory_all(search="pneumonia-v6-2-2", page_size=20, max_workers=6)
3. resolve exact owner/kernel refs
4. kaggle_kernel_status(account_id, kernel_ref)
5. kaggle_kernel_logs(account_id, kernel_ref)
6. kaggle_kernel_output_manifest(
     account_id,
     kernel_ref,
     ["KAGGLE_EXECUTION_V62_2", "fingerprint"],
     "fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838"
   )
```

Only after the existing runs are classified should write/submission tooling be enabled.

## Security notes

- The gateway does not invoke `subprocess` or the Kaggle CLI.
- Each account receives an isolated temporary config directory and `KaggleApi` instance.
- Kernel owner refs are checked against the selected logical account.
- Gateway credentials are redacted from returned errors and logs.
- Selected output recovery uses a temporary directory, hashes files, returns a manifest, and deletes
  the local download before returning.
- `CGP_GATEWAY_ALLOW_UNAUTHENTICATED_REMOTE=1` is a development escape hatch only and must not be
  used as a production deployment method.
