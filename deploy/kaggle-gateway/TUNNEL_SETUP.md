# Secure MCP Tunnel Setup — Kaggle Direct Gateway

This is the last connection step between the private Kaggle Gateway and ChatGPT.

The Kaggle credentials remain on the private machine. The MCP server remains bound to loopback at:

```text
http://127.0.0.1:8000/mcp
```

`tunnel-client` makes an outbound HTTPS connection to OpenAI and forwards MCP JSON-RPC requests to
that private endpoint. No inbound firewall port is required.

## Required OpenAI-side prerequisites

1. Create an MCP tunnel in OpenAI Platform tunnel settings.
2. Associate the tunnel with the ChatGPT context/workspace that will use the app.
3. Obtain:
   - the `tunnel_id`;
   - a runtime API key for `tunnel-client`.
4. Ensure the account has the tunnel permissions required to use the tunnel.
5. Enable ChatGPT Developer Mode. On Pro, the current target is the read/fetch MCP surface.

Do not commit the runtime API key.

## 1. Start the private Kaggle Gateway

In PowerShell, in a process that already contains the six working Kaggle credential environment
variables:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\start-private.ps1
```

Keep that process running.

## 2. Install `tunnel-client`

Use the download offered by OpenAI Platform tunnel settings or the current official OpenAI
`tunnel-client` release. Do not pin a stale download URL in this repository.

Verify:

```powershell
tunnel-client help quickstart
```

## 3. Inject the tunnel runtime key into the process

Set the control-plane key only in the local shell/session or another secure secret mechanism:

```powershell
$env:CONTROL_PLANE_API_KEY = "<runtime key>"
```

Never put the value in this repository, ChatGPT, logs, or a command-line argument.

## 4. Initialize, diagnose, and run the tunnel

Use the helper shipped in this repository:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\start-tunnel.ps1 `
  -TunnelId "tunnel_..."
```

The helper:

1. confirms `tunnel-client` exists;
2. confirms `CONTROL_PLANE_API_KEY` exists without printing it;
3. confirms the private gateway is listening on `127.0.0.1:8000`;
4. initializes a named tunnel profile targeting `http://127.0.0.1:8000/mcp`;
5. runs `tunnel-client doctor --profile kaggle-direct --explain`;
6. starts `tunnel-client run --profile kaggle-direct`.

For an existing profile:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\start-tunnel.ps1 `
  -TunnelId "tunnel_..." `
  -SkipInit
```

For validation only:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\start-tunnel.ps1 `
  -TunnelId "tunnel_..." `
  -SkipInit `
  -DoctorOnly
```

## 5. Create the developer-mode app in ChatGPT

In ChatGPT web Developer Mode, create a custom app and choose **Tunnel** as the connection type.
Select the tunnel or paste the valid `tunnel_id`, then scan tools.

Expected initial tool set:

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

## 6. First live validation from ChatGPT

The first call must be read-only:

```text
kaggle_auth_check_all(max_workers=6)
```

Only if the six intended accounts return `auth_ok=true`, continue with:

```text
kaggle_kernels_inventory_all(
  search="pneumonia-v6-2-2",
  page_size=20,
  max_workers=6
)
```

Then recover existing runs via status, logs, and the selected output manifest. Do not submit new
Kaggle compute before the existing runs are classified.

## Troubleshooting boundary

If ChatGPT cannot discover tools:

- keep `tunnel-client run` alive;
- run `tunnel-client doctor --profile kaggle-direct --explain`;
- verify the tunnel is associated with the intended ChatGPT context/workspace;
- verify the operator has tunnel Use permission;
- verify `127.0.0.1:8000` is reachable from the tunnel-client host.

The gateway must not be made public as a workaround.
