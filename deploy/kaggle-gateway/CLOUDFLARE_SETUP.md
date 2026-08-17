# Cloudflare Tunnel fallback — remote HTTPS MCP

This is an optional fallback when Secure MCP Tunnel is unavailable or inconvenient for the current ChatGPT plan.

The Kaggle runtime remains unchanged:

```text
ChatGPT
  -> HTTPS MCP hostname
  -> Cloudflare Tunnel
  -> http://127.0.0.1:8000/mcp
  -> Kaggle Direct Gateway
  -> isolated KaggleApi instances
  -> Kaggle API
```

Cloudflare is transport only. It never replaces `KaggleApi`, never receives Kaggle credentials from Git, and never invokes the Kaggle CLI.

## Cloudflare configuration

Cloudflare recommends remotely-managed tunnels for most use cases.

1. In Cloudflare Dashboard, open **Networking > Tunnels**.
2. Create a remotely-managed Tunnel such as `chatgpt-kaggle-gateway`.
3. Add a **Published application** route.
4. Choose a hostname on a domain already managed by Cloudflare, for example:

```text
kaggle-mcp.example.com
```

5. Set the service URL to:

```text
http://localhost:8000
```

6. Obtain the Tunnel token from the tunnel's **Add a replica** / installation command.
7. Do not commit that token.

Cloudflare documentation:

- https://developers.cloudflare.com/tunnel/setup/
- https://developers.cloudflare.com/tunnel/advanced/tunnel-tokens/

## Run locally on Windows

First make sure the six enabled Kaggle credentials are available to the Gateway process, or use the interactive launcher that prompts for them.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\start-cloudflare.ps1
```

If `TUNNEL_TOKEN` is not already defined, the launcher prompts for it as a SecureString and keeps it process-local.

The launcher:

1. starts the existing loopback-only Kaggle Gateway;
2. verifies `127.0.0.1:8000` is reachable;
3. runs the remotely-managed tunnel with the provided token;
4. stops the Gateway and clears the process-local Tunnel token when the launcher exits.

If the Gateway is already running:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\start-cloudflare.ps1 -GatewayAlreadyRunning
```

## Security

Do not expose the origin port publicly. The Gateway stays on loopback.

If the HTTPS hostname is internet-reachable, protect it with a compatible authentication layer before treating it as production. Cloudflare Access is suitable for HTTP applications, but the final authentication mechanism must also be compatible with the ChatGPT custom-MCP connection flow in use.

Do not put Kaggle username/token pairs in Cloudflare configuration. They remain in the Gateway process only.

## Live verification before ChatGPT

Before connecting the remote MCP hostname, verify the six account connections directly with:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\kaggle-gateway\selftest-live.ps1
```

This uses only the Python `KaggleApi` path and performs:

```text
api = KaggleApi()
set user
set token
authenticate()
kernels_list(page_size=1)
```

then inventories matching `pneumonia-v6-2-2` kernels without starting new compute.

A sanitized JSON result can be exported by calling the underlying Python self-test with `--output` if needed for offline review.
