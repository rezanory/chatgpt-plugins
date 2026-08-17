# Implementation Status — V0.1

## Current architecture

```text
READ / RECOVERY
ChatGPT Pro -> custom MCP (read-only) -> Render Free -> direct KaggleApi -> Kaggle

WRITE / EXECUTION
ChatGPT -> connected GitHub app -> signed control Issue -> GitHub repository webhook
        -> Render Free -> direct KaggleApi -> Kaggle
        -> receipt/failure comment -> ChatGPT reads it through GitHub
```

The user's PC is not part of the runtime. Kaggle CLI, browser sessions, and GitHub Actions are not
used for Kaggle authentication or execution.

## Hosting policy

V0.1 is explicitly free-only:

- active host: Render Free Web Service;
- `render.yaml` uses `plan: free`;
- active Docker package: `deploy/render-free/Dockerfile`;
- paid Cloudflare Containers path removed and forbidden by the security gate.

## Direct Kaggle authentication

Every enabled account uses an isolated `KaggleApi()` and the operator-proven sequence:

```python
api = KaggleApi()
api.set_config_value(api.CONFIG_NAME_USER, username)
api.set_config_value(api.CONFIG_NAME_KEY, token)
api.authenticate()
api.kernels_list(page_size=1)
```

## Enabled accounts

```text
kg-01 -> azadka
kg-02 -> radlinaradlina
kg-04 -> reyhanehazad
kg-05 -> trickermark
kg-06 -> msdenis
kg-07 -> nisabulutmark
```

`kg-03` remains disabled pending exact owner-slug resolution.

## Read surface

- `kaggle_accounts`
- `kaggle_auth_check`
- `kaggle_auth_check_all`
- `kaggle_kernels_list`
- `kaggle_kernels_inventory_all`
- `kaggle_kernel_status`
- `kaggle_kernel_logs`
- `kaggle_kernel_output_manifest`

## Write/control surface

V0.1 exposes one narrow write action through the signed GitHub webhook control plane:

```text
rerun_existing
```

A control Issue:

- must have title prefix `[KAGGLE-RUN]`;
- must contain `<!-- chatgpt-plugins-kaggle-control:v1 -->`;
- must use schema `chatgpt.kaggle.control/v1`;
- must be opened in `rezanory/chatgpt-plugins` by an allowlisted actor;
- must pass GitHub webhook HMAC verification;
- must reference an `owner/kernel` matching the selected logical account.

The gateway writes a persistent claim marker before execution to prevent duplicate `job_id`
redelivery, then uses direct Python API calls:

```python
api.kernels_pull(..., metadata=True)
api.kernels_push(...)
```

The final receipt or redacted failure is posted back to the same Issue.

## Current source validation

The functional validation after introducing the free runtime and direct control path has shown:

- Security Gate: PASS;
- Python tests: 41 PASS;
- Python compile: PASS;
- Render Free production Docker image build: PASS.

The final consolidated CI is being used to close the remaining Ruff/style-only findings before live
activation.

## Activation remaining

One-time account-side actions still required because Render is not exposed as a connector in this
ChatGPT session:

1. Connect Render to the private GitHub repo `rezanory/chatgpt-plugins`.
2. Create the Blueprint from root `render.yaml`; it is explicitly `plan: free`.
3. Fill only the `sync: false` secrets:
   - six Kaggle API tokens for kg-01,02,04,05,06,07;
   - `CGP_GITHUB_WEBHOOK_SECRET`;
   - `CGP_GITHUB_TOKEN` restricted to this repository with Issues read/write.
4. Configure one GitHub repository webhook:
   - payload URL: `https://<render-service>.onrender.com/github/webhook`;
   - content type: `application/json`;
   - secret: same `CGP_GITHUB_WEBHOOK_SECRET`;
   - event: Issues only.
5. Connect the private MCP endpoint emitted by the Render service to ChatGPT Developer Mode.

## First live sequence after activation

Do not submit new compute first. Run:

```text
kaggle_auth_check_all(max_workers=6)
kaggle_kernels_inventory_all(search="pneumonia-v6-2-2")
status/logs for the six existing shards
artifact/fingerprint recovery
```

Evidence target:

```text
KAGGLE_EXECUTION_V62_2
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

Only after those existing runs are classified should `rerun_existing` be used from ChatGPT.
