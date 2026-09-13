# Live Onboarding Runbook

## Canonical surfaces

Repository: `rezanory/chatgpt-plugins`

Inspect before every batch:
- `plugins/kaggle/config/accounts.json`
- `deploy/cloudflare-worker-free/src/kaggle.ts`
- `deploy/cloudflare-worker-free/src/index.ts`
- `deploy/cloudflare-worker-free/src/control-plane-v3-index.ts`
- `deploy/cloudflare-worker-free/src/control-plane-v3-kaggle.ts`
- `deploy/cloudflare-worker-free/src/control-plane-v3-action.ts`
- `deploy/cloudflare-worker-free/src/control-plane-v3-oidc.ts`
- `scripts/security_gate.py`

Never trust stale local state over GitHub `main`.

## Discovery phase

1. Enumerate GitHub Environments matching `kaggle-*`.
2. Confirm secret presence only; do not retrieve values into chat output.
3. Enumerate Cloudflare Worker secret names only.
4. Read canonical registry and calculate missing, extra, stale, and conflicting accounts.
5. Run `scripts/plan_fleet.py` with existing registry and new metadata.
6. If owner slug is missing, run `kagglehub.whoami()` inside that Environment and emit only the username.
## Secure promotion from GitHub Environments

Use this only when a new credential already exists in GitHub Environment secrets and must be copied to Cloudflare.

- Run staging only on trusted self-hosted Windows runners.
- Bind one GitHub Environment per matrix job so only that job can see its `KAGGLE_API_TOKEN`.
- Immediately mask the value and convert it to `SecureString`.
- Persist only DPAPI-protected text under a dedicated temporary directory on the same machine/user context.
- Never upload the staged file as an Actions artifact and never send it through job outputs.
- A promotion job on the same runner identity reads staged DPAPI values, constructs a per-shard in-memory JSON map, and pipes it to `wrangler secret bulk --name <shard-worker>`.
- `wrangler secret bulk` accepts up to 100 operations; shard capacity should normally be lower.
- Delete the staging directory in a `finally` block even if promotion fails.
- Re-inventory Cloudflare secret names after promotion.

If the stage and promote jobs cannot guarantee the same Windows user/DPAPI context, do not use this bridge. Use a direct secure local import instead.

## Deployment phase

Patch from the generated plan, not ad hoc string lists. Prefer data-driven account metadata so future additions do not require touching multiple TypeScript account tables. Dry-run the exact target entry point before deployment.

Deploy shards first, validate shard health, then deploy/update the router. Keep the previous router compatible until every new shard is ready.
## Readiness gate

For every planned account, make a read-only Kaggle API call through the production Cloudflare route. Prefer accelerator quota statistics because it proves authentication and yields useful capacity without starting compute.

Require:
- exact account count equals plan count;
- every account returns auth/status OK;
- owner identity matches canonical metadata when identity is available;
- all shards report configured secret counts expected by the plan;
- router rejects unknown/unassigned account IDs;
- no mutation/compute action occurs during readiness.

Report exact `N/N`; do not round partial success up to ready.

## Cleanup phase

After success:
1. Remove temporary promotion, whoami, secret-inventory, and readiness workflows created only for the batch.
2. Remove temporary OIDC trust entries used only by those workflows.
3. Deploy hardened trust configuration.
4. Verify staging directories are absent on all involved self-hosted runners.
5. Run focused security gate and configuration checks.
6. Preserve a secret-free receipt with plan hash, commit SHA, Worker version IDs, and N/N readiness.

Do not delete GitHub Environment secrets automatically unless the user explicitly chooses Cloudflare as the sole credential store.