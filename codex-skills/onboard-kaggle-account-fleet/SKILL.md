---
name: onboard-kaggle-account-fleet
description: Securely discover, onboard, migrate, and validate authorized Kaggle account fleets backed by GitHub Environments and Cloudflare Workers. Use when adding one or many Kaggle accounts, expanding the current pool toward hundreds of accounts, promoting Kaggle credentials to Cloudflare without exposing values, updating canonical account routing, sharding Workers when binding limits require it, or proving N/N readiness before a new project starts.
---

# Onboard Kaggle Account Fleet

Use this skill to make Kaggle account onboarding repeatable, idempotent, fast, and secret-safe.

Read `references/ARCHITECTURE.md` before changing Worker topology. Read `references/RUNBOOK.md` before executing a live onboarding batch. Use `scripts/plan_fleet.py` before any write.

## Core contract

- Treat `rezanory/chatgpt-plugins` and GitHub `main` as source of truth.
- Reconcile remote/local and current Cloudflare state before writes.
- Never print, return, commit, artifact, or persist plaintext Kaggle/Cloudflare credentials.
- Use only explicitly authorized accounts. Do not use multi-account orchestration to evade Kaggle rules, quotas, or platform restrictions.
- Keep account IDs, owner slugs, GitHub environments, Cloudflare secret names, shard IDs, and routes deterministic.
- Make every operation idempotent: already-correct accounts must be skipped, not re-authored.
- Use read-only Kaggle identity/quota probes for readiness. Do not launch compute during onboarding.
## Standard workflow

1. Fetch `main`, `plugins/kaggle/config/accounts.json`, active Cloudflare runtime files, security policy, and current workflow state.
2. Inventory GitHub Environments matching `kaggle-*`; record only environment names and secret presence.
3. Inventory Cloudflare secret names only; never request secret values back from Cloudflare.
4. Build a normalized fleet plan with `scripts/plan_fleet.py` using existing registry plus new account metadata.
5. Discover missing owner slugs with a read-only `kagglehub.whoami()` probe inside each account environment.
6. Select topology from the plan: single Worker while safely below the binding budget; sharded Workers once the threshold is crossed.
7. Promote credentials securely. Prefer Cloudflare bulk secret upload per shard. If credentials exist only in isolated GitHub Environments, use the trusted self-hosted Windows DPAPI staging pattern in `references/RUNBOOK.md`.
8. Update canonical registry and runtime routing from the plan; do not hand-edit account lists in multiple files.
9. Dry-run/bundle the exact Worker surfaces, deploy, then verify health from a trusted self-hosted runner.
10. Probe every account read-only through Cloudflare and require exact `N/N` success before declaring ready.
11. Remove temporary workflows, temporary OIDC trust, staging files, and direct GitHub-secret execution paths.
12. Re-run focused security/config/runtime gates and record a secret-free handoff.

## Scaling rules

- Default shard capacity: 48 account secrets. Keep headroom below Cloudflare binding limits.
- Assign accounts deterministically by numeric account ID; never reshuffle existing accounts merely because the fleet grew.
- Probe at most 6 Kaggle calls concurrently inside one Worker request.
- Validate shards independently, then aggregate readiness centrally.
- When projected fleet exceeds the single-Worker safe threshold, perform the one-time sharding migration before adding further accounts.
- For hundreds of accounts, use a router plus credential shards; the router must not hold every Kaggle token.
## Required completion evidence

- Canonical registry contains every expected account exactly once.
- Every expected Cloudflare secret name exists in the correct shard.
- Worker/router code recognizes every planned account and no unplanned account.
- Exact build/dry-run and deploy checks pass.
- Read-only identity/quota checks return `N/N` success through the production Cloudflare route.
- No secret value appears in logs, artifacts, commits, issues, or reports.
- Temporary promotion/readiness workflows and temporary trust entries are removed.
- Security policy passes for the permanent architecture, or any pre-existing unrelated failures are explicitly separated.

## Failure handling

Fix forward only. Do not delete working credentials because a later validation fails. If a batch partially succeeds, reconcile Cloudflare secret-name inventory and registry state, then resume only missing accounts. Preserve successful shards while repairing a failed shard.

Do not interpret an aggregate 401/403 as a credential failure until OIDC workflow/ref/event admission is reconciled. After temporary onboarding trust is removed, use the permanent allowlisted per-account read path; do not re-authorize a retired readiness workflow merely to make its old job green. A request comment without a matching workflow run or durable receipt is missing transport evidence, not a failed account.

If Cloudflare or GitHub limits have changed, stop using cached assumptions and re-check official current limits before selecting topology.

## Output

Report: total discovered, newly onboarded, already ready, failed, shard count, per-shard readiness, aggregate GPU/TPU quota when available, canonical commit SHA, deployed Worker versions, and cleanup/security status. Never include credential values.
