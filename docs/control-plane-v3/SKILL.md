---
name: kaggle-cloudflare-github-control-plane-v3
description: Operate Kaggle, Cloudflare, and GitHub as one capability-discovered control plane with safe read/write/compute/destructive routing and runtime CLI/API discovery.
---

# Kaggle + Cloudflare + GitHub Control Plane v3

## Goal

Provide the broadest practical control-plane coverage without hard-coding a small set of commands. The skill must discover installed CLI/API capabilities at runtime, route each operation through the best available transport, and fail closed when permissions or service support are absent.

"Complete" means:

1. cover every capability exposed by the installed official CLI/API transports that the account is permitted to use;
2. discover newly added commands instead of silently ignoring them;
3. preserve read/write/compute/destructive separation;
4. never fabricate a capability that the provider, account, plan, or token does not expose;
5. prefer provider-native large-artifact paths over Cloudflare/GitHub transit.

## Provider routing order

### GitHub

1. Use the native GitHub connector when the requested action exists there.
2. For functionality not exposed by the connector, use `gh` on an authorized runner.
3. Use `gh api` REST/GraphQL for endpoints not surfaced by first-class `gh` commands.
4. Discover available commands at runtime with `gh --help`, `gh <group> --help`, and authenticated permission probes.

Coverage families include repositories, contents/Git data, branches/tags, commits/status/checks, pull requests/reviews, issues/comments/reactions, Actions/workflows/runs/jobs/artifacts/caches/runners, environments/deployments, releases, packages, Pages, projects, organizations/teams/members, gists, Codespaces, discussions, webhooks/apps, search, security/Dependabot/code scanning/secret scanning/dependency graph/advisories, attestations, rulesets, variables/secrets, billing/usage where permitted, and REST/GraphQL fallback.

### Cloudflare

1. Prefer the installed Cloudflare skill/plugin in products where its direct tools are available.
2. Otherwise use project-local Wrangler on the authorized self-hosted runner.
3. Use Cloudflare REST/GraphQL APIs when Wrangler lacks a first-class command.
4. Discover command groups at runtime with `wrangler --help` and provider docs/schema, never a stale hand-maintained subset.

Coverage families include Workers, versions/deployments, Pages, KV, D1, R2, Queues, Workflows, Pipelines, Hyperdrive, Vectorize, AI/AI Search, Browser Rendering, Containers, Durable Objects, Workers for Platforms, VPC, Tunnel, certificates, secrets/secrets-store, service bindings, cron triggers, tail/observability, analytics, DNS/zones/rules/WAF where the token permits, Zero Trust/API operations when permissions permit, and direct REST/GraphQL fallback.

### Kaggle

1. Prefer official `kaggle` CLI for user-facing Kaggle resources and workflows.
2. Use Kaggle REST services for lower-level status/control or when CLI output is insufficient.
3. Use `kagglehub` for Python-native dataset/model/competition resource flows when appropriate.
4. Discover the installed CLI command tree at runtime with `kaggle --help` and group help.

Coverage families include competitions (including host/admin flows when authorized), datasets, kernels/notebooks/scripts, models, model variations and versions, files/inbox uploads, forums/discussions/topics/comments, benchmarks/tasks/model proxy, auth, config, accelerator quota, unified search, submissions/leaderboards/episodes/replays/logs, dataset/model/kernel metadata, downloads/uploads/versioning/deletes, and raw REST service calls for capabilities not wrapped by CLI.

## Runtime discovery is mandatory

Before claiming an operation is unavailable:

- run/read the provider's current capability discovery;
- inspect account/token permissions;
- check first-class transport, then fallback transports;
- only then report a provider/permission limitation.

Use `scripts/control_plane_discover.py` to produce a sanitized machine-readable command inventory.

## Safety classes

Every operation is classified as one of:

- `read`: list/get/search/status/log/output metadata; no mutation.
- `write`: creates or edits metadata/content but does not start compute or destroy state.
- `compute`: starts/restarts/submits execution, scoring, training, benchmark, deployment, build, or runner workload.
- `destructive`: delete/cancel/purge/revoke/rotate/transfer/force-update operations.
- `privileged`: organization/account/security/billing/host-admin operations requiring elevated permissions.

Rules:

- Read operations may run without extra confirmation when relevant.
- Write/compute/destructive operations require current user authorization for the concrete task and scope.
- Destructive or privileged actions require explicit scope and must not be inferred from a read request.
- Unknown/new CLI commands are not blocked: if read-only status cannot be proven, route them with an explicit non-read safety class and grant.
- Generic REST read-POST is GraphQL-only; normal REST mutations cannot be relabeled as read.
- Raw Kaggle REST methods classified as read must be read-like (`Get/List/Search/Query/Read/Fetch/Download/Check/Describe/Validate`).
- Never turn a transport/probe error into a compute failure.
- Never rerun Kaggle compute automatically after an ambiguous failure.

## Live execution observability

Every newly generated Kaggle compute script should emit coarse, non-sensitive phase markers to stdout:

```text
CGP_PHASE:BOOTSTRAP
CGP_PHASE:DATA_AUDIT
CGP_PHASE:TRAIN_SEED_42_HEAD
CGP_PHASE:TRAIN_SEED_42_FINETUNE
CGP_PHASE:CAL_SEED_42
CGP_PHASE:SHADOW_SEED_42
CGP_PHASE:TRAIN_SEED_2026_HEAD
CGP_PHASE:TRAIN_SEED_2026_FINETUNE
CGP_PHASE:CAL_SEED_2026
CGP_PHASE:SHADOW_SEED_2026
CGP_PHASE:RECEIPT
CGP_PHASE:COMPLETE
```

Rules:

- phase markers must not contain labels, predictions, secrets, metrics, thresholds, or patient/sample identifiers;
- use the read-only `kagglePhaseProbe`/`kaggleLiveLog` path to inspect the bounded live log where Kaggle exposes it;
- sanitize bounded log output before returning it to the control plane;
- absence of a phase marker is `PHASE_UNKNOWN`, not a compute failure;
- phase telemetry must never be used to tune a locked-test execution.

This lets the operator distinguish TRAIN vs CAL vs SHADOW without reopening evidence or waiting for the terminal receipt.

## Ephemeral capability contract

Non-read Cloudflare/Kaggle bridge capabilities must be time-bounded and narrowly scoped:

- `CGP_PROJECT_CONTROL_TOKEN`: random masked token, minimum 32 characters;
- `CGP_CONTROL_SCOPES`: exact or hierarchical scopes such as `kaggle:compute:kernels.KernelsApiService:SaveKernel` or `kaggle:compute:kernels.KernelsApiService:*`;
- `CGP_CONTROL_EXPIRES_AT`: required expiry in epoch seconds/milliseconds or an ISO timestamp;
- global `*` scope is rejected unless `CGP_CONTROL_ALLOW_GLOBAL_SCOPE=1` is separately and explicitly set;
- capability cleanup/revocation is mandatory after the operation.

## Kaggle scientific governance

For ML projects with locked test data:

- never reopen a locked test just to obtain more diagnostics;
- old locked/external cohorts are historical evidence, not a new blind test;
- do not retune model/weights/calibration/threshold after locked-test feedback and then claim the same holdout as unbiased;
- bind model/config/calibrator/threshold artifacts by explicit IDs and hashes;
- use Kaggle-native datasets/model artifacts for large files;
- validate terminal receipts and hashes, not just `COMPLETE`.

## Cloudflare mutation governance

- Prefer separate Worker names/routes for temporary bridges.
- If the canonical Worker name must be reused, acquire the global mutation concurrency lock.
- Record predecessor version when possible.
- Use ephemeral, masked, narrowly scoped, expiring capability tokens.
- Cleanup must delete temporary capability and restore canonical Worker.
- Restore must be attested by health/route/version checks; cleanup failure must fail visibly.

## GitHub governance

- GitHub is the control plane/journal, not the large-artifact plane.
- Use Actions artifacts for small receipts and short-lived source packages only.
- Preserve exact run/job/artifact IDs and digests.
- Do not confuse GitHub Actions run IDs with Kaggle kernel/session IDs.
- Prefer native connector operations; use runner `gh` fallback only for missing capabilities.

## No-delay operating pattern

1. Resolve provider(s) and required safety class.
2. Read current capability inventory and permissions.
3. Choose the highest-fidelity available transport.
4. Execute once with idempotency/duplicate guards.
5. Persist receipt/provenance.
6. If a transport fails, try a safe alternate transport before asking the user for logs or credentials already present in the control plane.
7. Never delay a task merely because the first transport lacks a helper function.

## Capability freshness

The static matrix in `CAPABILITY_MATRIX.md` is a baseline, not the source of truth. Runtime discovery and current official docs win when they differ.
