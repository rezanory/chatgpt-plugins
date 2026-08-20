# No-delay routing policy

This policy prevents the control plane from becoming artificially limited by whichever helper happens to be loaded first.

## Core rule

A missing first-class helper is not a provider limitation.

Before asking the operator for manual intervention, the agent must try the provider transports in order and record why each unavailable transport cannot satisfy the request.

## GitHub route

1. Native GitHub connector/action.
2. GitHub CLI first-class command.
3. `gh api` REST.
4. `gh api graphql` for GraphQL-only surfaces.
5. One-shot Actions workflow only when the operation must execute inside the runner environment.

Do not use a one-shot workflow when the native connector can perform the operation directly.

## Cloudflare route

1. Installed Cloudflare plugin/direct tool when available in the current product.
2. Project-local Wrangler on the authorized runner.
3. Cloudflare REST API.
4. Cloudflare GraphQL Analytics API for analytics surfaces.
5. One-shot Worker bridge only when a credential must remain inside Cloudflare or edge execution is required.

Do not overwrite the canonical Worker merely because a first-class endpoint is missing. Prefer a separate Worker name/route. When the same Worker name is unavoidable, use the global mutation lock and restore attestation.

## Kaggle route

1. Official Kaggle CLI for resource-level operations.
2. Kaggle REST service calls for lower-level status, output metadata, quota, or capabilities not wrapped by CLI.
3. `kagglehub` for Python-native dataset/model/competition flows.
4. Protected Cloudflare bridge when Kaggle credentials must remain in Cloudflare.
5. One-shot GitHub Actions wrapper only to orchestrate the above transports, never as a reason to expose credentials.

## Operation resolution

For every request, record:

- provider;
- capability family;
- safety class (`read`, `write`, `compute`, `destructive`, `privileged`);
- selected transport;
- alternate transports checked;
- required permission/account/plan;
- idempotency/duplicate key;
- expected receipt/evidence;
- cleanup requirements.

## Read operations

Read operations should be attempted immediately through all safe transports before reporting a limitation. Examples: status, logs, output inventory, quota, command discovery, metadata, run/job state, analytics queries, repository files, dataset/model metadata.

## Mutation operations

Mutations require concrete user authorization for the task. Once authorized, the control plane should not stop just because a preferred helper is missing; it should move to the next official transport while preserving the same scope.

## Compute operations

Compute actions require duplicate detection and a launch receipt. A transport error after submission must not trigger an automatic second submission.

## Destructive and privileged operations

Deletion, cancellation, secret revocation/rotation, force-updates, transfers, host/admin actions, billing/account changes, security-policy changes, and similar operations must use explicit scope and fail closed.

## Runtime discovery

Use `scripts/control_plane_discover.py` before claiming a CLI command is absent. Use current official API documentation before claiming an API capability is absent. Installed CLI help and current provider docs override this repository's static baseline.

## Runner bootstrap

Use `scripts/bootstrap_control_plane_tools.ps1` to ensure the runner has:

- current Kaggle CLI and `kagglehub`;
- project-local Wrangler;
- portable GitHub CLI when not already installed.

The bootstrap is idempotent and does not print credential values.

## Evidence hierarchy

1. Provider-native receipt/status/output.
2. GitHub Actions job/artifact receipt with digest.
3. Cloudflare deployment/version/health attestation.
4. Sanitized logs.

Green CI alone is never sufficient evidence of a scientific or compute PASS when a provider-native receipt is expected.
