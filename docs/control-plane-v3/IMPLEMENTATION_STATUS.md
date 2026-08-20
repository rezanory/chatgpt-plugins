# Control Plane v3 implementation status

## Implemented on `feature/control-plane-v3`

### Shared control plane

- Unified provider/safety capability registry for Kaggle, Cloudflare, and GitHub.
- Transport-priority router with explicit fallback behavior.
- Runtime CLI capability discovery (`scripts/control_plane_discover.py`).
- Idempotent runner bootstrap for Kaggle CLI/kagglehub, Wrangler, and portable GitHub CLI.
- Fixed-host GitHub/Cloudflare REST/GraphQL fallback with recursive secret redaction and safety-class grants.
- Weekly/manual capability-discovery workflow and hashed artifact manifest.
- Baseline capability matrix and no-delay routing policy.

### Kaggle

- Existing `run_kaggle(args)` is explicitly promoted to the universal official CLI transport.
- Added credential-free CLI `--version` and `--help` discovery helpers.
- Universal internal Cloudflare Kaggle service caller supports arbitrary valid Kaggle service/method identifiers while keeping the host fixed to `api.kaggle.com`.
- Read calls are directly available to protected bridge code; non-read calls require a scoped ephemeral control capability.
- Multi-account mapping remains explicit and credential-isolated.
- CLI/REST/kagglehub are all declared transports rather than kernel-only helpers.

### Cloudflare

- Wrangler runtime discovery covers all command families exposed by the installed version.
- Generic Cloudflare REST/GraphQL fallback fixes the host to `api.cloudflare.com` and requires explicit safety grants for mutations.
- Existing Cloudflare skill/plugin remains the preferred direct route when present.
- Temporary-bridge policy requires scoped capabilities, mutation lock, cleanup, and restore attestation.

### GitHub

- Native GitHub connector remains first choice.
- GitHub CLI is bootstrapped as runner fallback.
- `gh api`/REST/GraphQL are declared fallbacks for surfaces missing from the native connector.
- Generic GitHub REST/GraphQL fallback fixes the host to `api.github.com` and redacts sensitive fields.

## Deliberately not merged/deployed yet

The V6.2.3-P0 execution currently owns a temporary Cloudflare bridge and the canonical-Worker restore sequence has not completed. Merging a branch that changes `deploy/cloudflare-worker-free` could trigger unrelated deployment automation while that bridge is active.

Therefore:

- no Control Plane v3 code is deployed to the canonical Worker yet;
- the current P0 execution is not modified or restarted;
- merge/deploy is gated on P0 terminal receipt plus successful canonical Worker cleanup/restore.

## Post-P0 activation gate

Before activation:

1. P0 must be terminal and its expected receipt validated (success or reviewed terminal failure).
2. Temporary P0 capability must be deleted.
3. Canonical Cloudflare Worker restore must be attested.
4. V3 branch must pass Python tests/lint/compile and Worker TypeScript check.
5. Capability discovery must run once and produce a hashed inventory artifact.
6. Only then may V3 be merged/deployed.

## Meaning of "complete"

V3 removes artificial project-side command restrictions by combining runtime discovery with official CLI, REST/GraphQL, and provider SDK fallbacks. Provider-enforced constraints still exist: account permissions, API availability, rate limits, plan/entitlement, regional restrictions, and operations that a provider does not expose programmatically. Those are surfaced as exact provider constraints rather than being mislabeled as missing project capability.
