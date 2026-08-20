# Kaggle provider — Control Plane v3

Kaggle is the compute/data provider in the `chatgpt-plugins` control plane. The original V0.1 issue-driven dispatcher remains compatible, but V3 removes the artificial assumption that the provider consists of only a handful of kernel helpers.

```text
ChatGPT / Operator
  -> GitHub native connector or Actions control plane
  -> Cloudflare protected gateway when credentials must remain at the edge
  -> Kaggle CLI / Kaggle REST services / kagglehub
  -> Kaggle resources and compute
```

## V3 operating principles

- Runtime capability discovery is authoritative (`kaggle --help` and group help).
- `run_kaggle(args)` is the universal official CLI transport and always uses `shell=False`.
- Resource-specific helpers are convenience wrappers, not capability boundaries.
- Low-level Cloudflare bridges can use the scoped universal Kaggle service caller for REST service methods not wrapped by CLI.
- `kagglehub` is available as a Python-native fallback for datasets, models, and competition resources.
- Read, write, compute, destructive, and privileged operations are explicitly separated.
- A missing helper must not be reported as "Kaggle cannot do this" until CLI, REST, and SDK routes have been checked.

## Capability families

V3 covers the currently exposed official surfaces for:

- authentication and configuration;
- accelerator quota;
- unified search;
- competitions, including submissions, leaderboards, episodes/replays/logs and authorized host/admin flows;
- datasets and dataset versions;
- kernels/notebooks/scripts, source, metadata, output, status and logs;
- models, model variations, and variation versions;
- file/inbox and resumable upload flows;
- forums/discussions/topics/comments;
- benchmarks/tasks/model-proxy workflows;
- raw Kaggle REST service calls for capabilities not wrapped by the installed CLI;
- `kagglehub` Python resource flows.

The baseline matrix is in `docs/control-plane-v3/CAPABILITY_MATRIX.md`. The installed command tree is discovered by `scripts/control_plane_discover.py`.

## Multi-account execution

The canonical account mapping remains explicit and credential-isolated. Large parallel workloads should use separate accounts and provider-native storage rather than routing large artifacts through GitHub or Cloudflare.

## Source and artifact transfer

The relay may package already-authorized GitHub source into a temporary private Kaggle Dataset and run a private Kernel that references it. Large checkpoints and durable ML artifacts remain on Kaggle-native storage; GitHub is for code, journal entries, manifests, and small receipts.

## Safety and scientific governance

- Do not automatically rerun compute after ambiguous transport failures.
- Validate expected receipts/hashes instead of treating `COMPLETE` alone as PASS.
- Preserve locked-test discipline and never retune from locked-test feedback and then reuse the same holdout as unbiased evidence.
- Mutating generic REST calls require a protected scoped capability; generic read calls do not expose credentials.
- Destructive/privileged operations require explicit user scope.

## Related V3 files

- `docs/control-plane-v3/SKILL.md`
- `docs/control-plane-v3/CAPABILITY_MATRIX.md`
- `docs/control-plane-v3/NO_DELAY_ROUTING.md`
- `scripts/control_plane_discover.py`
- `scripts/bootstrap_control_plane_tools.ps1`
- `packages/core/src/chatgpt_plugins_core/capability_registry.py`
- `deploy/cloudflare-worker-free/src/control-plane-v3-kaggle.ts`
