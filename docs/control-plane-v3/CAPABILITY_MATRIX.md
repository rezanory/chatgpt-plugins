# Control Plane v3 capability matrix

Baseline date: 2026-08-20. Runtime discovery is authoritative when it differs from this file.

## Kaggle

| Family | Read | Write | Compute | Destructive/admin | Preferred transport |
|---|---|---|---|---|---|
| Auth/config | auth state, config view | login/config set | — | revoke/unset | Kaggle CLI |
| Search | unified search across resources/users/discussions | — | — | — | Kaggle CLI/REST |
| Quota | GPU/TPU quota | — | — | — | Kaggle CLI/REST |
| Competitions | list/files/pages/topics/submissions/leaderboard/team/episodes/replay/logs | host pages/settings/team metadata where authorized | submit predictions/code | host/admin create/launch/update/delete where authorized | Kaggle CLI first, REST fallback |
| Datasets | list/search/files/metadata/status/download/topics | create/version/update metadata/upload | dataset processing/version finalize | delete | Kaggle CLI/kagglehub/REST |
| Kernels/notebooks/scripts | list/pull/status/logs/output/metadata | init/push/update metadata | run/save-and-run | delete/cancel when supported | Kaggle CLI/REST |
| Models | list/get/download/metadata/topics | create/update | version/upload workflows | delete | Kaggle CLI/kagglehub/REST |
| Model variations | list/get/download | create/update | variation version upload | delete | Kaggle CLI/kagglehub/REST |
| Variation versions | list/get/download/status | create/upload | model-version processing | delete | Kaggle CLI/kagglehub/REST |
| Files/inbox | list/upload status | upload/resumable upload | server-side finalize/compress | delete where exposed | Kaggle CLI/REST |
| Forums/discussions | forums/topics/messages/comments/search | create/reply where API permits | — | moderation where authorized | Kaggle CLI/REST |
| Benchmarks | leaderboard/tasks/status/log/models/topics/download | init/push/publish task | run benchmark/model proxy | delete task, privileged benchmark admin | Kaggle CLI/REST |
| Resource Python flows | dataset/model/competition download/cache/load | upload/version where supported | Python workflow execution | cache cleanup/local only | kagglehub |
| Raw service coverage | any authenticated read endpoint surfaced by current Kaggle services | explicit allowlisted writes | explicit compute endpoints | explicit destructive endpoints | REST service fallback |

Current official Kaggle CLI groups include competitions, datasets, kernels, models/model variations/versions, files, forums, benchmarks, config, auth, quota, and search. Competition hosting commands are also available when the account is authorized.

## Cloudflare

| Family | Read | Write | Compute/deploy | Destructive/admin | Preferred transport |
|---|---|---|---|---|---|
| Workers | list/config/versions/deployments/tails | vars/bindings/routes/settings | dev/deploy/version upload/rollback | delete Worker/version/route | Wrangler/API |
| Pages | projects/deployments/logs | project config | deploy/build | delete project/deployment | Wrangler/API |
| KV | namespaces/keys/values | put/bulk put | — | delete keys/namespaces | Wrangler/API |
| D1 | databases/schema/query/export | SQL execute/import | migrations/remote execution | delete DB | Wrangler/API |
| R2 | buckets/objects/domain/config | put/upload/config | multipart operations | delete object/bucket | Wrangler/API/S3 API |
| Queues | list/config/consumers | create/update consumers | send/consume | delete/purge | Wrangler/API |
| Workflows | list/instances/status | create/update | trigger/resume/retry | terminate/delete | Wrangler/API |
| Pipelines | list/config | create/update | run/ingest | delete | Wrangler/API |
| Hyperdrive | list/config | create/update | connection/runtime use | delete | Wrangler/API |
| Vectorize | list/index info/query metadata | create/upsert/delete vectors | index operations | delete index | Wrangler/API |
| AI / AI Search | models/bindings/instances | config/index sources | inference/indexing | delete instances | Cloudflare plugin/API/Workers AI |
| Browser Rendering | status/config | session config | browser jobs | terminate | Wrangler/API |
| Containers | list/status/images/config | create/update | run/deploy | delete/stop | Wrangler/API |
| Durable Objects | namespace/config/storage inspection where exposed | storage/RPC operations | alarms/WebSocket workloads | destructive storage/admin | Worker/API |
| Workers for Platforms | namespaces/scripts/dispatch | create/update | deploy tenant scripts | delete | Wrangler/API |
| VPC/networking/Tunnel | list/status/routes | config/routes | tunnel/connect | delete/revoke | Wrangler/API |
| Secrets / Secrets Store | names/metadata | put/rotate | — | delete/revoke | Wrangler/API |
| DNS/zones/rules/WAF | read zone/rulesets/settings | create/update records/rules | cache/config propagation | delete/purge/zone admin | REST/API plugin when token permits |
| Zero Trust | identities/apps/policies/tunnels | create/update policies | remote access flows | revoke/delete | REST/API when token permits |
| Observability/analytics | logs/tail/metrics/GraphQL analytics | dashboards/config | tail/query jobs | retention/admin | Wrangler/GraphQL/API |
| Certificates/domains | inspect certs/custom domains | issue/config | validation/deploy | revoke/delete | Wrangler/API |
| Account/billing | usage/subscription read | account settings where permitted | — | privileged admin | REST/API |

Wrangler runtime discovery must include command families exposed by the installed version, including Workers, Artifacts, Browser, Certificates, Containers, D1, Hyperdrive, KV, Pages, Pipelines, Queues, R2, Secrets Store, Tunnel, Vectorize, VPC, Workers for Platforms, and Workflows.

## GitHub

| Family | Read | Write | Compute | Destructive/admin | Preferred transport |
|---|---|---|---|---|---|
| Repositories | metadata/files/trees/blobs/branches/tags | create/update contents/refs/settings | template/fork/import tasks | delete/transfer/archive/admin | Native connector, `gh`, REST |
| Git data/commits | commits/diffs/statuses | refs/statuses/commits | merge/rebase server operations | force ref/delete tag/branch | Native connector/REST |
| Pull requests | PRs/files/reviews/threads/checks | create/update/comment/review/merge | merge queue/automerge | close/delete branch/admin merge policy | Native connector/gh/REST/GraphQL |
| Issues | issues/comments/labels/milestones/reactions/sub-issues | create/update/comment/assign/label | — | close/lock/delete where exposed | Native connector/gh/REST |
| Actions | workflows/runs/jobs/logs/artifacts/caches/runners | dispatch/config/variables/secrets | run/rerun/cancel runner workloads | delete runs/artifacts/caches/runners | Native connector/gh/REST |
| Checks/status | check suites/runs/statuses | create/update check/status | CI jobs | rerequest/delete where exposed | REST |
| Environments/deployments | envs/rules/deployments/status | create/update | deployment workflows | delete env/deployment rules | REST/gh |
| Releases | releases/assets/tags | create/update/upload | build/publish automation | delete release/asset/tag | gh/REST |
| Packages | packages/versions/download metadata | publish via package tooling | registry workflows | delete/restore versions | REST/package clients |
| Pages | status/builds/config | create/update config | build/deploy | delete site | REST/gh |
| Projects v2 | projects/items/fields | create/update items/fields | automation | delete project/item | GraphQL/REST where supported |
| Organizations/teams | members/teams/repos/roles/audit where permitted | manage membership/teams | org workflows | privileged org admin | REST/GraphQL |
| Codespaces | list/status/machines/secrets | create/update/publish | start/stop/rebuild | delete | gh/REST |
| Discussions | categories/discussions/comments | create/update/reply | — | delete/lock/moderate | GraphQL/REST |
| Gists | list/get | create/update | — | delete | gh/REST |
| Webhooks/apps | hooks/deliveries/installations/permissions | create/update/redeliver | delivery processing | delete/revoke | REST |
| Security | Dependabot/code scanning/secret scanning/dependency graph/advisories/code security/campaigns | dismiss/update/fix metadata | autofix/security workflows | delete/revoke/admin security config | Native connector/REST |
| Attestations/artifact metadata | read attestations/records | create attestations/metadata | provenance jobs | delete where supported | REST/gh |
| Rulesets/branch protection | read rules/rulesets | create/update | enforcement | delete/bypass admin | REST |
| Search | code/repos/issues/PRs/commits/users/topics | — | — | — | Native connector/gh/REST |
| Billing/usage/Copilot | usage/license/seat metrics | seat/settings where permitted | — | org/enterprise admin | REST |
| Raw REST/GraphQL | any endpoint allowed by token and service | explicit authorized mutation | explicit authorized compute | explicit authorized destructive/admin | `gh api` fallback |

## Capability resolution algorithm

For any requested operation:

1. Check the native connector/tool surface.
2. Check provider CLI discovery for a first-class command.
3. Check official REST/GraphQL/API surface.
4. Check provider-specific SDK (`kagglehub`, Cloudflare SDK/API, Octokit/gh API) if it adds functionality.
5. Probe permissions without mutation when possible.
6. If supported and authorized, execute through the highest-fidelity transport.
7. If not supported, return the exact missing permission/plan/API constraint rather than a generic "unavailable".

## Coverage guarantee

The matrix is intentionally broader than the current project endpoints. It does not promise access to features that the provider does not expose publicly or that the current account/token/plan forbids. The control plane must, however, discover and use all currently available official transports before declaring a limitation.
