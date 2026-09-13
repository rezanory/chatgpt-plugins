# Fleet Architecture

## Current constraints to re-check before large batches

As of 2026-09-13, Cloudflare documents 64 environment variables/secrets per Worker on Free and 128 on Paid. Worker bulk secret upload supports up to 100 operations per command. Secrets Store open beta documents 100 production secrets per account, so it is not sufficient by itself for a 300-account fleet.

Official references:
- https://developers.cloudflare.com/workers/platform/limits/
- https://developers.cloudflare.com/workers/wrangler/commands/workers/
- https://developers.cloudflare.com/secrets-store/manage-secrets/

Never assume these limits are permanent; verify them again when a batch would approach a boundary.

## Topology

Use two modes:

1. `single-worker`: use the existing `chatgpt-kaggle-gateway` while projected account secrets fit comfortably below the Worker binding limit.
2. `sharded`: use one public/control router plus private credential shards. Each shard holds only its assigned Kaggle account secrets.

Default shard capacity is 48 account secrets. This leaves Free-plan headroom for MCP/control/recovery bindings and avoids operating at the hard 64-variable boundary.
## Deterministic shard mapping

Parse the numeric suffix from `kg-NN` or `kg-NNN`. For capacity `C`, assign `shard_index = floor((numeric_id - 1) / C)`. This guarantees that adding higher IDs never moves existing accounts.

Recommended names:
- Account: `kg-01` ... `kg-300`
- GitHub Environment: `kaggle-01` ... `kaggle-300`
- Environment secret: `KAGGLE_API_TOKEN`
- Cloudflare account secret: `CGP_KAGGLE_KG01_TOKEN` ... `CGP_KAGGLE_KG300_TOKEN`
- Shard Worker: `chatgpt-kaggle-gateway-s00`, `s01`, ...
- Router: `chatgpt-kaggle-gateway`

Keep owner slug and role in canonical non-secret metadata. Do not encode owner identity into secret values.

## Request routing

The router resolves `account_id -> shard_id` from canonical metadata and sends one request to the target shard. Prefer Cloudflare Service Bindings for internal shard calls when available in the deployed architecture. Keep OIDC and authorization enforcement at the router and fail closed on unknown accounts.

A shard derives the secret binding name from canonical metadata and accesses only its local assigned account. Do not let callers choose arbitrary secret binding names.

## Readiness

Within a shard, cap outbound Kaggle concurrency at 6. Keep a readiness request below the applicable subrequest limit; with capacity 48, one quota probe per account fits the current Free-plan 50-subrequest boundary. For a larger shard or changed platform limit, split readiness into pages.

Validate each shard independently and aggregate only secret-free results: account ID, owner slug, auth status, GPU/TPU quota, refresh time, and error class.