# ADR-007 — Cloudflare Plugin as Management Plane for Kaggle Control Plane

- Status: Accepted
- Date: 2026-10-07
- Repository: `rezanory/chatgpt-plugins`

## Context

The production Kaggle control plane is implemented by our Cloudflare Worker
`chatgpt-kaggle-gateway`. ChatGPT now has a connected Cloudflare plugin with
direct account-level management access.

The new plugin must not become an additional runtime hop in Kaggle execution.
Its purpose is management, inspection, diagnosis, and explicitly authorized
Cloudflare administration.

## Decision

### Execution plane — unchanged

```text
GitHub
  -> GitHub Actions
  -> self-hosted runner when required
  -> Radlina Kaggle Control Plane (Cloudflare Worker)
  -> Kaggle API
  -> Kaggle kernel / CPU / GPU
```

The ChatGPT Cloudflare plugin is NOT part of this execution path.

### Management plane — direct Cloudflare access

```text
ChatGPT
  -> Cloudflare Plugin
  -> Cloudflare account
  -> Worker deployments / settings / bindings / observability
```

Use the Cloudflare plugin first for Cloudflare-native inspection and
administration. Use the GitHub connector for repository/evidence operations.
Use Radlina Remote MCP / local Wrangler when local execution or a fallback
is required. Remote Desktop Commander is the final host-level fallback.

### Tool failover policy

Do not stop at the first unavailable tool when another approved path can provide
the same safe operation or evidence. Try the next path when failure is caused by
authentication, outage, transport failure, missing tool coverage, or a connector
limitation.

Preferred fallback order by scope:

| Scope | Primary | Secondary | Tertiary |
| --- | --- | --- | --- |
| Cloudflare management/read | Cloudflare Plugin | Radlina Remote MCP + Wrangler/API | Remote Desktop Commander |
| GitHub source/evidence | GitHub Connector | Radlina Remote MCP + local git/gh | Remote Desktop Commander |
| Windows/runner operations | Radlina Remote Full MCP | Remote Desktop Commander | GitHub remote evidence/actions where applicable |
| Kaggle execution/status | Existing GitHub/Runner/Control-Plane path | Alternate approved runner/account path | Direct operator recovery only when explicitly authorized |

Failover must preserve the same security boundary and source of truth. A failed
management connector does not justify bypassing OIDC, secret handling, write
gates, or repository-controlled deployment. Read-only diagnostic fallbacks may
be used automatically; state-changing fallbacks must remain auditable and must
not create configuration drift.

## Safety and drift rules

1. Default to read-first inspection.
2. Never expose secret values in chat, evidence, commits, or logs.
3. Secret metadata may be checked by name/type only when needed.
4. Normal Worker code changes remain source-controlled in GitHub.
5. Direct Cloudflare mutations must be explicitly intended and any persistent
   configuration/code change must be reconciled back to source control to avoid drift.
6. Do not change the production Kaggle execution chain merely because the
   Cloudflare plugin is available.
7. Preserve OIDC/allowlist controls and fail-closed behavior.

## Verified production snapshot

Verified directly through the connected Cloudflare plugin on 2026-10-07:

- Worker: `chatgpt-kaggle-gateway`
- Control repository binding: `rezanory/chatgpt-plugins`
- Allowed GitHub actor: `rezanory`
- `CGP_WRITE_ENABLED=0`
- Kaggle fleet secret bindings for kg-01 through kg-11 are present
- Latest deployed Worker version observed: `417`
- Latest deployment source: `wrangler`
- Latest deployment traffic: `100%` to version 417
- Workers subdomain is enabled
- Preview subdomains are disabled

No Worker code, secrets, routing, or runtime execution behavior was changed
while establishing this management-plane policy.

## Operational consequence

From this point forward, Cloudflare-specific troubleshooting can be performed
directly from ChatGPT without routing through the Windows host first, while
Kaggle jobs continue to use the existing production execution path.
