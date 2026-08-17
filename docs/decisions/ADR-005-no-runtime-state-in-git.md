# ADR-005 — Do not use Git commits as runtime state

## Status

Accepted.

## Decision

Do not create commits for RUNNING/FAILED/SUCCEEDED transitions, leases, logs, or result payloads.
Source/configuration remains versioned in Git. Runtime control uses Issues/comments and Actions
artifacts in V0.1.

## Rationale

This avoids parallel push races, merge conflicts, repository history noise, and coupling the
control plane to source history.
