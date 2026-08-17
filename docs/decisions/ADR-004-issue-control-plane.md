# ADR-004 — GitHub Issue control plane for V0.1

## Status

Accepted for V0.1.

## Context

The target ChatGPT environment can write Issues/comments through the existing GitHub integration,
but does not expose a comparable arbitrary new `workflow_dispatch` action. GitHub Actions can
trigger on Issue creation and Issue comments.

## Decision

Use a machine-readable GitHub Issue as the V0.1 job request/control record.

- Issue open -> validate and submit.
- `/kaggle status` comment -> query status and collect terminal evidence.
- high-churn state is not committed to Git.
- terminal evidence is an Actions artifact.

## Consequences

This preserves a direct normal-ChatGPT UX today. The Issue transport is explicitly temporary and
must not leak into the Kaggle provider contract. A future DirectMCPTransport can replace it.
