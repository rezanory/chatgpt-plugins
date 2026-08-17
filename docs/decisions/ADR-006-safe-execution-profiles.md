# ADR-006 — Execution profiles are argv-only

## Status

Accepted.

## Decision

A job may select a repository-owned execution profile but may not provide a command string.
Profiles contain argv arrays and optional relative working directories. The Kaggle bootstrap uses
`shell=False`.

## Consequences

Parameterization must happen through structured JSON/environment inputs or repository-owned code,
not shell interpolation. This intentionally reduces flexibility in exchange for a much smaller
command-injection surface.
