# ADR-002 — Kaggle secrets remain in GitHub execution scopes

## Status

Accepted for V0.1.

## Decision

The repository stores account metadata only. `secret_scope` names a trusted GitHub Environment
that contains the Kaggle credential. The planner validates the account ID against repository-owned
configuration before a credential-bearing matrix job starts.

ChatGPT, GitHub Issues, source repositories, and Kaggle job code never need the raw provider token.
