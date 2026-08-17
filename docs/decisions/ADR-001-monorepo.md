# ADR-001: Monorepo with provider components

**Status:** accepted for V0.1

Use one `chatgpt-plugins` repository. Shared contracts live under `packages/`; each provider lives
under `plugins/<provider>`. Provider packages must not import one another. This reduces duplicated
scheduler/security code while preserving independent deployment.
