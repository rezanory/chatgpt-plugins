# Contributing / Architecture Review

V0.1 is intentionally reviewable by other AI systems and humans. Provider additions should:

1. live in `plugins/<provider>`;
2. reuse provider-neutral contracts where possible;
3. document new secret boundaries;
4. include deterministic tests;
5. not weaken repository/profile allowlists;
6. add a short ADR for architecture changes that affect more than one provider.
