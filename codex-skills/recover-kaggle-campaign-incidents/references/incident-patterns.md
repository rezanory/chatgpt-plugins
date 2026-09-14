# Kaggle Incident Patterns

Use this reference to separate symptoms that look similar but require different repairs.

## Pattern matrix

| Symptom | Evidence required before classification | Correct response | Do not do |
|---|---|---|---|
| `PERSISTENCE_RESTORE_FAILED` plus KaggleHub `BackendError` | Parse the JSON after `POST failed with:` and require `wasSuccessful=false`, integer `error.code=5`, and `errors=["Not found"]` exactly | Classify contract drift if a fresh campaign should accept confirmed absence; patch only the absence classifier in a superseding candidate and test decoys | Treat every code 5, every “not found” string, permission failure, or malformed response as absence |
| Expected/actual notebook or manifest SHA differs | Compare bytes from the canonical Git object, builder output, workflow upload, and provider payload; inspect runner OS/newline behavior | Serialize deterministically and write bytes; rebuild and verify on the runner before launch | Normalize or rewrite an already-frozen candidate in place |
| HTTP 403 from Cloudflare/OIDC route | Compare token claims (`workflow_ref`, `event_name`, `ref`, actor/repository identity) with the deployed allowlist and its Git history | Use the permanent allowlisted route, or change trust only when the workflow truly has required authority and validation | Rotate Kaggle credentials or re-add retired temporary trust based on 403 alone |
| Account readiness reports the wrong owner | Compare `plugins/kaggle/config/accounts.json` with every active runtime map; remember a display ID such as `kg-01` may route as `master` | Probe the exact route and canonical owner with a read-only identity/quota call; update canonical generated metadata rather than stale ad hoc lists | Trust a dated readiness matrix or shift usernames between account IDs |
| Issue query exists but no run/receipt appears | Search workflow runs, issue receipts, and provider state by exact request ID | For a read-only request only, retry once with a new ID after proving the first produced no run/provider side effect | Count missing event delivery as credential failure or blindly replay writes |
| `Connection interrupted`, `Stopped thinking`, delivery timeout, or frozen UI | Resolve the stable conversation identity and compare its last completed action with GitHub/provider state | Continue the same chat from the unfinished point; deduplicate incident/retry keys | Start a new campaign or resend the last mutation before reconciliation |
| Two successors exist after recovery | Establish canonical source SHA, evidence ancestry, launch run, and provider refs for both | Quarantine the non-canonical candidate from selection, fan-in, and acceptance; record disposition | Merge both result sets or delete provider state without exact cancellation authority |

## KaggleHub fresh-absence predicate

The predicate is deliberately narrow. A compliant implementation should accept:

```text
BackendError: POST failed with: {"errors":["Not found"],"error":{"code":5},"wasSuccessful":false}
```

It must reject at least:

- `{"errors":["Permission denied"],"error":{"code":5},"wasSuccessful":false}`;
- `{"errors":["Not found"],"error":{"code":7},"wasSuccessful":false}`;
- malformed or non-JSON payloads;
- generic network/authentication errors;
- a matching JSON body carried by an unrelated exception type.

HTTP 404 remains a separate accepted proof of absence when the exception or its response exposes an integer `status_code` of 404.

## Exact-byte checklist

For any hash-bound notebook or payload:

1. Read the canonical input as bytes.
2. Parse only when modification is required.
3. Serialize with fixed encoding, separators/indentation, ordering policy, and exactly one chosen trailing newline.
4. Encode explicitly as UTF-8 and write with a byte API.
5. Hash the bytes that will actually be uploaded.
6. Rebuild and compare on the launch runner.
7. Refuse the provider mutation on any mismatch.

Text-mode writes on Windows can translate line endings and make a correct logical document fail exact object identity.

## OIDC/readiness checklist

Before attributing 401/403 to a Kaggle account:

1. Confirm the gateway health and configured-secret count without reading secret values.
2. Identify the exact deployed worker version/SHA.
3. Inspect the trusted workflow reference, event, repository/ref, and actor constraints.
4. Inspect the commit that added or removed the trust entry; removal after onboarding may be intentional cleanup.
5. Run one permanent-path, per-account, read-only identity or quota query.
6. Treat a query comment without a matching run/receipt as transport evidence missing, not account failure.
7. Require exact `N/N`; preserve individual receipts so a fail-fast aggregate cannot hide the failing account.
