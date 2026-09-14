# Kaggle Recovery Contract

Read this before any retry, successor launch, cancellation, or interrupted-chat continuation.

## Canonical incident snapshot

Freeze these fields before mutation:

- repository and remote URL;
- branch/ref, commit SHA, and dirty-tree state;
- campaign/candidate and predecessor/successor relationship;
- phase, lane/slot, account ID, canonical owner, and kernel reference;
- workflow run/job IDs and event/request IDs;
- expected and observed object hashes;
- latest provider status/log receipt;
- stable ChatGPT/Codex conversation identity and project/worktree binding;
- already-completed side effects;
- current phase, locked-test, fan-in, release, and acceptance fences.

Missing required identity is a blocker, not permission to reconstruct it from names.

## Mutation decision table

| Confirmed class | Mutation allowed after reconciliation | Recovery shape |
|---|---|---|
| Source/test contract drift | Minimal superseding candidate only | Patch the narrow predicate/contract, run positive and negative tests, regenerate exact bytes, validate, then launch once |
| Exact object identity mismatch | Builder/workflow fix forward only | Preserve the immutable candidate; correct serialization/transport and create a new exact candidate if provider mutation already occurred |
| OIDC/admission policy mismatch | Policy change only with independently proven authority | Prefer an already-allowlisted permanent read path; never widen trust merely to make a retired readiness job green |
| Credential/route mismatch | Scoped route or authorized credential repair | Prove the exact account and canonical owner; avoid touching healthy siblings |
| Event/runner/network infrastructure | No product mutation unless later evidence proves it necessary | Reconcile durable state, retry only safe read operations, and restore the execution channel |
| Chat stream interruption | No blind replay | Continue the same stable chat after deduplicating prior effects; replace the chat only after bounded delivery failure |

## Retry and deduplication rules

- Read-only query: retry with a new request ID only after proving the earlier request has no receipt/run or ended without a provider mutation.
- Compute/write request: do not retry from UI uncertainty. Resolve the provider object/session and GitHub action first.
- Same logical incident: reuse a durable incident key and record every attempt.
- Duplicate launch: select one canonical candidate from exact lineage and identity; quarantine the other. Do not combine metrics or feed it into selection/fan-in.
- Cancellation: require the exact provider session ID and an authorized, documented cancellation contract. A kernel slug alone is not enough.

## Validation after a source fix

Collect applicable gates without allowing a focused pass to hide an unrun requirement:

1. focused classifier/unit behavior, including negative decoys;
2. deterministic build on the target runner OS;
3. Git blob/tree/candidate hash comparison;
4. workflow payload validation before provider mutation;
5. provider acceptance receipt and exact kernel ref;
6. read-only live status/log confirming observable resumed progress;
7. aggregate/system/security/migration/release gates required by the project;
8. predecessor immutability and successor lineage;
9. duplicate quarantine and unchanged phase/locked-test fences.

## Secret-free incident record

Record:

- classification and parent failure family;
- subreason and smallest affected scope;
- exact source, run, receipt, and provider references;
- whether compute/write/deploy occurred;
- attempted retries and deduplication key;
- fix or reconciliation performed;
- validation performed and required gates not run;
- unblock condition and next action;
- canonical/duplicate disposition;
- authority fences.

Never include Kaggle tokens, Cloudflare credentials, OIDC bearer tokens, or unredacted logs that may contain them.

