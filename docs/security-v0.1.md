# Security Model — V0.1

## Trust boundaries

1. **ChatGPT / user conversation** — may request work but never receives Kaggle credentials.
2. **GitHub control repository** — contains source contracts/configuration, Issues, workflows, and
   sanitized evidence metadata.
3. **GitHub Environment** — credential boundary for exactly one authorized Kaggle account.
4. **GitHub runner** — short-lived relay that may see the account credential during submission or
   status collection.
5. **Kaggle kernel** — untrusted execution of target repository code; it does not receive the
   Kaggle API token used by the relay.
6. **Kaggle outputs/logs** — untrusted data that must be sanitized before presentation to ChatGPT.

## Secret rules

- Kaggle tokens are GitHub Environment secrets only.
- Account registry stores only `account_id`, public Kaggle owner slug, capabilities, and the trusted
  GitHub Environment name.
- Job Issues may reference `account_id`; they cannot name a secret directly.
- The planner resolves `account_id -> environment` from repository-owned configuration before a
  credential-bearing job starts.
- Source packages exclude `.env`, key/certificate formats, and common generated dependency trees.
- No provider credential is embedded in the Kaggle bootstrap.

## Untrusted input rules

Job JSON is parsed with a strict allow-list of fields. Source repository names, commit IDs,
account IDs, profile names, task IDs, accelerator IDs, and parameter keys are validated.

Issue-provided values never become shell fragments. Execution profiles use argv arrays and the
Kaggle bootstrap uses `subprocess.run(..., shell=False)`.

Parameters are serialized to `job-parameters.json`; profile code may choose to read that file.
They are not concatenated into executable commands by the bridge.

## Log/artifact handling

Everything returned by Kaggle is untrusted.

Before a text excerpt is returned to the control Issue it is:

- stripped of ANSI control sequences;
- NUL-stripped;
- scanned for known Kaggle/token/authorization patterns;
- redacted where a secret-like value is detected;
- line-length bounded;
- total-size bounded.

Terminal evidence is stored under a fresh temporary directory. The bridge requests only selected
`result.json` / `job.log` output. Every stored evidence file receives SHA-256 metadata in
`manifest.json` before upload to GitHub Actions artifacts.

Never `eval`, source, execute, or interpolate a Kaggle output filename or log line.

## Source packaging

- symlinks are skipped;
- paths containing traversal components are skipped;
- per-file, total-size, and file-count limits are enforced;
- common secret and dependency-cache paths are excluded;
- the source commit is verified again with `git rev-parse HEAD` before submission.

## Repair

The workflow itself has no source-write step.

Failure evidence may cause ChatGPT to propose a patch, but source repair uses the normal GitHub
change path and must produce a **new commit**. Historical run evidence is never mutated.

V0.1 repair safeguards:

- default hard budget: 2 attempts;
- deterministic failure fingerprint;
- same fingerprint after a repair => stop/escalate;
- authentication/quota/policy errors => no source modification;
- no secret/account registry edits by repair automation;
- no push to default branch from Kaggle workflow.

## Known V0.1 limitation

GitHub Issues are not a transactional lease database. A single batch guarantees unique account IDs
by contract, but two independently created Issues can theoretically race for the same account.
The intended V0.1 operator flow is to use one batch Issue for parallel work and check active Issue
records before submitting another batch. Transactional cross-Issue leases belong in V0.2.
