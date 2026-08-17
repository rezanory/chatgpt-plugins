# Architecture Freeze — V0.1

## Decision

V0.1 uses the existing ChatGPT ↔ GitHub integration as the only write-capable ChatGPT control
surface. Kaggle remains the compute provider. The runtime control record is a GitHub Issue;
runtime status is represented by machine-readable Issue comments and terminal evidence is stored
as Actions artifacts.

```text
ChatGPT
  -> GitHub Issue control plane
  -> GitHub Actions thin relay
  -> Kaggle provider
  -> Kaggle compute
```

No Git commit is created merely to represent QUEUED/RUNNING/FAILED state.

## Why an Issue control plane

The current ChatGPT GitHub integration available to this project can create/read/update Issues and
Issue comments, while a direct "start arbitrary new workflow" action is not exposed in the same
way. GitHub Actions can natively trigger on `issues: opened` and `issue_comment: created`.
Therefore an Issue gives us a direct user-visible control surface today without depending on Work
or Codex execution.

## Protocol

Job Issues:

- title prefix: `[KAGGLE-JOB]`
- body marker: `<!-- chatgpt-plugins-job:v1 -->`
- JSON schema identifier: `chatgpt.compute.job/v1`
- source must be an immutable 40–64 hex commit id
- V0.1 batch: 1–16 tasks
- each task must use a distinct configured account ID

Submission produces a machine-readable run comment marked:

`<!-- chatgpt-plugins-run:v1 -->`

Status collection produces:

`<!-- chatgpt-plugins-status:v1 -->`

## Parallelism model

A single Issue can request multiple tasks. The planner validates every account against the trusted
repository-owned account registry and maps each task to its preconfigured GitHub Environment.
The workflow then uses a matrix to submit the tasks in parallel.

V0.1 intentionally requires unique account IDs inside a batch. This gives deterministic parallel
multi-account behavior without pretending GitHub concurrency groups are a durable distributed
lease system.

Cross-Issue atomic leasing is deferred. ChatGPT should inspect active Issue records before creating
a new batch. A V0.2 control-plane store may add transactional leases and quota/health state.

## Transport separation

Provider semantics are separate from transport semantics:

```text
ComputeProvider
  + TransportAdapter
```

V0.1 transport:

```text
KaggleProvider + GitHubIssueTransport
```

Future:

```text
KaggleProvider + DirectMCPTransport
```

The provider must never require GitHub Issue paths, labels, secret names, or workflow concepts.

## Execution security

Job Issues select a profile and parameters. They do not contain commands.

Profiles are repository-owned configuration with steps represented as argv arrays. The Kaggle
bootstrap executes each step with `shell=False`. Parameters are passed as a JSON file/environment
reference, never interpolated into a command string.

Source packaging excludes common secret files, symlinks, caches, virtual environments, node
modules, and oversized inputs.

## Monitoring

Dispatcher workflow:

1. validate Issue/job contract;
2. validate account/profile mapping;
3. checkout immutable source commit;
4. load exactly one account environment;
5. create private Kaggle source dataset;
6. push private Kaggle kernel;
7. write run record to the Issue;
8. exit.

It does **not** poll for hours.

Status workflow starts only when `/kaggle status` is posted. It queries non-terminal tasks,
downloads selected evidence only on terminal state, sanitizes it, hashes each file, uploads an
Actions artifact, and posts a normalized status record.

## Failure/repair boundary

Failure classification is deterministic and returns category, confidence, retryability, and a
normalized fingerprint. ChatGPT may reason on top of that report.

V0.1 does not let the Actions workflow modify source code. A repair must become a new source
commit (preferably on a repair branch/PR) and a new job. Historical run evidence is immutable.

Hard repair defaults:

- max attempts: 2
- same failure fingerprint after repair: stop/escalate
- auth/quota/policy errors: never source-repair
- no secret/config registry edits by repair logic

## Deferred

- transactional cross-Issue leases
- Redis/Postgres control-plane state
- provider router
- predictive quotas
- web dashboard
- arbitrary notebook generation
- unattended agent loop
- public Plugin Directory packaging
