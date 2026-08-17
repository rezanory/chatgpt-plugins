# AI Review Packet — Implementation V0.1

## Implemented direction

The earlier architecture review converged on a narrower system:

```text
ChatGPT
  -> GitHub Issue control plane
  -> short GitHub Actions relay
  -> Kaggle
```

Status is on demand via `/kaggle status`; the relay does not wait for hours. Runtime state is not
committed to Git. Terminal evidence is sanitized, hashed, and uploaded as an Actions artifact.

## Key implementation decisions

- immutable source commit SHA;
- machine-readable Issue protocol;
- repository-owned account registry maps IDs to GitHub Environments;
- up to 16 parallel tasks per Issue, each on a unique authorized account;
- profile selection only, no arbitrary job commands;
- profile steps are argv arrays and execute with `shell=False`;
- private source Dataset + private Kernel;
- deterministic failure classifier + fingerprint;
- no Actions-side source write/repair;
- direct MCP transport deferred behind provider/transport contracts.

## Known deliberate limitation

Cross-Issue account leases are not transactional in V0.1. Parallel jobs should be grouped into one
batch Issue. A V0.2 state/lease service can be added after the end-to-end path is proven.

## Review focus

1. Can Issue-provided data still reach a shell or unsafe filesystem path?
2. Is dynamic GitHub Environment selection sufficiently constrained by the trusted account registry?
3. Is the source Dataset packaging safe enough for private source?
4. Are Kaggle status/output assumptions compatible with current official CLI behavior?
5. Are failure fingerprints stable enough to detect non-progressing repair attempts?
6. Are terminal artifact manifests sufficient for audit/provenance?
7. What is the smallest transactional lease store worth adding in V0.2?
8. Can DirectMCPTransport replace GitHubIssueTransport without provider rewrite?
