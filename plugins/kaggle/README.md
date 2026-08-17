# Kaggle provider — V0.1

Kaggle is the first provider component in `chatgpt-plugins`.

V0.1 is operated from a normal ChatGPT conversation through the existing GitHub integration:

```text
ChatGPT -> [KAGGLE-JOB] GitHub Issue -> Actions dispatcher -> Kaggle
ChatGPT -> /kaggle status comment -> Actions collector -> Kaggle -> sanitized evidence
```

## Provider responsibilities

- validate trusted account/profile mappings;
- package an immutable GitHub source commit while excluding common secrets;
- create a private Kaggle source dataset;
- create/push a private Kaggle kernel;
- use argv-only execution profiles (`shell=False`);
- query kernel status and selected output;
- sanitize returned text;
- normalize failures and generate deterministic fingerprints;
- hash terminal evidence before GitHub artifact upload.

## Multi-account V0.1

One Issue may contain up to 16 tasks. Each task must name a distinct authorized `account_id`.
GitHub Actions fans the tasks out as a matrix and each task loads only its account's GitHub
Environment.

This reliably supports the primary parallel-use case **inside one batch**. V0.1 does not claim a
transactional lease across two independent Issues created at the same instant.

## Source transfer

The relay packages already-authorized GitHub source into a temporary private Kaggle Dataset and
runs a private Kernel that references it. This avoids putting source-repository credentials inside
the Kaggle Kernel.

## Not in V0.1

- direct write-capable MCP transport;
- cross-Issue transactional lease service;
- autonomous repair/commit loop;
- provider router;
- automatic long-term artifact storage/cleanup.
