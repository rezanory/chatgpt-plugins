# PNEUMONIA V6.2.2 — recovered run classification

## Decision

The recovered evidence is sufficient to classify the six final TRAIN shards as complete and
source-consistent. No TRAIN rerun is justified at this point.

The next safe phase is **read-only merge readiness**: identify the exact final model/metric artifacts
from W01-W06, normalize their manifests, and determine whether an existing merge/champion-selection
result already exists before enabling any new Kaggle compute.

## Account topology

Execution pool:

```text
kg-02  radlinaradlina
kg-03  rezanory
kg-04  reyhanehazad
kg-05  trickermark
kg-06  msdenis
kg-07  nisabulutmark
```

Separate Master:

```text
master  azadka  read-only
```

## Final TRAIN shard classification

| Shard | Owner | Kernel | Recovery state | Source fingerprint |
|---|---|---|---|---|
| W01 | azadka / Master | `pneumonia-v6-2-2-train-w01-m01-r224` | COMPLETE | VERIFIED |
| W02 | radlinaradlina | `pneumonia-v6-2-2-train-w02-m01-r320` | COMPLETE | VERIFIED |
| W03 | reyhanehazad | `pneumonia-v6-2-2-train-w03-m01-r384` | COMPLETE | VERIFIED |
| W04 | trickermark | `pneumonia-v6-2-2-train-w04-m02-r224` | COMPLETE | VERIFIED |
| W05 | msdenis | `pneumonia-v6-2-2-train-w05-m02-r320` | COMPLETE | VERIFIED |
| W06 | nisabulutmark | `pneumonia-v6-2-2-train-w06-m02-r384` | COMPLETE | VERIFIED |

Expected fingerprint recovered from the bounded end-of-run logs:

```text
fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838
```

The recovered logs also show terminal PASS-style completion evidence and matrix-report output paths.
The literal string `KAGGLE_EXECUTION_V62_2` was not present in the checked log tails and is therefore
not used as an acceptance gate.

## Upstream HPO classification

Existing V6.2.2 HPO kernels observed during inventory:

```text
radlinaradlina / pneumonia-v6-2-2-hpo-s01
reyhanehazad   / pneumonia-v6-2-2-hpo-s02
trickermark    / pneumonia-v6-2-2-hpo-s03
msdenis        / pneumonia-v6-2-2-hpo-s04
nisabulutmark  / pneumonia-v6-2-2-hpo-s05
```

Classification: **historical upstream evidence, not a current blocker**.

These HPO kernels were not individually re-polled for status during the final recovery pass. However,
the later final TRAIN shards are complete, share the expected source fingerprint, and contain a frozen
training recipe plus HPO-related source/contracts. Therefore there is no evidence-based reason to
rerun HPO before merge readiness is assessed.

## Confirmation classification

Existing confirmation kernels observed during inventory:

```text
radlinaradlina / pneumonia-v6-2-2-confirm-c01
trickermark    / pneumonia-v6-2-2-confirm-c02
msdenis        / pneumonia-v6-2-2-confirm-c03
```

Classification: **historical upstream evidence, not a current blocker**.

As with HPO, these confirmation runs were not individually re-polled in the final pass. Their presence
before the completed final TRAIN batch means they should be treated as consumed upstream work unless a
merge-readiness artifact explicitly shows a missing dependency.

## `kg-03 / rezanory`

The account is valid and authenticated, but its newest matching inventory entries belong to an older
`pneumonia-v62-deep-*` sequence rather than the W01-W06 final training shard set. It remains an execution
account but is not counted as one of the six recovered TRAIN shard owners.

## Merge / champion-selection state

The recovered evidence did **not** establish a canonical completed merge/champion-selection kernel.
This does not prove that no such run exists; the recovery report intentionally retained only bounded
inventory/output evidence.

Training outputs include source for ensemble/champion logic such as `auto_ensemble_selection.py`,
`backbone_ensemble_selection.py`, and `select_champion.py`. That makes merge/champion selection the
next logical phase, but it must begin with read-only artifact discovery rather than a fresh submission.

## Next operation

1. For W01-W06, enumerate final output filenames and identify model checkpoints, metrics, matrix
   reports, prediction files, and any champion/ensemble manifests.
2. Build a six-shard merge-readiness manifest containing exact kernel refs, artifact names, sizes and
   hashes where feasible.
3. Search all existing V6.2.2 kernels for a prior merge/champion-selection result and compare it to the
   six-shard manifest.
4. Only if no valid existing merge result exists should the write bridge be considered for a narrowly
   scoped rerun/submission decision.

Until that classification is complete:

```text
CGP_WRITE_ENABLED=0
NO NEW KAGGLE COMPUTE
```
