# PNEUMONIA V6.2.2 — Kaggle-native finalization architecture

## Decision

GitHub is the control plane / bridge / journal only. It is not a model-compute or model-artifact execution environment for V6.2.2 finalization.

The Kaggle master account remains `azadka` and owns finalization execution. Completed worker kernels remain the durable source of their outputs. Large selected checkpoints do not need to be copied into GitHub in order to authorize finalization.

## Responsibility split

### Kaggle

- durable custody of completed model/checkpoint outputs;
- selected-backbone checkpoint/config validation and hashing;
- FINAL_FREEZE_MANIFEST assembly;
- frozen ensemble inference;
- official locked-test evaluation;
- external validation;
- final metrics, predictions, manifests, and evidence bundles.

### GitHub

- versioned bridge/control code;
- governance policy and immutable identifiers/hashes;
- command/dispatch receipts;
- run status monitoring;
- compact manifests/hashes/metrics receipts after Kaggle finishes;
- issue journal and closure record.

GitHub Actions/self-hosted runners must not be used to perform model inference, ensemble scoring, locked-test evaluation, external validation, or as a required store for large checkpoints.

## Frozen selected ensemble

Source kernel: `trickermark/pneumonia-v6-2-2-backbone-m06-r224`

Selected members:

1. `M06__convnext_tiny`
2. `M06__densenet121`
3. `M06__resnet50v2`

Ensemble weights are frozen at `1/3, 1/3, 1/3` and the deployment policy is governed by the recovered `FROZEN_BACKBONE_ENSEMBLE_POLICY.json`.

Frozen recipe SHA-256:

`27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f`

Frozen policy SHA-256:

`7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861`

## Finalization gates

1. **Kaggle-native freeze**
   - master consumes the exact selected outputs from the completed backbone kernel;
   - validates three config hashes and hashes the three selected checkpoints in Kaggle;
   - writes `FINAL_FREEZE_MANIFEST.json` and a SHA-256 receipt;
   - no locked-test or external cohort access.

2. **Official locked test**
   - only after the freeze manifest is complete and immutable;
   - uses the frozen ensemble/deployment policy;
   - threshold, calibration, weights, architecture, and selected members may not change;
   - results cannot feed back into training/HPO/selection.

3. **External validation**
   - only after official locked-test execution is complete;
   - evaluation-only; no adaptation or retuning;
   - records dataset identity, metrics, predictions, reliability/statistics, and provenance.

4. **Closure**
   - GitHub receives only compact receipts/manifests/hashes/metrics needed for audit and project closure.

## Prohibited work

- no M01-M12 retraining;
- no HPO rerun;
- no confirmation rerun;
- no selection using locked-test or external data;
- no GitHub-hosted model inference/freeze/external evaluation;
- no remapping of master from `azadka`.
