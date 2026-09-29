# SoilBin V31–V50 Parallel Scientific Campaign

## Purpose

Execute every distinct idea retained from the post-V30 reframing as a separate, auditable experiment while preserving the V31+ preflight rules.

## Immutable validation rules

- Predictive experiments use whole Speed×Load outer holdout.
- Hyperparameter or representation selection is restricted to outer-training groups.
- No random-row split.
- No target-pass peak, target waveform, target timing, or any future target-derived field may be used as a predictor for its own scored outcome.
- History features must be available before the target pass.
- Descriptive geometry versions may analyze the observed response directly, but they are explicitly marked DESCRIPTIVE and cannot be promoted to predictive evidence.
- All reported improvements remain repeated-development evidence on the same nine Speed×Load groups unless genuinely independent data are later introduced.
- Existing V23 and V29 locked references are preserved; no result is silently re-baselined.
- No Kaggle kernel already running for another project may be cancelled or overwritten.

## Canonical references

- V23 locked joint Group-MAE: 23.104189837450967 N.
- V29 fixed-routing joint Group-MAE: 21.657327708415846 N.
- Best observed V29 T2+T6 development variant: 20.970434406316773 N.
- Canonical 51-run feature table fingerprint is checked by the inherited loader.
- V30 established strong lag-aligned cross-sensor shape coupling; timing/shape and amplitude/state must remain analytically separated.

## Parallel execution model

The campaign is deliberately lane-independent at runtime. When a later scientific idea conceptually depends on latent state discovered elsewhere, that lane re-learns its state representation *inside each outer-training fold* instead of importing outer-test-derived state from another V. This allows safe parallel execution without cross-lane leakage.

Accounts are capacity-audited before submission. The controller uses the existing private CPU Kaggle control-plane route, reuses a matching kernel if it already exists, never cancels an existing kernel, and queues excess lanes until an account is free.

Reserved accounts for unrelated active programs are excluded unless separately cleared.

## Completion contract

A V is COMPLETE only when:
Input -> Runtime -> Execution -> Output -> Acceptance evidence exists.

Each Kaggle lane must emit RESULTS.json containing:
- schema/version
- scientific_status
- data fingerprint
- primary metrics or descriptive findings
- outer-group evidence where applicable
- leakage guards
- acceptance status

A campaign-level synthesis is allowed only after all twenty lane outputs have been collected or explicitly classified as FAILED/BLOCKED with evidence.
