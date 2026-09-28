# SoilBin Experiment Preflight — Mandatory for V31+

Before any new SoilBin version is coded or run, answer all items below.

## 1. Proposed scientific question
State one falsifiable question.

## 2. Closest prior versions
List every overlapping prior version from V1–V30.

## 3. Exact novelty delta
For each closest prior version, state what is genuinely new:
- new observable?
- new temporal alignment?
- new endpoint?
- new causal/physical constraint?
- new validation population?
- new routing mechanism?
- new independent data?

“Different model name” or “more features” is not sufficient novelty.

## 4. Duplicate gate
Search the Master Registry and Do-Not-Repeat Matrix.
If the same hypothesis and information set were already tested and no new evidence/data/constraint exists, stop and label the proposal DUPLICATE.

## 5. Leakage contract
Declare exactly what is known at inference time.
Explicitly list forbidden current/future target-derived fields.

## 6. Validation contract
Default:
- whole Speed×Load outer holdout;
- training-only inner selection;
- no random-row split;
- no outer-test rerouting;
- no target modification.

## 7. Baselines
Every new predictive version must compare against all scientifically relevant locked references, normally:
- V23;
- V29_FIXED_ROUTING;
- best applicable negative control from the closest prior version.

## 8. Ablation plan
The new mechanism must be isolated with an ablation. If improvement cannot be attributed to the claimed mechanism, do not interpret it causally.

## 9. Scientific status
Classify before execution:
- PREDECLARED DEVELOPMENT
- REPEATED DEVELOPMENT
- INDEPENDENT CONFIRMATION
- DESCRIPTIVE ONLY
- DIAGNOSTIC ONLY

## 10. Completion evidence
No version is COMPLETE from unit tests alone.
Required evidence:
Input → Runtime → Execution → Output → Acceptance.

## Current V31 preflight
Closest prior versions:
- V2/V4: previous-pass waveform models, but without V30's cross-sensor lag alignment.
- V27: cross-sensor directionality and history map, but no waveform alignment transfer.
- V28/V29: targeted pass/sensor routing and reverse stack, but peak/state level.
- V30: discovered timing/shape coupling but did not convert aligned shape into a next-pass amplitude expert.

Novelty delta:
V31 uses **predicted/legal lag alignment as a transformation before compact cross-sensor shape-transfer features**, then tests whether these features improve target amplitude/state under whole-Speed×Load holdout.

Therefore V31 is **not a duplicate** of V4 or V30 if and only if target timing/waveform is not used to construct the scored prediction.
