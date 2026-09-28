# SoilBin V29.2 — Relative Reward/Penalty

## Historical audit
V18 already used a relative advantage:
(ref weighted expert loss - expert loss) / scale.
That compared experts on the same observed result and updated weights only for later passes.

V18 did NOT explicitly test:
- rank-based reward,
- pairwise win/loss reward,
- historical-percentile reward,
- result-to-previous-result trend reward,
- rank+trend,
- pairwise+historical combinations.

## New arms
- V23_BASE
- STATIC_INV_MAE
- TRAIN_PAIRWISE_PRIOR
- V18_REFERENCE
- RELATIVE_RANK
- RELATIVE_PAIRWISE
- RELATIVE_HIST_PERCENTILE
- RELATIVE_TREND
- RELATIVE_RANK_TREND
- RELATIVE_PAIRWISE_HIST

## Leakage guard
At pass T:
1. weights are frozen before observing the T target;
2. prediction for T is emitted;
3. only then is the actual T result observed;
4. reward/penalty may change weights for T+1 or later.

Outer-test results never tune eta or initial weights.
Eta and initial priors come only from outer-training OOF predictions.
