# SoilBin V29 — Final Synthesis

## V29 fixed routing
- Frozen fixed-routing Joint Group-MAE: 21.657328 N
- LC1: 12.659261 N
- LC5: 30.655394 N
- Improvement vs V23: 1.446862 N (6.262%)
- Best observed V29 arm: V29_T2_T6_UNCONDITIONAL at 20.970434 N.
- This is repeated development evidence on the same historical groups, not independent validation.

## V29.1 — all four requested meta-learning modules
1. Error Predictor: Joint 27.761621 N
2. Winner Classifier: Joint 29.888402 N
3. Generalization-Gap Predictor: Joint 27.793878 N
4. Meta-Router: Joint 27.603148 N

None beats V23 at 23.104190 N.
The descriptive three-expert oracle is 14.922985 N, showing large selection headroom but poor present reliability identification.

## V29.2 — relative reward/penalty
Historical V18 already used expert loss relative to the current weighted reference, but did not explicitly test rank, pairwise, historical-percentile, or trend-based relative functions.

- V18 reference Joint: 23.821530 N
- Best new relative mode: RELATIVE_PAIRWISE_HIST
- Best new relative Joint: 23.919537 N
- Gain vs V18 reference: -0.098007 N

## Scientific interpretation
The strongest evidence continues to favor pass/sensor-specific architecture rather than a universal router or universal reward law. Reliability learning has substantial theoretical headroom, but 9 Speed×Load groups are too small for the current sample-specific meta-router to generalize reliably.

No outer-test label was used to train its own prediction, no random row split was used, and sequential reward updates use an observed pass only for later passes.
