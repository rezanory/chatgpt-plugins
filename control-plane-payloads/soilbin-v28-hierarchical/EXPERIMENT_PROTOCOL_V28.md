# SoilBin V28 — Hierarchical + Forward/Reverse Ensemble Challenge

## Frozen baseline
- Data SHA256: dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059
- 51 valid runs, 9 Speed×Load groups, 41 consecutive force transitions.
- V23 Joint Group-MAE: 23.104189837450967 N.
- V23 LC1 Group-MAE: 13.009658000484931 N.
- V23 LC5 Group-MAE: 33.198721674417 N.

## New question
Can information learned in the reverse direction be combined with the normal forward model without using the true future target at inference, and can a hierarchical pass/sensor/memory design improve V23?

## Forward/Reverse ensemble guards
The true future target is never an inference feature.
The reverse-only target candidate is obtained by:
1. fitting a reverse relation on outer-training data;
2. observing the known previous/source response at inference;
3. algebraically inverting the fitted linear reverse relation for the unknown target;
4. clipping only to outer-training target support.

Ensemble weights and selective thresholds are chosen only from outer-training OOF predictions.

## Force-endpoint reverse arms
1. V23_BASE
2. REVERSE_ONLY
3. GLOBAL_CONVEX
4. SELECTIVE
5. LC5_ONLY_CONVEX
6. LC5_T2T6
7. PASSWISE

## Hierarchical arms
1. V23_BASE
2. SAME_GLOBAL — same-sensor past-memory Ridge head.
3. CROSS_GLOBAL — both-sensor past-memory Ridge head.
4. SAME_PASS — pass-specific same-sensor memory head.
5. CROSS_PASS — pass-specific cross-sensor memory head.
6. SPARSE_FUSION — training-only pass/sensor gate chooses SAME vs CROSS.
7. HIER_CONVEX — V23 backbone + selected head with pass/sensor training-only convex weight.
8. HIER_SELECTIVE — use the hierarchical head only under training-learned disagreement threshold.
9. HIER_PROB_GATE — for LC5, transition probability may choose separate training-only weights when enough training support exists.

## Memory
Candidate past-only depths: 1, 2, 3, 4 passes.
Depth and Ridge alpha are selected entirely within training groups.
No future pass is used in a forecasting feature.

## Validation
- Whole Speed×Load group held out.
- No random row split.
- No target modification.
- No current/future target leakage.
- All reported improvements are post-lock exploratory development signals, not independent confirmation.
