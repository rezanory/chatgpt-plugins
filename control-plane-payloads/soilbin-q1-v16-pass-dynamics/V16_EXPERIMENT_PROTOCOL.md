# SoilBin V16 — Functional Pass-Dynamics Challenge

## Status
Post-lock exploratory challenge. Frozen V5 remains unchanged.

## Scientific question
Does repeated-pass response follow a single monotone law, or is a more flexible / regime-changing law required?

No functional family is privileged a priori.

## Frozen data contract
- 51 valid runs
- 9 Speed×Load groups
- 41 observed consecutive transitions
- source SHA256: dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059
- missing design slots remain V1W1T1, V2W3T2, V3W2T1
- no random split
- no target modification
- no external label used for selection

## Primary state variable
For transition t-1 -> t:

D_joint(t) = (abs(Delta LC1) + abs(Delta LC5)) / 2

Channel-specific D_LC1 and D_LC5 are sensitivity endpoints.

## Primary prediction contract
ONE_STEP:
Given the observed transition magnitude immediately before the target transition, predict the next transition magnitude.

This is intentionally different from V5/V11 force prediction. V16 MAE values must not be numerically interpreted as direct improvements over V5/V11 force MAE.

## Secondary contract
ANCHOR:
Given the first observed consecutive-transition magnitude in a Speed×Load group, forecast later transition magnitudes.

## Families
- Persistence
- Linear
- Logarithmic
- Power
- Free exponential
- Monotone exponential decay
- Hyperbolic decay
- Saturating exponential decay
- Stretched exponential decay
- Quadratic log-link
- Piecewise log-link
- Constrained decay -> rebound

## Validation
- Every performance estimate leaves one full Speed×Load group out.
- Nested family selection uses only the outer-training groups.
- Breakpoint is a discrete model parameter selected only from training data.
- Candidate breakpoints: 3, 4, 5.
- Current transition target is never used to predict itself.

## T6 transition hypothesis
T6 is tested as a candidate regime-transition / shakedown-break signal using:
1. paired D6-D5 within all 9 groups,
2. bootstrap CI,
3. exact sign-flip inference,
4. directional exact binomial test,
5. leave-one-group-out breakpoint stability,
6. comparison of monotone-decay families with non-monotone / rebound families.

T6 must not be called confirmed collapse, fatigue failure, permanent-strain failure, or a confirmed shakedown class without additional physical evidence.

## Local acceptance result
The local execution is an implementation/acceptance run, not the final operational receipt.

Observed before Kaggle confirmation:
- best ONE_STEP D_joint fixed family: QUADRATIC_LOG
- nested family selection chose QUADRATIC_LOG in 9/9 outer folds
- DECAY_REBOUND breakpoint: 5 in 9/9 outer folds
- PIECEWISE_LOG breakpoint: 5 in 9/9 outer folds
- T6 D6>D5: 8/9 groups
- T6 D6>2×D5: 7/9 groups

Final reportable V16 evidence requires the Kaggle operational run to complete and its JSON artifacts to pass the workflow gates.
