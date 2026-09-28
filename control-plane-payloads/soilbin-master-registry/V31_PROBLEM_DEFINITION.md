# SoilBin V31 — Problem Definition Before Implementation

## Why V31 exists
V1–V30 show that the project is no longer best described as a generic two-output regression problem.

The empirical evidence separates the system into at least three mechanisms:

1. **Timing transfer**
   - highly structured;
   - strongly related to speed;
   - leave-one-Speed×Load-group-out delay model R²≈0.963.

2. **Waveform-shape transfer**
   - weak/misleading at zero lag;
   - extremely strong after alignment;
   - median aligned cross-sensor waveform R²≈0.972.

3. **Amplitude/state transfer**
   - still difficult;
   - simple ratios, residual experts, probabilistic amplitude models and generic state augmentation have not solved LC5.

Therefore V31 should attack mechanism 3 while explicitly controlling mechanisms 1–2.

## New formal problem

Given a held-out Speed×Load condition and only information legitimately available before a target prediction:

- infer an allowed timing offset representation;
- transform the observed source-sensor waveform into an aligned shape representation;
- predict the target sensor/pass amplitude/state;
- preserve sensor asymmetry;
- preserve pass/regime heterogeneity.

The model must not use the target waveform, target peak, target timing or any future target-derived quantity to align or construct features for its own scored prediction.

## Main V31 hypothesis

A **Lag-Aligned Cross-Sensor Expert** can reduce LC5 error because V4 failed on unaligned waveform representations, while V30 demonstrates that aligned LC1/LC5 waveform shape is strongly coupled.

## Predeclared decomposition

### Module A — timing
Use only leakage-safe known/predicted timing information.
Candidate inputs:
- Speed
- Load
- Pass
- training-only learned inverse-speed delay law
- prior-pass observed delay when available

Target-pass true LC5 timing is prohibited as an inference feature.

### Module B — aligned source waveform shape
From the legal source waveform:
- shift using Module A's predicted delay;
- extract low-capacity shape descriptors;
- do not feed huge raw traces directly by default.

Candidate descriptors:
- aligned normalized waveform basis coefficients;
- active-window width/skewness;
- rise/fall asymmetry;
- normalized impulse/RMS morphology;
- compact PCA/SVD basis learned strictly inside training groups.

### Module C — amplitude/state expert
Predict target peak/delta or residual relative to V29/V23.
Separate:
- LC1 expert
- LC5 expert
and allow pass-specific heads.

### Module D — routing
Start with fixed routing learned from V28/V29 evidence.
Do not begin with a learned sample-specific meta-router because V29.1 failed with 9 groups.

## Mandatory ablations

1. V23 reference.
2. V29_FIXED_ROUTING reference.
3. no-waveform / no-alignment.
4. unaligned waveform representation equivalent in spirit to V4.
5. predicted-delay alignment only.
6. predicted-delay + compact aligned shape.
7. predicted-delay + aligned shape + amplitude residual.
8. LC5-only aligned expert.
9. LC1-only aligned expert.
10. pass-specific aligned expert.
11. T2-only aligned expert.
12. T6-only aligned expert.
13. T2+T6 aligned expert.

## Primary endpoints
- Joint Group-equal MAE.
- LC1 Group-equal MAE.
- LC5 Group-equal MAE.

## Secondary endpoints
- per-pass LC1/LC5 MAE;
- group-paired improvement;
- exact sign-flip;
- group bootstrap CI;
- helped/harmed group fraction;
- timing MAE;
- aligned-shape reconstruction/correlation diagnostics.

## Failure criteria
V31 should be considered scientifically negative if:
- aligned shape adds no gain beyond V29 routing;
- gains come only from using target timing/target waveform information;
- improvement collapses under full Speed×Load group holdout;
- LC5 improves only by materially damaging LC1;
- improvement exists only after post-hoc per-test rerouting.

## Interpretation
Even a strong V31 result on the current 9 groups remains repeated-development evidence.
Independent experimental groups or truly compatible external force data are still required for statistical closure.
