# SoilBin V6.1 Experiment Protocol

## Purpose

V6.1 is a post-lock exploratory extension of the completed V6 analysis. It tests whether deterministic virtual channels derived from the previous-pass LC1/LC5 state, and training-only synthetic augmentation, improve prediction on real unseen Speed × Load groups.

V6.1 does not overwrite or supersede the frozen V5 champion.

## Frozen reference

- Frozen champion: `V3_D_DELTA_PREV_PEAK_FROZEN`
- Frozen joint group-equal MAE: `24.28878365273039 N`
- Valid runs: `51`
- Speed × Load groups: `9`
- Consecutive previous-pass transitions: `41`
- Missing design slots: `V1W1T1`, `V2W3T2`, `V3W2T1`

## Scientific status of virtual channels

The V6.1 virtual channels are deterministic mathematical transformations of the observed previous-pass LC1 and LC5 values. They are engineered features, not additional measured load cells and not evidence that LC7–LC10 were observed.

The registered channels are:

1. common mean;
2. LC5−LC1 differential;
3. normalized imbalance;
4. LC5/LC1 ratio;
5. logarithmic ratio;
6. vector norm;
7. geometric mean;
8. force-vector angle.

No value from the current target pass is used to construct a virtual channel.

## Synthetic augmentation

Synthetic samples are generated after every training split, never before splitting. The permitted methods are:

- conservative jitter on previous-pass LC1/LC5;
- local same-pass nearest-neighbour mixup using only rows available in the current training partition;
- a registered hybrid of the two methods.

Synthetic-to-real ratios tested are 0.5, 1.0 and 2.0. No synthetic row is permitted in inner validation or outer test data.

Every outer-fit synthetic row records its real source row IDs. Source IDs are checked against held-out row IDs and execution fails closed on any intersection.

## Registered experiment matrix

- `BASELINE_REAL_ONLY`
- `VS_CORE_REAL_ONLY`
- `VS_FULL_REAL_ONLY`
- `BASE_JITTER_050`
- `VS_FULL_JITTER_050`
- `BASE_LOCAL_MIXUP_050`
- `VS_FULL_LOCAL_MIXUP_050`
- `VS_FULL_LOCAL_MIXUP_100`
- `VS_FULL_LOCAL_MIXUP_200`
- `VS_FULL_HYBRID_100`

The pre-registered flagship ablation is `VS_FULL_LOCAL_MIXUP_100`.

The primary deployable V6.1 estimate is `V61_NESTED_SELECTED`, in which both the experiment arm and tree model are selected using only the inner grouped folds of each outer split.

## Validation contract

- Outer validation: leave one complete Speed × Load group out.
- Inner validation: `GroupKFold` on the remaining real groups.
- Model family: the frozen V3/V5 Random Forest and Extra Trees candidate catalog.
- Random row splitting: prohibited.
- External labels for selection: prohibited.
- Current-pass waveform or current target as a predictor: prohibited.
- Test-set-informed augmentation or feature fitting: prohibited.

The real-only baseline must reproduce the frozen V5 result within 2%; otherwise execution terminates without interpreting V6.1 challengers.

## Metrics and inference

Primary metric:

- joint group-equal MAE across LC1 and LC5.

Secondary metrics:

- LC1 and LC5 group-equal MAE;
- ordinary MAE, RMSE, bias and R²;
- number of groups improved;
- paired mean improvement by group;
- exact one-sided sign-flip test across the nine held-out groups;
- group bootstrap 95% interval;
- augmentation-seed stability.

## Exploratory acceptance rule

A challenger is flagged as meeting the registered exploratory acceptance rule only when all of the following hold:

1. joint group-equal MAE is lower than the real-only runtime baseline;
2. at least 6 of 9 held-out groups improve;
3. neither LC1 nor LC5 group-MAE worsens by more than 5%;
4. the exact one-sided sign-flip p-value is at most 0.05.

This flag is not authority to replace V5. Any replacement requires independent confirmation and a separately approved selection policy.

## Stability contract

The flagship arm is repeated with augmentation seeds:

- `20260914`
- `20260928`
- `20261005`

The runner reports whether improvement is positive under all three seeds.

## Mandatory evidence

The run must emit:

- `RESULTS_V6_1.json`
- `RUN_MANIFEST_V6_1.json`
- `LEAKAGE_AUDIT_V6_1.json`
- real-only OOF prediction tables;
- nested selection and search tables;
- group-paired comparison tables;
- augmentation source provenance for outer fits;
- seed-stability tables;
- figures and a final Markdown report.

The workflow accepts the run only if the V5 reproduction gate and leakage audit both pass.
