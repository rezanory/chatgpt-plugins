# SoilBin V1–V30 Master Version Registry

**Purpose:** canonical preflight reference for V31 and every later SoilBin experiment.  
**Rule:** before proposing or executing a new experiment, check this registry and the Do-Not-Repeat matrix.

## Evidence labels

- **ARTIFACT-VERIFIED** — result/evidence is present in the current repository artifacts.
- **CHECKPOINT-VERIFIED** — result was verified in prior project execution/checkpoints but the full final artifact is not currently stored beside this registry.
- **PROTOCOL-VERIFIED** — protocol/design is present; do not treat a planned arm as a successful result without execution evidence.
- **LOCK-ONLY** — scientific lock/documentation version; no new model fitting by design.
- **NEGATIVE** — executed challenger did not improve the relevant baseline.
- **EXPLORATORY** — same historical groups; useful development evidence, not independent validation.

---

# 0. Frozen project facts

- **51 valid experimental runs**.
- **9 Speed×Load groups** = 3 speed levels × 3 load levels.
- Passes **T1–T6**.
- **41 valid adjacent previous-pass transitions**.
- Missing logical design slots:
  - `V1W1T1`
  - `V2W3T2`
  - `V3W2T1`
- Real measured force channels: **LC1** and **LC5** only.
- Working depth mapping used in later analysis:
  - LC1 ≈ 15 cm
  - LC5 ≈ 5 cm
- Historical V5 claims lock retained this mapping as provisional pending physical-layout confirmation.
- Canonical data fingerprint:
  `dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059`
- Canonical waveform source:
  `timeseries_long_lc1_lc5.csv`
  - 80,958 rows
  - SHA256 `ea227f412827a962a1876ceb9aeb9946034b435052e9e74115e32bd7dca73cb0`
- Scientific evaluation rule:
  - hold out **whole Speed×Load groups**;
  - **no random-row split**;
  - no current/future target leakage;
  - no target modification;
  - external datasets with incompatible units/endpoints are corroboration only.

---

# 1. V1 — First predictive baseline

**Evidence:** CHECKPOINT-VERIFIED + source runner present.

**Question:** Can next-pass LC1/LC5 force be predicted at all, and does previous-pass state help?

**Main work**
- Initial model-family benchmark.
- Random Forest, Extra Trees and simpler baselines.
- Persistence and median baselines.
- Direct vs delta-style targets.
- Grouped validation.
- History/delta ablations and uncertainty/importance analysis.

**Reported result**
- Historical `C_delta / RandomForest`:
  - Group-equal MAE ≈ **23.0877046 N**.

**Critical metric guard**
- V1 used **82 sensor-depth history observations**, not the later 41-pair dual-target joint metric.
- Therefore the 23.09 N number is **historical context only** and must not be ranked directly against V2+ joint results.

**What V1 taught us**
- Previous-pass state is strongly informative.
- Delta-style prediction is useful.
- Group-aware validation is necessary.

**Do not repeat**
- Generic “does history help?” baseline work.

---

# 2. V2 — Engineered previous-pass waveform morphology

**Evidence:** ARTIFACT-VERIFIED via V5 comparison + V2 runner.

**Question:** Do engineered waveform-shape features from the previous pass improve next-pass force prediction?

**Main work**
- Previous-pass LC1/LC5 waveform morphology.
- Coupled dual-channel delta target.
- Nested grouped validation.

**Result**
- `D_delta_waveform_coupled / Selected_inner`
- Joint Group-MAE = **26.7358963 N**.

**Interpretation**
- Engineered morphology was not enough.
- Later V3 compact previous-peak state was better.

**Do not repeat**
- Same V2 engineered-morphology feature set unchanged.

---

# 3. V3 — Compact previous-peak champion

**Evidence:** ARTIFACT-VERIFIED.

**Question:** Can a compact state representation beat the more elaborate waveform representation?

**Main work**
- Current load/speed/pass.
- Previous LC1 and LC5 peak-force proxies.
- Delta target.
- Nested leave-one-Speed×Load-sequence-out selection.
- External physics/trend corroboration.

**Result**
- `D_delta_prev_peak / Selected_inner`
- Joint Group-MAE = **24.2887837 N**
- LC1 Group-MAE = **12.785420 N**
- LC5 Group-MAE = **35.792147 N**
- LC1 R² ≈ **0.533**
- LC5 R² ≈ **0.616**

**Important negative result**
- Adding engineered previous-pass waveform morphology to the compact model worsened:
  - 24.2888 → **26.8058 N**.

**External evidence**
- 70 numeric external rows from Canada/Ayetan + ERDC/CRREL.
- Supports physical/trend consistency only.
- No N↔kPa predictive-validation claim.

**What V3 taught us**
- Compact prior-state peaks outperform generic waveform complexity.
- LC5 is the dominant error source.

---

# 4. V4 — Raw/spectral waveform challenge

**Evidence:** ARTIFACT-VERIFIED.

**Question:** Was V2 failure caused by poor engineered waveform features; can raw/spectral previous-pass waveform representation beat V3?

**Main work**
- Raw waveform representations.
- Spectral representation.
- Same grouped validation and target.

**Best result**
- `SPECTRAL` = **25.0684622 N**
- ≈ **3.21% worse** than V3.
- Paired one-sided sign-flip p ≈ **0.8164**.

**Conclusion**
- Unaligned raw/spectral waveform representation did not beat compact previous peaks.

**Important later reinterpretation**
- V30 showed waveform shape becomes extremely informative **after inter-sensor time alignment**.
- Therefore V31 lag-aligned waveform work is **not a repetition of V4**.

---

# 5. V5 — Final scientific/manuscript lock

**Evidence:** ARTIFACT-VERIFIED.  
**Status:** LOCK-ONLY.

**Purpose**
- Freeze the selected model, claims, limitations, external-evidence ledger, uncertainty, tables and reproducibility.
- **No new scientific fitting by design.**

**Locked model**
- V3 `D_delta_prev_peak`.

**Locked result**
- Joint = **24.2887837 N**
- LC1 = **12.785420 N**
- LC5 = **35.792147 N**

**Other benchmark**
- Persistence ≈ **35.09 N**.
- V3/V5 improves persistence by ≈ **30.8%**.

**Claims guards introduced**
- Do not call force proxy verified soil stress.
- Do not convert N to kPa without effective sensor area/calibration.
- Do not treat 102 channel observations as 102 independent experiments.
- Do not claim universal cross-soil generalization.

---

# 6. V6 — 23-item exploratory structural extension

**Evidence:** CHECKPOINT-VERIFIED + full V6.23 source runner present.  
**Status:** EXPLORATORY.

**Purpose**
- Exhaust a broad family of “obvious next ideas” before deeper specialization.

## V6 work package

### 6.1 Calibration/conditions audit
- Conversion consistency.
- Sample timing/sample count.
- No invented sensor area or depth reassignment.

### 6.2 Cumulative multichannel history
Compared:
- conditions only;
- lag-1 both channels;
- cumulative multi-channel history.

### 6.3 Prior-art gap matrix
Questions:
- Does cumulative pass history help beyond previous state?
- Is there a stable pass×channel graph?
- Can experimental burden be reduced by information-optimal design?

### 6.4 Repeated-pass analyses
- trajectory-family fits;
- memory depth;
- leave-one-pass-out;
- all 15 forward pass-pair relations;
- raw vs summarized history;
- shared backbone vs pass-specific heads;
- recursive forecasting;
- missing-pass reconstruction;
- LC1/LC5 ablation/fusion;
- history-shuffle negative control;
- missing-history robustness;
- multi-horizon/early-history forecasting.

### 6.5 Graph/spectral/design analyses
- pass graph;
- 12-node T×LC graph;
- Laplacian spectrum;
- effective resistance where interpretable;
- D-optimal experimental ordering;
- spectral/Laplacian regularization;
- spectral vs current numeric encoding;
- TSP experimental logistics order.

**Representative reported values**
- Conditions only ≈ **44.83 N**.
- Prior LC1+LC5 history ≈ **30.68 N**.
- 3-pass history ≈ **30.38 N**.
- Spectral Ridge ≈ **29.89 N**.
- None displaced V5=24.29 N.

**Conclusion**
- Most generic “more history / more graph features / more spectral structure” ideas were already tested.
- V6 is a major anti-duplication reference.

---

# 6.1 V6.1 — Virtual channels + synthetic augmentation

**Evidence:** PROTOCOL-VERIFIED + CHECKPOINT-VERIFIED.  
**Status:** NEGATIVE / EXPLORATORY.

**Question**
Can deterministic virtual channels and training-only synthetic augmentation improve real held-out groups?

**Virtual channels**
1. LC1/LC5 common mean
2. LC5−LC1 differential
3. normalized imbalance
4. LC5/LC1 ratio
5. log ratio
6. vector norm
7. geometric mean
8. force-vector angle

**Synthetic training arms**
- conservative jitter;
- local same-pass nearest-neighbor mixup;
- hybrid;
- synthetic:real ratios 0.5, 1, 2.

**Leakage contract**
- synthetic rows created only after split;
- never in validation/test;
- source IDs checked against held-out rows.

**Reported results**
- V5 reference = **24.2888 N**
- VS Core ≈ **24.7841 N**
- nested selection ≈ **25.2926 N**
- flagship mixup ≈ **27.37 N**

**Conclusion**
- No challenger accepted.
- Synthetic/virtual channels did not add independent information.

---

# 7. V7 — Multi-memory + fusion + spectral integration

**Evidence:** CHECKPOINT-VERIFIED + runner present.  
**Status:** NEGATIVE / EXPLORATORY.

**Tasks explicitly in runner**
- V5 baseline
- lag-2
- lag-3
- summarized history
- explicit LC fusion
- spectral replace pass
- spectral augment pass
- full lag3+summary+fusion+spectral

**Reported results**
- V5 reproduced.
- LC fusion challenger ≈ **24.367 N**.
- Full memory+fusion+spectral ≈ **24.870 N**.

**Conclusion**
- Generic feature pile-up did not beat the compact V5 state.

---

# 8. V8 — State/error modeling and hard-group diagnostics

**Evidence:** CHECKPOINT-VERIFIED + runner present.  
**Status:** NEGATIVE / DIAGNOSTIC.

**Question**
Can explicit state descriptors correct the V5 error and explain hard groups?

**Main work**
- EWMA state.
- multiscale/path/state features.
- compact/full state variants.
- cross-fitted residual Ridge correction.
- hard-group error atlas.
- training-state-distance diagnostics.

**Reported result**
- Best EWMA state ≈ **24.772 N**, worse than V5.

**Hard groups identified**
- V3W1
- V1W1
- V3W2

**Conclusion**
- State descriptors are useful diagnostically.
- Generic state augmentation alone is insufficient.
- LC5 dominates hard-group error.

---

# 9. V9 — Ensemble / Mixture-of-Experts

**Evidence:** CHECKPOINT-VERIFIED + runner present.  
**Status:** MIXED / EXPLORATORY.

**Expert bank**
- V5
- lag/history representations
- fusion
- spectral
- EWMA/multiscale/path/compact/full-state
- persistence

**Methods**
- equal all-expert average;
- fixed Ridge stacking;
- V5+candidate pairwise convex blends;
- channel-specific selection;
- regime selection;
- channel×regime gates;
- greedy blend/MoE-like selection.

**Best reported signal**
- simple **V5 + STATE_PATH** blend ≈ **24.0492 N**.

**Conclusion**
- Small targeted blending headroom exists.
- Complex gates/MoE are too unstable for current sample size.

---

# 10. V10 — Signal/measurement audit

**Evidence:** CHECKPOINT-VERIFIED + campaign protocol.

**Purpose**
- Check whether signal/data defects explain modeling error.

**Audit scope**
- 102 channel traces.
- 80,958 time-series samples.
- nonfinite checks.
- timestamp checks.
- force/peak reproduction.

**Result**
- 0 nonfinite failures.
- 0 timestamp errors.
- 0 N reproduction error.

**Conclusion**
- Computational signal extraction is internally reproducible.
- Unknown hardware calibration, sensor geometry and DAQ details cannot be reconstructed from data.

---

# 11. V11 — Loss/weight/capacity ablation

**Evidence:** CHECKPOINT-VERIFIED + campaign protocol.  
**Status:** POSITIVE.

**Question**
Can better regularization/loss weighting improve the same force endpoint without adding feature complexity?

**Methods**
- 8 controlled model variants.
- shallow RF.
- group-equal weighting.
- quantile/median-type alternatives.

**Champion**
- `RF_SHALLOW_WEIGHTED`
- Joint = **23.8604408 N**
- LC1 ≈ **12.9017 N**
- LC5 ≈ **34.8192 N**
- ≈1.76% improvement vs V5.
- paired evidence p≈0.0918.

**Conclusion**
- Shallow regularization + group weighting is useful.
- Not statistically closed.

---

# 12. V12 — TabPFN v2

**Evidence:** CHECKPOINT-VERIFIED + protocol.  
**Status:** NEGATIVE.

**Methods**
- TabPFN v2.
- Direct target.
- Delta target.
- CPU, no external-data API.

**Reported**
- Delta ≈ **25.4226 N**
- Direct ≈ **26.613 N**

**Conclusion**
- Delta remains preferable.
- TabPFN did not beat tree baseline.

---

# 13. V13 — Probabilistic low-rank latent state

**Evidence:** CHECKPOINT-VERIFIED + protocol.  
**Status:** NEGATIVE.

**Question**
Can low-rank latent-state filtering explain repeated-pass dynamics?

**Methods**
- rank-1/rank-2 probabilistic latent-state filters;
- forward-only;
- log-state variant.

**Best reported**
- rank-2 log formulation ≈ **29.1082 N**.

**Conclusion**
- Poor fit for small heterogeneous dataset.

---

# 14. V14 — Physics relaxation prior + GP discrepancy

**Evidence:** CHECKPOINT-VERIFIED + protocol.  
**Status:** NEGATIVE / PHYSICS-BLOCKED.

**Methods**
- relaxation-law priors;
- power/exponential physics proxy;
- Gaussian-process discrepancy.

**Best reported**
- power-style proxy ≈ **25.4975 N**.

**Guard**
- validated multifidelity claim **blocked** pending physical metadata.
- synthetic/analytic proxy was never treated as measured data.

**Conclusion**
- Physics prior is informative but not competitive/validated enough.

---

# 15. V15 — Broad heterogeneous ensemble

**Evidence:** CHECKPOINT-VERIFIED + protocol.  
**Status:** NEGATIVE.

**Designated members**
- V5
- QRF median weighted
- TabPFN delta
- state rank-1 log
- physics-exp/GP
- persistence

**Method**
- fixed equal and anchored convex ensemble;
- strict meta-crossfit;
- V5 minimum weight 0.5;
- sparsity penalty.

**Best reported**
- ≈ **25.3329 N**.

**Conclusion**
- “ensemble everything” did not help.

---

# 16. V16 — Functional pass-dynamics challenge

**Evidence:** ARTIFACT/PROTOCOL-VERIFIED + CHECKPOINT result.  
**Endpoint warning:** **different endpoint from force prediction**.

**Scientific question**
Does repeated-pass transition magnitude follow one monotone law or require regime change?

**State variable**
`D_joint(t) = (|ΔLC1| + |ΔLC5|)/2`

**Families**
- persistence
- linear/log/power
- free and monotone exponential
- hyperbolic
- saturating/stretched exponential
- quadratic log-link
- piecewise log-link
- decay→rebound

**Result**
- best ONE_STEP fixed family = **QUADRATIC_LOG**
- Group-MAE ≈ **17.3137 N**
- selected in **9/9 outer folds**
- piecewise/rebound breakpoint = **5 in 9/9**
- T6 D6>D5 in **8/9 groups**
- T6 D6>2×D5 in **7/9 groups**

**Conclusion**
- T5→T6 is a robust candidate regime transition.
- Do not call it collapse/fatigue/failure without deformation evidence.

---

# 17E. V17E — Dynamics ensemble/gating

**Evidence:** CHECKPOINT-VERIFIED + protocol.  
**Status:** NEGATIVE.

**Experts**
- QLOG
- piecewise decay→rebound
- saturating exponential
- persistence

**Arms**
- QLOG
- equal blend
- leakage-safe convex blend
- pass gate
- state gate

**Result**
- QLOG-only ≈ **17.3137 N** remained best.

**Conclusion**
- dynamics ensemble/gating did not improve QLOG.

---

# 17T. V17T — Prospective T6 transition prediction

**Evidence:** CHECKPOINT-VERIFIED + protocol.  
**Status:** POSITIVE DIAGNOSTIC.

**Question**
Can pre-target history predict the T6/rebound-type transition?

**Result**
- state logistic AUROC ≈ **0.83594**
- balanced accuracy ≈ **0.78125**
- Brier ≈ **0.168852**
- breakpoint 5 stable **9/9**

**Conclusion**
- occurrence/state transition is learnable.
- physical failure class remains unproven.

---

# 18. V18 — Reward/penalty factorial

**Evidence:** CHECKPOINT-VERIFIED + code/protocol.  
**Status:** MINOR / NEGATIVE AS PRIMARY LEVER.

**Arms**
- no update
- reward only
- penalty only
- reward+penalty

**Historical mechanism**
Expert loss was compared to the current weighted reference loss; updates only affected later passes.

**Reported**
- reward-only ≈ **18.0181 N** on dynamics endpoint.

**Conclusion**
- adaptive reward has tiny effect only.

---

# 19. V19 — Learning-intensity law

**Evidence:** CHECKPOINT-VERIFIED + protocol.

**Laws**
- constant
- pass increasing/decreasing
- history-ratio dependent
- state-magnitude dependent
- hybrid

**Best reported**
- STATE_MAG ≈ **17.9830 N**.

**Conclusion**
- tiny/local gain.

---

# 20. V20 — Reward update functional form

**Evidence:** CHECKPOINT-VERIFIED + protocol.

**Forms**
- exponential
- power
- hyperbolic
- bounded tanh
- clipped linear

**Best reported**
- POWER ≈ **18.0218 N**.

**Conclusion**
- update-law choice is not a major lever.

---

# 21. V21 — Transition memory policy

**Evidence:** CHECKPOINT-VERIFIED + protocol.

**Policies**
- keep memory
- soften after rebound
- reset multiplier
- T6-specific training-only prior

**Best reported**
- T6_PRIOR ≈ **17.9609 N**.

**Conclusion**
- small effect only.

---

# 22. V22 — Shared vs channel-specific adaptive policy

**Evidence:** CHECKPOINT-VERIFIED + protocol.

**Policies**
- shared static
- channel-specific static
- shared dynamic
- channel-specific dynamic
- hierarchical shared+channel dynamic

**Reported**
- shared static ≈ **22.88263 N**
- LC1 ≈ **12.8430 N**
- LC5 ≈ **32.9222 N**

**Interpretation guard**
- V17–V25 protocol explicitly designates V23 as the first full state-program force-level integration comparable to V11.
- Do not promote V22 alone as the main force champion.

---

# 23. V23 — State/regime integration on original force endpoint

**Evidence:** ARTIFACT/CHECKPOINT-VERIFIED.  
**Status:** POSITIVE.

**Question**
Can useful pass-dynamics information be transferred back to the original force endpoint?

**Arms**
- exact V11 baseline
- state-feature augmented RF
- magnitude scaling
- augmented RF + scaling

**Champion**
- `STATE_FEATURE_RF`
- Joint = **23.1041898 N**
- LC1 = **13.009658 N**
- LC5 = **33.198722 N**

**vs V11**
- gain ≈ **0.756251 N**
- ≈ **3.17%**
- 5/9 groups improved
- sign-flip p≈ **0.1582**
- bootstrap improvement CI ≈ **[-0.4748, +2.0056]**

**Conclusion**
- QLOG/state-derived features genuinely help force prediction.
- uncertainty still crosses zero.

---

# 24. V24 — Component ablation

**Evidence:** CHECKPOINT-VERIFIED + protocol/code.

**Removed one at a time**
- functional ensemble/gating
- reward/penalty
- adaptive intensity
- non-default update function
- transition policy

**Result**
- removing those adaptive pieces changed final V23 force MAE essentially **0**.

**Conclusion**
- V23 gain came mostly from **QLOG-derived state features**, not the adaptive machinery.
- Major anti-duplication result.

---

# 25. V25 — Robustness + external corroboration

**Evidence:** CHECKPOINT-VERIFIED.

**Internal**
- frozen candidate group bootstrap.

**Result**
- model-MAE bootstrap 95% CI ≈ **[19.6471, 26.2962] N**.

**External**
- ERDC/CRREL repeated-pass pressure sequences.
- Canada/Ayetan directional stress evidence.

**Conclusion**
- useful dynamics/physics corroboration.
- still no same-endpoint independent external force validation.

---

# 26. V26 — LC5 transition-aware residual challenge

**Evidence:** ARTIFACT/PROTOCOL-VERIFIED.  
**Status:** NEGATIVE.

**Question**
Can dominant LC5 error be reduced with LC5-specific residual experts while LC1 is frozen to V23?

**Arms**
1. V23
2. LC5 specialist RF
3. global residual
4. pass-regime residual (T2 / T3–T5 / T6)
5. state-gated residual
6. hybrid transition residual

**LC5 results**
- V23 = **33.1987**
- LC5 specialist ≈ **36.2965**
- global residual ≈ **34.5415**
- pass-regime ≈ **36.8244**
- state-gated ≈ **34.5800**
- hybrid ≈ **33.8234**

**Conclusion**
- all challengers worse.
- simple LC5 residual specialization rejected.

---

# 27. V27 — Exhaustive bidirectional + multi-history map

**Evidence:** ARTIFACT-VERIFIED.  
**Status:** MAJOR STRUCTURAL MAP.

## Layer A — directed pair map
- 12 nodes = T1..T6 × LC1/LC5
- **132/132 directed edges**
- 66 unordered pairs
- source improved over conditions-only in **70/132**
- mean gain ≈ **+2.628 N**
- reverse easier is **not universal**
- among 60 temporal unordered pairs, reverse lower raw MAE in 28/60.

**Information-hub finding**
LC5 is hard as a target but highly informative as a source.

## Layer B — joint forward/reverse/cycle
- joint better than independent: 27/66
- cycle better than joint: 33/66
- benefits highly selective.

## Layer C — global directed head
- global mean MAE ≈ **58.63**
- edge-specific mean ≈ **47.98**
- global better only 20/132.

**Conclusion**
- reject monolithic universal head.

## Layer D — 1116 multi-history tasks
- Forecast = 342
- Reconstruction = 342
- Smoothing = 432
- median MAE:
  - forecast ≈37.65
  - reconstruction ≈37.41
  - smoothing ≈28.02
- LC1 target median ≈23.65
- LC5 target median ≈53.85
- sensor fusion helped only **71/372** triplets.
- typical useful contiguous memory depth ≈ **2**.
- full history best only 3/8 sensor×target cases.

**Notable exception**
LC5 T6 forecast from T2–T5:
- LC1-only 67.89
- LC5-only 52.30
- both 36.83

## Probabilistic V27
- Two-stage ≈25.5887
- quantile ≈26.2821
- mixture ≈24.2141
- selective ≈24.6325
- all worse than V23.

Event classifier:
- AUROC≈0.75
- Brier≈0.2086
- nominal 50% interval coverage observed≈31.7%
- mean interval width≈49.12 N

**V27 conclusion**
- heterogeneity/selectivity is the central structure.
- do not repeat exhaustive pair enumeration.

---

# 28. V28 — Hierarchical + legal reverse-informed stacking

**Evidence:** ARTIFACT-VERIFIED.  
**Status:** POSITIVE BUT MIXED.

## V28 predeclared hierarchical family
`HIER_SELECTIVE`
- Joint = **22.7394053**
- LC1 = **12.659261**
- LC5 = **32.819549**
- gain vs V23 ≈ **0.3648 N / 1.579%**
- 5/9 groups improved
- sign-flip p≈0.3359
- bootstrap improvement CI ≈[-1.183,+1.908]

## Simple reverse/averaging arms
- reverse-only ≈50.527
- global convex ≈23.681
- selective simple blend ≈24.054
- passwise ≈24.382

**Conclusion**
- simple forward+reverse averaging does not help.

## Edge-level legal stacked reverse
- 60 true forward edges.
- STACKED improved **34/60**.
- mean gain ≈ **+1.930 N**.
- LC5-target mean gain ≈ **+3.884 N**.
- LC5 T2/T6 mean gain ≈ **+7.620 N**.
- TARGET_T2 particularly strong.
- T6 highly heterogeneous.

## V28.1 post-hoc force stack
- LC5 T2 stack:
  - Joint ≈ **22.3158**
  - LC5 ≈ **31.6220**
  - T2 LC5 MAE 45.691 → **33.866**
- LC5 T2+T6:
  - Joint ≈ **22.105659**
  - LC5 ≈ **31.201661**
  - most gain from T2; T6 small endpoint gain.

**Conclusion**
- targeted pass/sensor stacking is useful.
- generic blending is not.

---

# 29. V29 — Fixed routing + reliability meta-learning + relative reward

**Evidence:** ARTIFACT-VERIFIED.  
**Status:** POSITIVE FIXED ROUTING / NEGATIVE META-ROUTER / NEGATIVE REWARD.

## V29 fixed routing
- V23 = 23.104190
- V28 Hier = 22.739405
- **V29_FIXED_ROUTING = 21.657328**
  - LC1 = 12.659261
  - LC5 = 30.655394
  - +6.262% vs V23
- no-reverse fixed = 21.968965
- T2-only reverse = **21.180581**
- T2+T6 reverse observed = **20.970434**
  - LC1 = 12.659261
  - LC5 = 29.281608
  - +9.235% vs V23

**Scientific status**
Repeated-development evidence on the same 9 historical groups; not independent confirmation.

## V29.1 Generalization Reliability Meta-Learning

Executed all four requested modules:
1. Error Predictor → **27.7616**
2. Winner Classifier → **29.8884**
3. Generalization-Gap Router → **27.7939**
4. Meta-Router → **27.6031**

All worse than V23.

Meta quality:
- LC1 error-prediction MAE≈13.98 N
- LC1 winner accuracy≈29.4%
- LC5 error-prediction MAE≈24.84 N
- LC5 winner accuracy≈28.3%

Three-expert **descriptive oracle**:
- Joint ≈ **14.9230 N**
- LC1 ≈10.137
- LC5 ≈19.709

**Conclusion**
- expert-selection headroom is enormous;
- current 9 groups are insufficient to learn reliable sample-specific routing.

## V29.2 Relative Reward/Penalty
Tested:
- rank
- pairwise
- historical percentile
- trend
- rank+trend
- pairwise+historical

Best new relative mode:
- `RELATIVE_PAIRWISE_HIST` ≈ **23.9195 N**
- worse than V18 reference and V23.

**Conclusion**
- reward/penalty adaptation is not a major lever.

---

# 30. V30 — LC1↔LC5 coupling study

**Evidence:** ARTIFACT-VERIFIED.  
**Status:** MAJOR STRUCTURAL DISCOVERY.

**Goal**
Study LC1 relative to LC5 directly, rather than treating them only as independent targets.

## Peak/ratio work
Executed:
- audited all 12 same-pass LC1↔LC5 directed edges from V27;
- all 72 cross-sensor directed edges;
- LC5/LC1, inverse ratio, difference, relative difference, log-ratio;
- Pass, Speed, Load, Speed×Load and Speed×Load×Pass stratification;
- impulse ratio;
- RMS ratio;
- duration ratio;
- peak timing delay;
- adjacent-transition elasticity;
- group-held-out ratio models.

**Peak findings**
- median LC5/LC1 = **1.7623**
- mean = **1.7672**
- CV = **35.4%**
- range = **0.526–3.249**
- LC5>LC1 in **88.2%** of runs
- load explains more log-ratio variation than pass alone
- peak-ratio ↔ impulse-ratio Pearson ≈ **0.946**

**Same-pass direction**
LC1 helped predict LC5 in **5/6 passes**, but direction reverses/weakens in T6.

**T5→T6**
- LC1 mean change ≈ **−8.1%**
- LC5 mean change ≈ **−19.6%**
- mean ratio change ≈ **−0.219**

## Waveform-level work
Canonical source:
- 80,958 rows.

**Crucial finding**
- zero-lag comparison is misleading due to timing offset.
- median best shifted cross-correlation ≈ **0.9861**
- median lag-aligned waveform R² ≈ **0.9723**
- cross-correlation lag vs direct peak delay r≈ **0.9952**

Median peak delay:
- V1 ≈ **2337 ms**
- V2 ≈ **1200 ms**
- V3 ≈ **737 ms**

Delay structure:
- corr(delay,1/speed)≈ **0.9898**
- leave-one-Speed×Load-group-out inverse-speed delay model:
  - R²≈ **0.9626**
  - MAE≈ **95.2 ms**

Amplitude:
- aligned scale median≈1.535
- aligned scale vs peak-ratio r≈ **0.9395**
- amplitude remains much harder to generalize than timing/shape.

**Interpretation guard**
Delay is an **observed sensor/channel response offset**, not proven soil-propagation delay until sensor geometry and synchronization are verified.

**V30 conclusion**
Timing transfer and waveform shape are highly structured.
**Amplitude/state transfer is now the central unresolved bottleneck.**

---

# Global synthesis after V1–V30

1. **History/state dependence is the strongest repeatable signal.**
2. More model capacity/features usually do not solve the problem.
3. LC1 and LC5 are **asymmetric tasks**.
4. LC5 is harder as target but often informative as source.
5. Passes are not exchangeable.
6. T2, T3–T5, and T6 repeatedly behave differently.
7. T5→T6 is supported by multiple independent analyses as a candidate regime change, but not a confirmed failure mechanism.
8. Fixed pass×sensor routing currently beats monolithic models and learned routers.
9. Reverse information helps **selectively**, especially T2.
10. Generic reverse averaging fails.
11. Generic fusion fails more often than it helps.
12. Full-history models are not automatically better; useful memory is often short/sparse.
13. Reward/penalty adaptation is not a major lever.
14. Generic probabilistic occurrence/amplitude separation did not beat V23.
15. Reliability meta-learning is **data-limited**, not conceptually disproven; oracle≈14.923 N shows large headroom.
16. V4 and V30 are not contradictory:
    - V4 tested unaligned raw/spectral previous-pass waveform representation;
    - V30 discovered very strong **lag-aligned cross-sensor** waveform similarity.
17. The dominant unresolved problem is **amplitude/state generalization under unseen Speed×Load groups**.

---

# Do-Not-Repeat Matrix for V31+

Do not rerun unchanged:

- conditions-only regression;
- V2 engineered waveform-morphology feature set;
- V4 unaligned raw/spectral waveform challenge;
- deterministic virtual channels + jitter/mixup from V6.1;
- generic “add more lags/summaries/fusion/spectral features” from V6/V7;
- generic EWMA/multiscale/path state augmentation from V8;
- broad ensemble-everything / complex MoE from V9/V15;
- same TabPFN direct/delta setup from V12;
- same rank-1/rank-2 latent-state formulation from V13;
- physics/multifidelity proxy without new physical metadata from V14;
- QLOG ensemble/gating from V17E;
- reward/penalty/intensity/update-law tuning from V18–V21 and V29.2;
- simple LC5 RF specialist/residual correction from V26;
- exhaustive forward/reverse edge enumeration from V27;
- universal/global directed head from V27;
- always-on sensor fusion from V27;
- simple reverse averaging/global convex blending from V28;
- current-feature-set sample-specific meta-router from V29.1 without new groups/information;
- fixed/simple LC5/LC1 ratio model from V30.

---

# Reframed scientific problem after V30

## Old framing
> Predict LC1 and LC5 force from Speed, Load and Pass using one best regressor.

## Better framing
> Learn a **partially observed, pass-dependent, two-sensor soil–tool state system** in which **timing transfer, waveform-shape transfer, and amplitude/state transfer are separate subproblems**, and where sensor/pass-specific experts are evaluated under whole-Speed×Load holdout.

## Empirically separated subproblems

### A. Timing transfer
- highly structured;
- inverse-speed relationship is strong;
- currently easiest part.

### B. Waveform-shape transfer
- very strong after alignment;
- median aligned R²≈0.972.

### C. Amplitude/state transfer
- still difficult;
- likely dominant source of LC5 error.

### D. Pass-regime transfer
- T2, T3–T5 and T6 differ.
- T5→T6 repeatedly looks special.

### E. Sensor asymmetry
- LC1→LC5 and LC5→LC1 are not interchangeable.

### F. Routing/uncertainty
- fixed routing currently works;
- learned sample-specific routing needs more independent information/groups.

---

# V31 design implication — Lag-Aligned Cross-Sensor Expert

V31 should **not** be “another waveform model.”

It should specifically test whether V30's alignment discovery can improve the amplitude/state bottleneck.

Required design:

1. Estimate/encode the sensor response offset using only permitted pre-target/geometry-safe information.
2. Align LC1 and LC5 source waveforms.
3. Extract lag-aligned shape-transfer features.
4. Separate **shape transfer** from **amplitude residual**.
5. Train separate LC1 and LC5 experts.
6. Use pass-specific heads/routing.
7. Include strict ablations:
   - V23;
   - V29 fixed routing;
   - no-alignment waveform;
   - alignment only;
   - alignment + shape;
   - alignment + amplitude residual;
   - alignment + pass-specific expert.
8. Keep whole Speed×Load group holdout.
9. Never use the target waveform/value to align or construct a feature for its own prediction.
10. Any gain on these same 9 groups remains repeated-development evidence until independently confirmed.

---

# Scientific priority after V30

**Highest-value next question:**
Can the highly predictable inter-sensor timing/shape relationship be converted into better **amplitude/state prediction** without target leakage?

That question is materially different from V2/V4 and is the recommended basis for V31.
