# SoilBin Q1/Q2 V5 — Pre-registered Final Selection Policy

Frozen before V3/V4 results are inspected.

## Evidence boundary
- Model/representation selection uses SoilBin internal grouped out-of-fold evidence only.
- Independent external labels (Canada/ERDC) and literature corroboration never select or tune a model.
- Response remains calibrated vertical force proxy (N), not verified soil stress (kPa), until effective sensor area/direct stress calibration exists.
- LC5→5 cm and LC1→15 cm mapping remains provisional until physically confirmed.

## Eligible candidate requirements
A candidate is eligible only if all are true:
1. Complete OOF prediction for all 41 consecutive history pairs.
2. Outer validation leaves one V×W sequence out; no random split.
3. All hyperparameter/architecture choices are made inside outer-training groups.
4. No current-pass waveform/raw trace/target-derived current feature is used as predictor.
5. Seed, source fingerprint, code fingerprint, and environment manifest are present.
6. No external-validation label is used in training, model selection, feature selection, thresholding, or uncertainty calibration.

## Primary ranking
1. Lowest **joint group-equal MAE**, defined as the mean of LC1 and LC5 group-equal MAE.
2. Report target-specific LC1/LC5 MAE, RMSE, bias and R² as guardrails, not as hidden alternative primary endpoints.
3. Compare candidates paired by the same 9 V×W groups using exact sign-flip inference and bootstrap CI; inference is descriptive because n_groups=9.

## Complexity / tie rule
- If two eligible candidates differ by **<2% relative joint group-MAE**, choose the simpler representation/model family.
- A deep/raw-waveform model may replace a simpler morphology/tabular model only when it improves joint group-MAE by **≥5%** OR its exact paired group evidence is materially stronger, and it does not worsen either LC1 or LC5 group-MAE by more than 5% relative.
- If deep learning is unstable across outer folds or seeds, prefer the best classical representation.

## Robustness gates
The selected candidate must be reported with:
- QC-OK sensitivity.
- leave-one-speed-out and leave-one-load-out results when applicable.
- group-calibrated 80% and 95% interval coverage where available.
- comparison with Persistence and TrainMedian baselines.
- comparison with V1/V2 historical champions.

## External validation rule
V3 independent numeric external evidence is used only after internal selection is frozen.
- Canada 2026: independent stress-depth/load/tire-pressure physics/trend validation.
- ERDC/CRREL 2009: independent repeated-pass / force-pressure physics validation with device limitations retained.
- Keller/Arvidsson/Bahrami: mechanistic/literature corroboration.
- Urmia-domain studies: near-domain replication only.

No absolute N↔kPa prediction claim is allowed until sensor effective area/direct stress calibration and depth mapping are confirmed.

## Final V5 lock
V5 will freeze:
- champion representation/model decision,
- all scientific claims and limitations,
- all source/data/code hashes,
- final manuscript tables/figures,
- reproducibility manifest,
- external-validation evidence ledger,
- deployment-fit model (if useful) trained on all internal development data only after performance claims are frozen.
