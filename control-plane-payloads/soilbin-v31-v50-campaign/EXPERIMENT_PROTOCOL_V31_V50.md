# SoilBin V31–V50 Campaign Protocol

Campaign: `20260929-v31-v50-state-geometry-r1`

## Fixed scientific contract
- Canonical development data SHA-256: `dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059`.
- 51 valid runs, 9 Speed×Load groups, passes T1–T6 with three known missing logical slots.
- Predictive tasks use whole Speed×Load outer holdout.
- Any model/hyperparameter choice is training-group-only.
- No random-row split.
- No future target or target-pass waveform is permitted as an inference feature.
- Current results remain repeated-development evidence, not independent validation.

## Baseline references
- V23: 23.1041898375 N joint Group-MAE.
- V29_FIXED_ROUTING: 21.6573277084 N.
- Best observed V29 T2+T6 unconditional development arm: 20.9704344063 N.
- Historical descriptive oracle: ~14.923 N.

## Versions
| V | Question | Status class |
|---|---|---|
| V31 | Lag-aligned previous-pass waveform geometry for target amplitude/state | predictive |
| V32 | Reducible vs irreducible/error-floor decomposition | diagnostic |
| V33 | Low-dimensional latent predictive soil state | predictive/state discovery |
| V34 | Markov/state sufficiency: does older history add information? | predictive/diagnostic |
| V35 | Minimum sufficient memory depth | predictive |
| V36 | Cross-depth observability: LC1 vs LC5 vs both histories | predictive |
| V37 | Conditional directional predictive information LC1↔LC5 | predictive; not causal |
| V38 | Path dependence/hysteresis descriptors | predictive/diagnostic |
| V39 | Predictability profile over pass and sensor | diagnostic |
| V40 | Pass-blind soil-state/soil-age reconstruction | diagnostic prediction |
| V41 | Empirical invariant search | diagnostic |
| V42 | Empirical normalized-collapse search | diagnostic |
| V43 | Model-based state transition scenarios | scenario analysis; not causal |
| V44 | Controlled operator/Koopman-style lifted dynamics | predictive |
| V45 | Sparse interpretable state-transition law | predictive/equation discovery |
| V46 | Conventional vs full-history vs native latent representation | predictive |
| V47 | 3D response-surface geometry over Load×Speed×Response | descriptive geometry |
| V48 | 3D state-trajectory/phase-space geometry T1→T6 | descriptive geometry |
| V49 | 3D waveform geometry | descriptive geometry |
| V50 | Predictive ablation of geometry features | predictive |

## Interpretation guards
- V37 measures conditional predictive information, not causality.
- V38 loop/phase area is geometric; it is not declared mechanical energy without physical derivation.
- V41/V42 use ordinal Speed/Load levels; any invariant/collapse is empirical, not a physical dimensionless law.
- V43 is model-based scenario analysis, not an identified causal counterfactual.
- V45 standardized coefficients are not a constitutive law until physical variables/units and independent confirmation exist.
- V30 sensor delay remains an observed channel offset until geometry/synchronization are independently verified.

## Execution waves
Wave 1: V32–V48. These require only the canonical 51-run feature table and can execute immediately in parallel.

Wave 2: V31, V49, V50. These additionally require the frozen waveform-native feature package. V31/V50 use only **previous-pass** waveform-derived information for scored target predictions; target-pass waveform geometry is prohibited.

## Operational policy
- Private Kaggle notebooks.
- CPU only.
- Internet disabled.
- Never cancel or overwrite unrelated existing runs.
- kg-05 and kg-10 remain reserved.
- kg-01 remains master/control.
- Use the remaining safe accounts with dynamic inventory checks.
- A version is COMPLETE only with Input → Runtime → Execution → Output → Acceptance evidence.
