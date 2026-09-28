# SoilBin V26 — LC5 Transition-Aware Residual Challenge

## Decision question
Can the dominant LC5 error be reduced by modeling LC5 as a distinct, regime-sensitive target while keeping LC1 frozen to the V23 prediction?

## Frozen references
- Dataset: 51 valid runs, 9 Speed×Load groups, 41 force transitions.
- Data SHA256: dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059
- V11 Joint Group-MAE: 23.86044078619803 N.
- V23 Joint Group-MAE: 23.104189837450967 N.
- V23 LC1 Group-MAE: 13.009658000484931 N.
- V23 LC5 Group-MAE: 33.198721674417 N.

## Non-negotiable guards
- Complete Speed×Load groups are held out.
- No random split.
- No target modification.
- LC1 prediction is exactly frozen to V23 in every V26 arm.
- Residual targets used for LC5 correction are generated from outer-training cross-fitted V23 predictions.
- Current/future target values are never inference features.
- Transition probability is predicted using only pass history/state available before the target pass.
- Pass regimes are predeclared: T2 / T3–T5 / T6.
- No Kaggle run is cancelled or overwritten for capacity.

## Arms
1. V23_BASE
   - Exact V23 state-feature RF reference.

2. LC5_SPECIALIST
   - LC5-only Random Forest delta model.
   - Capacity/hyperparameters selected using group-held-out OOF inside the outer-training universe.
   - LC1 unchanged.

3. GLOBAL_RESIDUAL
   - Low-capacity RF residual correction on cross-fitted V23 LC5 residuals.
   - Uses force conditions + history-derived QLOG state features.
   - No transition classifier or regime split.

4. PASS_REGIME_RESIDUAL
   - Separate residual experts for T2, T3–T5, T6.
   - Global residual fallback if a training regime is too sparse.

5. STATE_GATED_RESIDUAL
   - Stable/rebound residual experts.
   - At inference, blend is controlled by a history-only transition probability model.
   - Actual target-regime labels are used only for outer-training expert fitting.

6. HYBRID_TRANSITION_RESIDUAL
   - Single low-capacity residual RF with predicted transition probability, pass-regime indicators, QLOG dynamics and interactions.

## Primary endpoint
Joint Group-MAE on the original force endpoint.

## Key secondary endpoints
- LC5 Group-MAE.
- LC1 Group-MAE guard: must remain exactly equal to V23 across all arms.
- Per-pass LC5 MAE for T2–T6.
- Group-level paired improvement versus V23.
- Exact sign-flip p and group bootstrap CI.

## Interpretation
V26 is post-lock exploratory repeated development on the same 9 groups. An improvement is a development signal, not independent confirmation. A stronger scientific claim requires new independent experimental groups/runs.
