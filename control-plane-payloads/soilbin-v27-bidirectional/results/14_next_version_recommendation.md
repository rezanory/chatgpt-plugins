# SoilBin — Next Version Recommendation

## Recommended architecture

Build the next version as a **hierarchical gated multi-head soil-state model**, not as another deterministic global RF.

1. **Shared low-capacity latent soil-state backbone**
   - Speed, load, pass index/distance, sensor depth, previous valid observations.
   - Regularized linear/PLS/GPR or shallow tree representation first; deep model only as challenger.

2. **Sensor-specific heads**
   - LC1 head optimized for the more stable 15 cm response.
   - LC5 head separated because the 5 cm surface response has materially different uncertainty and memory.

3. **Pass-specific / regime-gated heads**
   - Do not hard-code T1–T2 / T3–T5 / T6 as proven regimes.
   - Learn a low-capacity gate with strong regularization and compare against pass-specific heads under grouped OOF.

4. **Sparse cross-sensor fusion**
   - Always-on fusion is rejected by the 1116-task map.
   - Permit LC1↔LC5 exchange only when a training-only gate predicts positive value.
   - Preserve high-value exceptions such as multi-pass LC1+LC5 support for T6-LC5.

5. **Probabilistic LC5 output**
   - Keep occurrence probability and conditional amplitude/distribution separate.
   - Use V27 calibration/quantile diagnostics as the baseline for probabilistic heads.

6. **Memory routing**
   - Default candidate memory depth around 2 passes.
   - Allow deeper memory for LC5-T6 (selected depth 4 in the current simple-history benchmark).
   - Select depth only inside outer-training.

## What not to do

- Do not replace edge-specific structure with one universal pooled head: it was worse on 112/132 edges.
- Do not force joint forward/reverse loss universally: it was better in only 27/66 pairs.
- Do not force cycle consistency universally: it beat independent models in only 26/66 pairs.
- Do not always concatenate LC1 and LC5: fusion was beneficial in only 71/372 comparisons.

## Proposed next experiment

Call it **V28 — Hierarchical Gated Multi-Head Soil-State Model**.

Predeclare:
- A: V23 frozen baseline
- B: shared backbone + sensor-specific heads
- C: B + pass-specific heads
- D: C + sparse cross-sensor gate
- E: D + probabilistic LC5 head
- F: E + adaptive memory-depth gate

Primary endpoint remains Joint Group-MAE; LC5 Group-MAE and per-pass errors are mandatory secondary endpoints.
Every gate, memory choice, head choice, and calibration parameter must be selected only within outer-training groups.
