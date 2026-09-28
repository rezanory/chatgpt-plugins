# SoilBin V17–V25 Adaptive-State Research Protocol

## Fixed rules
- Post-lock exploratory only; V5 and measured targets remain immutable.
- 51 valid runs, 9 Speed×Load groups, 41 force transitions.
- No random train/test split.
- Every reported performance estimate holds out complete Speed×Load groups.
- Any within-sequence adaptation may use only observations from prior passes.
- No current/future target may affect the prediction being scored.
- No Kaggle job is cancelled, overwritten, or displaced to make capacity.

## Parallel stage
### V17E — functional ensemble/gating
Five arms, kept separate:
A. Quadratic-Log only.
B. Equal four-expert ensemble.
C. Leakage-safe convex blend.
D. Pass-specific gating.
E. History/state supervised regime gate.

Experts:
- Quadratic-Log
- Piecewise decay→rebound
- Saturating exponential decay
- Persistence

### V17T — T6 regime-transition validation
- paired D6 versus D5
- exact sign-flip and directional tests
- leave-one-group-out breakpoint stability
- prospective rebound classifiers using only pre-target history/state
- no claim of collapse/fatigue/shakedown class without deformation evidence

### V18 — reward/penalty factorial
- no update
- reward only
- penalty only
- reward + penalty
Learning intensity selected within outer-training groups only.

### V19 — intensity law
- constant
- pass-increasing
- pass-decreasing
- history-ratio dependent
- state-magnitude dependent
- hybrid

### V20 — update functional form
- exponential
- power
- hyperbolic
- bounded tanh
- clipped linear

### V21 — transition policy
- keep memory
- soften memory after rebound
- reset adaptive multiplier after rebound
- T6-specific prior learned from training groups

### V22 — channel policy
- shared static
- channel-specific static
- shared dynamic
- channel-specific dynamic
- hierarchical shared + channel dynamic

## Fan-in stage
### V23 — force-level integration
Reproduce exact V11 RF_SHALLOW_WEIGHTED baseline:
Joint Group-MAE = 23.86044078619803 N.

Then compare:
- V11 baseline
- state-feature augmented RF
- V11 magnitude scaling toward predicted state dynamics
- augmented RF + magnitude scaling

This is the first stage where the new state/regime program is tested on the same force endpoint as V11.

### V24 — component ablation
Starting from the V23 composite, disable one component at a time:
- functional ensemble/gating
- reward/penalty
- adaptive intensity
- non-default update function
- transition policy

Report force-level removal harm.

### V25 — frozen robustness and external corroboration
Internal:
- group bootstrap uncertainty for frozen force candidate.

External:
- ERDC/CRREL repeated-pass pressure sequences: scale-free dynamics only.
- Canada/Ayetan: directional load/stress corroboration only.
- No direct force-model claim across incompatible units/sensors/designs.

## Execution graph
V17E, V17T, V18, V19, V20, V21, V22 run independently at maximum safe Kaggle concurrency.
After all upstream lanes are terminal:
V23 fan-in.
After V23 is terminal:
V24 and V25 run in parallel.

No downstream lane may alter an upstream result.
