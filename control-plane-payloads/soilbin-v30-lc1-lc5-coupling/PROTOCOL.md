# SoilBin V30 — LC1↔LC5 Coupling Protocol

## Objective
Study LC1 relative to LC5 across all available Speed × Load × Pass conditions, including same-pass, cross-pass, ratio, transition, impulse, timing, and waveform relationships.

## Historical same-pass verification
V27 already executed all 12 same-pass directed cross-sensor edges:
T1..T6 LC1→LC5 and T1..T6 LC5→LC1.
V30 does not relabel those as new experiments; it imports them as historical directed baselines.

## V30 analyses
- LC5/LC1 and LC1/LC5 peak-force ratios.
- Absolute and relative LC5−LC1 differences.
- log(LC5/LC1).
- Stratification by Pass, Speed, Load, Speed×Load and individual Speed×Load×Pass conditions.
- Peak ratio vs impulse ratio and peak-time delay.
- Adjacent-pass transition changes and log-elasticity.
- Nested leave-one-Speed×Load-group-out ratio models.
- Waveform-level zero-lag and lag-scanned LC1↔LC5 coupling.
- Lag-aligned waveform scale and R².
- Leave-one-Speed×Load-group-out inverse-speed delay model.

## Leakage guards
- No random row split.
- Predictive ratio and delay models hold out the whole Speed×Load group.
- Hyperparameter selection uses training groups only.
- Previous-ratio/scale features use only the observed immediately prior pass.
- No future-pass target is used.

## Interpretation guard
The measured LC1↔LC5 delay is an observed channel/sensor response offset. It must not be labeled soil-propagation delay until sensor geometry, longitudinal spacing, acquisition synchronization, and channel timing are independently verified.
