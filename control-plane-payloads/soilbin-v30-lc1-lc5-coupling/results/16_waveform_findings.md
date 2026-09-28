# SoilBin V30 — Waveform-Level LC1↔LC5 Findings

- Canonical waveform source: timeseries_long_lc1_lc5.csv, 80,958 rows, 51 runs.
- Zero-lag waveform correlation is weak because the two sensor responses are temporally offset.
- After allowing a ±4 s shift, median LC1↔LC5 cross-correlation is 0.9861.
- After lag alignment, median per-run waveform R² is 0.9723 (mean 0.9541).
- Cross-correlation lag and direct peak-time delay agree extremely closely: r=0.9952; median absolute difference 30 ms.
- Median peak delay by speed level: V1=2337 ms, V2=1200 ms, V3=737 ms.
- Delay is almost inverse to speed level: corr(delay, 1/speed)=0.9898; speed×delay CV=5.73%.
- A leave-one-Speed×Load-group-out model delay=a*(1/speed)+b achieved R²=0.9626 and MAE=95.2 ms.
- Lag-aligned amplitude scale median=1.5351 and correlates strongly with peak LC5/LC1 ratio (r=0.9395).
- Amplitude scale is much less predictable than lag: the best simple sequential scale model still gives ~60.4 N LC5 peak MAE.
- Interpretation: temporal alignment/shape coupling is highly structured, while amplitude coupling remains condition/state dependent.
- Engineering caveat: the speed-dependent delay may reflect sensor/tool spatial geometry or acquisition synchronization; it must not be labeled soil-propagation delay until geometry/timing metadata are checked.
