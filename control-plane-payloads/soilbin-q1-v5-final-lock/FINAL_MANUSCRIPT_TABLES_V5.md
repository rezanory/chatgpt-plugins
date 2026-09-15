# SoilBin V5 — Manuscript Tables

## Comparable paired-model progression

| Version | Candidate | Joint group-MAE (N) | Status |
|---|---|---:|---|
| V2 | Delta + previous peak + engineered waveform | 26.736 | historical comparable |
| V3 | **Delta + previous peak** | **24.289** | **final champion** |
| V4 | Previous raw waveform (SPECTRAL) | 25.068 | rejected by frozen policy |

V1 C_delta is reported separately as historical context because its 82 sensor-depth observation metric is not identical to the 41-pair dual-target joint metric used by V2–V4.

## Final target-specific metrics

| Target | Group-MAE (N) | MAE (N) | RMSE (N) | Bias (N) | R² |
|---|---:|---:|---:|---:|---:|
| LC1 | 12.785 | 12.960 | 18.725 | 2.233 | 0.533 |
| LC5 | 35.792 | 35.341 | 44.076 | -3.357 | 0.616 |

## External validation ledger

- Independent numeric rows: **70**
- Primary independent datasets: EV_CANADA_AYETAN_2026, EV_ERDC_CRREL_2009
- Corroboration: EV_KELLER_2014, EV_ARVIDSSON_KELLER_2007, EV_KELLER_2016, EV_BAHRAMI_2023
- Near-domain replication only: REP_GHESHLAGHI_MARDANI_2021, REP_FARHADI_2025
- Absolute external validation: **No** (unit/calibration mismatch).
