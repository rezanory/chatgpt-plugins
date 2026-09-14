# SoilBin V3 provenance and evidence boundary

V3 uses the frozen SoilBin 51-run feature table for all internal model selection. External labels are never used for hyperparameter/model selection.

Independent numeric external evidence:
- Ayetan et al. (2026), Soil Science Society of America Journal, DOI 10.1002/saj2.70200 — published peak mean soil stress at 15/30/50 cm under front/rear and inflated/deflated tractor tires.
- Olmstead & Fischer (2009), ERDC/CRREL TR-09-2 — public government report; MDT sequential passes at buried pressure-pad depths plus CIV independent trials.

Structured corroboration only (not row-level external prediction tests):
- Keller et al. (2014), DOI 10.1016/j.still.2014.03.001.
- Arvidsson & Keller (2007), DOI 10.1016/j.still.2007.06.012.
- Keller et al. (2016), DOI 10.2136/sssaj2015.07.0252.
- Bahrami et al. (2023), DOI 10.1016/j.biosystemseng.2023.04.013.

Near-domain replication only, not independent primary external validation:
- Gheshlaghi & Mardani (2021), DOI 10.1016/j.jterra.2021.02.004.
- Farhadi et al. (2025), DOI 10.1038/s41598-025-13535-w.

Scientific guardrails:
- Internal response remains calibrated vertical force proxy (N), not verified soil stress (kPa).
- External kPa values therefore support independent trend/physics validation only until effective sensor area or direct stress calibration is established.
- LC5→5 cm and LC1→15 cm mapping remains provisional pending physical confirmation.
- Current-run waveform descriptors are forbidden as predictors; only previous-pass waveform morphology is allowed in waveform history tasks.
