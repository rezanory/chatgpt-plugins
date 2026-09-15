# SoilBin V4 provenance and evidence boundary

V4 uses the frozen 51-run SoilBin feature table and raw LC1/LC5 time-series only.

- Raw time-series SHA256: `ea227f412827a962a1876ceb9aeb9946034b435052e9e74115e32bd7dca73cb0`
- Current-pass waveform descriptors and raw traces are forbidden as predictors.
- Only previous-pass raw traces, previous peaks, and current load/speed/pass are predictors.
- Event windows and DCT/spectral transforms are label-free and deterministic.
- Outer validation is leave-one-VxW-group-out; inner model/architecture selection uses training groups only.
- Deep CNN results are exploratory because only 41 history pairs / 9 groups exist.
- No external label is used for fitting or selection in V4. V3 remains the external-validation lane.
- Response remains calibrated vertical force proxy (N), not verified soil stress (kPa). Depth mapping is provisional.

- Frozen V3 internal champion `D_delta_prev_peak` (joint group-MAE 24.28878365273039 N) is reproduced inside V4 using the exact V3 candidate grid and grouped protocol before any raw-waveform comparison.
- V5 policy SHA256: `c3c802b74ee0426e23d6c8ea6e4fd9ca26cf1128f5da3a32ca2b0b20aa6c7c02`; V4 applies its <2% simplicity and >=5%/paired-evidence admission rules without external labels.

## V4 compact waveform transport

For transport/reproducibility, V4 embeds a lossless compact binary of only the four raw-source columns actually used by the V4 waveform model:
`Run_ID`, `time_ms`, `LC1_force_delta_N`, `LC5_force_delta_N`.

- Source raw CSV SHA256 (provenance): `ea227f412827a962a1876ceb9aeb9946034b435052e9e74115e32bd7dca73cb0`
- Compact uncompressed blob SHA256: `786cd72be8a6e37aa0e34cee044f70b047c9a84f5a9230abc9028117db765677`
- Compact XZ SHA256: `6ccca93e5e146e67908e37cf07575bdf1ed44b0f3b84fcf0ee3198cd2ffa02c9`
- Rows: `80958`; runs: `51`
- This is a lossless numerical extraction of the used columns; omitted raw columns are not used anywhere in V4 modeling.
