# SoilBin Q1/Q2 V5 — Final Frozen Results

## Final selected protocol
The final locked modeling protocol is **V3 D_delta_prev_peak**, retained after the pre-registered V4 raw-waveform challenge. It uses current load, speed and pass number plus the two previous-pass peak force proxies. No current-pass waveform is used.

Under nested leave-one-V×W-sequence-out validation across **41 consecutive history pairs in 9 groups**, joint group-equal MAE was **24.29 N**. LC1 group-equal MAE was **12.79 N** (R²=0.533); LC5 group-equal MAE was **35.79 N** (R²=0.616).

The protocol improves on the persistence benchmark (35.09 N) by **30.8%** and on the V2 delta+waveform coupled protocol (26.74 N) by **9.2%** under the comparable 41-pair joint metric.

## Waveform decision
In V3, adding engineered previous-pass waveform morphology to delta+previous-peak worsened joint Group-MAE from **24.29** to **26.81 N**. In V4, the best raw-waveform representation was **SPECTRAL** at **25.07 N**, **3.21% worse** than the frozen V3 baseline. The paired one-sided sign-flip test for the best raw representation gave p=0.8164. Therefore the pre-registered selection rule retains the simpler V3 protocol.

## External evidence
After internal selection was frozen, V3 evaluated **70 independent numeric external rows** from Canada/Ayetan 2026 and ERDC/CRREL 2009 for physics/trend consistency. Canada showed rear>front stress in 100% of comparisons, lower stress for deflated versus inflated treatment in 83.3%, and 15-cm>50-cm stress in 75.0%. ERDC repeated-pass groups showed positive pass slopes in 66.7% of the retained MDT groups, with original device/synchronization limitations preserved.

These external data are reported in **kPa**, whereas the SoilBin response remains a **calibrated vertical force proxy in N**. Accordingly, the external evidence supports trend/physics consistency, not absolute cross-unit predictive validation.

## Final interpretation boundary
No absolute N↔kPa claim is permitted until effective sensor area/direct stress calibration is established. LC5→5 cm and LC1→15 cm remain provisional until the physical sensor layout is confirmed. The 102 depth-level observations originate from 51 experimental runs and are not 102 independent experiments.
