# SoilBin — Final Scientific Findings (V27 + Full Bidirectional Map)

## Scope and validation

- 51 valid runs, 9 Speed×Load groups, T1–T6, LC1 (15 cm) and LC5 (5 cm).
- All reported predictive results use group-held-out evaluation; no random row split.
- Forecast, reconstruction, and smoothing are kept separate. Reverse/future actual measurements are never used as forward inference predictors.
- 132/132 directed edges, 66/66 bidirectional systems, and 1116/1116 multi-history tasks were executed; no multi-history task was skipped.

## Direct answers to the 20 predeclared questions

1. **Is reverse prediction usually easier than forward?** No general rule. Across 60 temporal unordered node pairs, reverse had lower raw MAE in 28/60 and lower target-SD-normalized MAE in 31/60. The aggregate is close to balanced, but individual directions can be extremely asymmetric.
2. **Largest forward/reverse asymmetry?** T1-LC1__TO__T2-LC5 vs T2-LC5__TO__T1-LC1; normalized-MAE difference = 1.186, raw MAEs = 144.62 vs 15.43 N.
3. **Did joint forward+reverse training improve forward prediction generally?** No. Restricting the comparison to the 60 true temporal Forward directions, joint forward+reverse training improved 22/60 Forward edges; mean Forward gain vs the independent model was -7.139 N and median gain was -2.261 N (negative means worse). Some individual edges improved strongly, but the effect was not general.
4. **Did cycle consistency help?** Selectively, not generally. On the same 60 true Forward directions, cycle correction improved 26/60 edges; mean Forward gain vs independent was -4.658 N and median gain was -2.341 N. Thus cycle information can rescue specific relations but should not be imposed universally.
5. **Can LC1 reduce LC5 error?** Sometimes, but not as an always-on fusion rule. Combined LC1+LC5 beat the best single-sensor input in 51/186 LC5 fusion comparisons, while mean gain was -4.065 N. For true Forecasting, a strong selected exception is passes T2+T3+T4+T5 → T6-LC5: LC1-only MAE = 67.89 N, LC5-only = 52.30 N, and LC1+LC5 = 36.83 N, a 15.47 N (29.6%) gain over the best single-sensor input.
6. **Does LC5 provide independent information for LC1?** Yes in selected relations, but universal fusion is not justified. LC1 fusion improved only 20/186 tasks and mean gain was -2.824 N. Nevertheless LC5 nodes dominate the directed-information hubs; T6-LC5 has the highest mean source gain (15.48 N).
7. **Best memory depth for LC1?** Target-dependent: T2=1 pass(es), T3=2 pass(es), T4=3 pass(es), T5=2 pass(es), T6=3 pass(es). Median selected depth = 2.0.
8. **Best memory depth for LC5?** Target-dependent: T2=1 pass(es), T3=2 pass(es), T4=2 pass(es), T5=2 pass(es), T6=4 pass(es). Median selected depth = 2.0; notably T6 selected depth 4.
9. **Is full history better than last 1/2 passes?** No. Full contiguous history was best in only 3/8 comparable sensor×target cases. More history can add variance/irrelevant state.
10. **Can early passes predict T5/T6?** They contain information, but performance is target/sensor dependent. Best forecasts restricted to T1/T2-only sources are:
   - T5 LC1: LC1_ONLY, passes 1+2, MAE 19.60 N
   - T5 LC5: LC5_ONLY, passes 2, MAE 52.37 N
   - T6 LC1: LC1_ONLY, passes 1+2, MAE 28.49 N
   - T6 LC5: LC1_LC5, passes 2, MAE 45.50 N
11. **Can late passes accurately reconstruct T1/T2?** Not uniformly. Best full-future reconstructions are:
   - T1 LC1: LC1_ONLY, MAE 22.38 N
   - T1 LC5: LC1_LC5, MAE 89.47 N
   - T2 LC1: LC1_LC5, MAE 25.78 N
   - T2 LC5: LC5_ONLY, MAE 51.17 N. Early LC5 remains especially difficult to reconstruct, so late state does not preserve all early surface-state information.
12. **Largest information hub?** T6-LC5; mean gain over conditions-only across outgoing edges = 15.48 N and positive-gain fraction = 72.7%.
13. **Is T1–T2 a distinct regime?** Evidence is suggestive but not closed. Descriptive clustering is weak (best silhouette 0.217), and its k=2 partition does not cleanly equal T1–T2 / T3–T5 / T6. A simple LC5 |Δ| change-point criterion preferred after T2; this is a different statistic from the previously observed modeled T5→T6 breakpoint.
14. **How much LC5 error has representation/modeling reduced?** V23 reduced LC5 by about 1.620 N vs V11. V27's best arm changes LC5 by 0.000 N relative to V23; its exact direction and paired uncertainty are reported in the V27 row of 11_lc5_error_reduction_ablation.csv.
15. **Is part of LC5 error irreducible?** Not proven. Repeated failure of always-on specialists/residual corrections plus strong directional asymmetries supports a hidden-state/contact-geometry bottleneck, but irreducibility would require additional measurements or repeated-condition experiments.
16. **Best next architecture?** A hierarchical soil-state model: shared low-capacity backbone for common physics + pass-specific and sensor-specific heads, sparse/learned routing, and an LC5 probabilistic head for transition occurrence and conditional amplitude. Avoid a single monolithic global head.
17. **Separate LC1 and LC5 experts?** Use separate heads/experts, yes; but keep a shared backbone and allow selective cross-sensor fusion. Fully shared and always-fused designs both underperformed broadly.
18. **Pass-specific experts?** Yes, preferably gated/regularized rather than six fully independent high-capacity models. Memory depth and directionality differ materially by target pass.
19. **Global shared model vs edge-specific?** Edge-specific won broadly: global was better in only 20/132 edges and worse in 112/132; mean global-vs-edge-specific improvement = -10.652 N.
20. **Which pass subsets give high information with minimal sensor requirement?** There is no single universal subset. Best one-sensor forecast per target is tabulated below; selected source sets are generally short for middle passes and deeper for T6/LC5.

|   target_pass | target_sensor   | sensor_mode   | source_passes   |   source_count |   MAE_N |
|--------------:|:----------------|:--------------|:----------------|---------------:|--------:|
|             2 | LC1             | LC5_ONLY      | 1               |              1 | 24.6377 |
|             2 | LC5             | LC5_ONLY      | 1               |              1 | 40.3061 |
|             3 | LC1             | LC1_ONLY      | 1+2             |              2 | 14.082  |
|             3 | LC5             | LC5_ONLY      | 1+2             |              2 | 44.0362 |
|             4 | LC1             | LC1_ONLY      | 1+2+3           |              3 | 11.7385 |
|             4 | LC5             | LC5_ONLY      | 2+3             |              2 | 29.7618 |
|             5 | LC1             | LC1_ONLY      | 3+4             |              2 | 13.7006 |
|             5 | LC5             | LC5_ONLY      | 3+4             |              2 | 34.5176 |
|             6 | LC1             | LC1_ONLY      | 3+4             |              2 | 24.0841 |
|             6 | LC5             | LC5_ONLY      | 1+3             |              2 | 44.7577 |

## V27 probabilistic closure

- V27 champion: **V23_BASE** — Joint Group-MAE 23.104190 N; LC1 13.009658 N; LC5 33.198722 N.
- Transition classifier OOF: Brier 0.208552; AUROC 0.750000.
- P25–P75 nominal coverage 0.50; observed OOF coverage 0.3171; mean interval width 49.12 N.
- LC1 was frozen to V23 in every V27 arm: True.

## Main scientific interpretation

The SoilBin response is not a simple one-way Markov chain. Information is strongly direction- and depth-dependent. LC5 is difficult as a prediction target but often highly informative as a source, especially at later passes. This combination is consistent with LC5 carrying rich but unstable surface/contact-state information. A useful next model should therefore preserve shared physical structure while avoiding indiscriminate pooling and indiscriminate sensor fusion.