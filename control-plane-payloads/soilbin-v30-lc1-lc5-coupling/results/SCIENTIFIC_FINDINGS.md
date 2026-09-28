# SoilBin V30 — LC1↔LC5 Coupling Study

- Runs: 51; Speed×Load groups: 9.
- V27 same-pass directed LC1↔LC5 edges verified: 12 (T1..T6, both directions).
- All V27 cross-sensor directed edges verified: 72.
- Overall LC5/LC1 median peak ratio: 1.7623; mean: 1.7672; CV: 35.4%.
- Fraction of runs with LC5 > LC1: 88.2%.
- Best leakage-safe same-pass ratio model for LC5-from-LC1: GLOBAL_MEDIAN_RATIO with group-equal MAE 62.916 N.
- Best leakage-safe same-pass ratio model for LC1-from-LC5: LC1_FROM_LC5_RATIO_PREV with group-equal MAE 36.918 N.

## Pass-level coupling
- T1: n=7, ratio median=2.102, Pearson r=0.06951226775911787, log-elasticity=-0.053.
- T2: n=8, ratio median=1.719, Pearson r=0.5922611236085483, log-elasticity=1.355.
- T3: n=9, ratio median=1.755, Pearson r=0.26166419544778297, log-elasticity=0.483.
- T4: n=9, ratio median=1.631, Pearson r=0.3892792425933238, log-elasticity=0.326.
- T5: n=9, ratio median=1.821, Pearson r=0.22946122447126685, log-elasticity=0.333.
- T6: n=9, ratio median=1.606, Pearson r=0.3921466278085587, log-elasticity=1.261.

## Adjacent transitions
- T1->T2: n=6, mean ratio change=-0.090, median elasticity=1.5988472595510739.
- T2->T3: n=8, mean ratio change=-0.014, median elasticity=1.0136184691459433.
- T3->T4: n=9, mean ratio change=-0.035, median elasticity=0.9355816465503132.
- T4->T5: n=9, mean ratio change=-0.001, median elasticity=1.1207455510400257.
- T5->T6: n=9, mean ratio change=-0.219, median elasticity=0.2191701792177996.