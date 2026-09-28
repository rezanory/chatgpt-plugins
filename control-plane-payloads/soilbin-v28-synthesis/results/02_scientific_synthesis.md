# SoilBin V28 Synthesis

## Strongest valid development results

### V28 hierarchical selective
- Joint Group-MAE: 22.739405 N
- LC1: 12.659261 N
- LC5: 32.819549 N
- Improvement vs V23 Joint: 0.364784 N
- Status: grouped OOF development result, not statistically closed.

### V28.1 hard-zone reverse stack
- Arm: LC5_T2T6_STACKED
- Joint Group-MAE: 22.105659 N
- LC1: 13.009658 N
- LC5: 31.201661 N
- Joint improvement vs V23: 0.998531 N (4.322%)
- LC5 improvement vs V23: 1.997061 N (6.015%)
- Status: post-hoc exploratory because the stacked hard-zone design followed the edge-map ensemble signal.

## Direct answer to Forward plus Reverse ensemble hypothesis

Simple reverse-only, equal average, global convex, and generic gated ensembles did not improve the endpoint. However, an OOF-trained stacked ensemble showed a positive edge-level signal and a targeted LC5 hard-zone version improved the main endpoint.

Edge map:
- STACKED improved 34/60 true Forward edges.
- Mean edge gain: 1.930 N.
- LC5 T2/T6 subset mean gain: 7.620 N.

Force endpoint:
- T2-only stack: Joint 22.315806 N.
- T2/T6 stack: Joint 22.105659 N.
- The major gain comes from LC5-T2; T6 adds only a small additional improvement.

## Statistical caution

The strongest V28.1 result improved 5/9 groups; its exact one-sided sign-flip p-value is 0.244141. The bootstrap improvement CI crosses zero: [-1.597097857144554, 3.632929770903407]. Therefore it is a strong development hypothesis, not statistical closure.

## Post-hoc architecture headroom

The file 01_candidate_comparison.csv includes two explicitly post-hoc composites. They are provided only to estimate architectural headroom and must not be cited as validated performance.

Best descriptive headroom candidate:
- POSTHOC_PASS_SPECIALIZED_HEADROOM
- Joint: 20.970434 N
- LC1: 12.659261 N
- LC5: 29.281608 N

## Next predeclared experiment

Use V29_HIERARCHICAL_WITH_HARD_ZONE_REVERSE_STACK and freeze all routing before outer-test evaluation:
- LC1: hierarchical selective head.
- LC5-T2: Forward plus legal reverse stack.
- LC5-T3/T4: hierarchical memory head.
- LC5-T5: V23 unless a training-only gate selects a challenger.
- LC5-T6: reverse stack only behind a training-only gate.
