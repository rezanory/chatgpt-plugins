# SoilBin V29 — Predeclared Hierarchical + Hard-Zone Reverse Stack

## Status

PREDECLARED FOR NEXT ROBUSTNESS / INDEPENDENT EVALUATION.

This protocol is frozen after V28/V28.1 development. It must not be tuned from the next outer-test results.

## Frozen references

- Data SHA256: dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059
- V23 Joint Group-MAE: 23.104189837450967 N
- V23 LC1 Group-MAE: 13.009658000484931 N
- V23 LC5 Group-MAE: 33.198721674417 N
- V28 predeclared development champion HIER_SELECTIVE: Joint 22.73940534358568 N
- V28.1 exploratory hard-zone stack: Joint 22.10565925264389 N, LC5 31.20166050480285 N
- Post-hoc architecture headroom is NOT a validation target and must not be cited as achieved V29 performance.

## Scientific hypothesis

The SoilBin system benefits from different predictive mechanisms by pass and sensor:

- LC1 benefits from the hierarchical selective memory architecture.
- LC5-T2 has the strongest evidence for a Forward + legal Reverse-inverted stacked head.
- LC5-T3/T4 benefits from hierarchical selective memory.
- LC5-T5 remains on the V23 backbone unless a training-only challenger gate is predeclared before outer evaluation.
- LC5-T6 may use a reverse-stacked challenger only behind a training-only reliability gate because V28.1 showed only a small T6 gain and the edge map showed heterogeneous T6 behavior.

## Fixed routing

### LC1
Use the V28 HIER_SELECTIVE mechanism for T2-T6.

### LC5
- T2: hard-zone OOF stacker of V23 Forward + legal Reverse-inverted candidate.
- T3: hierarchical selective memory head.
- T4: hierarchical selective memory head.
- T5: V23 backbone.
- T6: V23 backbone + reverse-stacked challenger under a training-only gate.

No pass may change its assigned family based on outer-test error.

## Reverse legality

At inference the reverse branch never receives the true target pass response.

The reverse candidate must be generated only from:
- training-fitted reverse relation,
- known prior/source force values,
- speed/load/pass/depth/state features available before the target,
- algebraic/model inversion or an equivalent target-free reverse-informed mechanism.

## Candidate arms

A. V23_BASE

B. V28_HIER_SELECTIVE

C. V29_FIXED_ROUTING
- fixed routing above.

D. V29_FIXED_ROUTING_NO_REVERSE
- same hierarchical routing but replace T2/T6 reverse-stack components with V23.
- isolates the contribution of reverse information.

E. V29_T2_ONLY_REVERSE
- only LC5-T2 uses the reverse stack.
- T6 remains V23.
- directly tests whether T2 accounts for nearly all reverse-stack benefit.

F. V29_T2_T6_REVERSE
- LC5 T2 and gated T6 reverse stack.
- compare against E to quantify the incremental T6 contribution.

## Validation

Primary:
- Joint Group-MAE

Secondary:
- LC1 Group-MAE
- LC5 Group-MAE
- LC1/LC5 MAE by T2-T6
- group-level paired improvements
- exact sign-flip test
- group bootstrap 95% CI
- calibration / disagreement diagnostics for T6 gate
- fraction of groups helped vs harmed

Guardrails:
- entire Speed×Load groups held out
- no random row split
- no target modification
- no current/future target leakage
- all model/gate/weight fitting inside training only
- no outer-test-driven rerouting
- no overwrite or cancellation of prior evidence

## Scientific interpretation rule

Even if V29 improves on the same 9 historical groups, it remains repeated-development evidence.
Statistical closure requires either:
- new independent experimental groups/runs, or
- a genuinely independent compatible dataset with the same endpoint/measurement semantics.
