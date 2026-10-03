# Phase-2 dual precision/recall amendment and ensemble evidence audit

User continuation: 2026-10-03T12:58:25Z. Source commit: e339dab78725ef90bec1e81b22099c94b694af69.

## Scope and scientific classification

This is an additive POST_RESULTS_EXPLORATORY_DUAL_METRIC_AMENDMENT. The original
balanced-accuracy hypothesis family and its 216-row output are preserved, not
rewritten or retrospectively reclassified as having two preregistered metrics.
The user requested both macro precision and macro recall after seeing results.
Each is the unweighted mean over the normal and pneumonia classes.

Reuse the sealed OOF artifact from GitHub run 37118976126, artifact 11272493898.
`source_lock.json` pins all 40 files from that artifact. No Kaggle job, training,
inference, HPO, calibration, threshold fitting or ensemble selection is started.

72 amended rows = 12 original contrasts x 3 resolutions x 2 metrics. Use 5000
paired patient-cluster bootstraps; seed 260915 plus the existing deterministic
contrast offset. The 36 macro-recall rows must reproduce the original balanced-
accuracy estimates, percentile intervals and approximate bootstrap-tail p-values
within 1e-12. Holm across all 72 is the joint amendment family; separate 36-row
per-metric corrections are also supplied as supplementary information. These are
approximate bootstrap-tail p-values, not exact permutation p-values.

The M09/R384 versus M10/R320 comparison is a separate, post-hoc exploratory
contrast selected from observed results. Its two-row family is NOT adjusted for
the prior model search and cannot support confirmatory superiority claims.
The descriptive nondominated set is not a deployment or scientific freeze.

## Operational acceptance

Input: 40 hash-pinned source files, including 36 patient-aligned OOF vectors.
Runtime: GitHub-hosted Python 3.12, numpy 2.0.2, scikit-learn 1.6.1.
Execution: focused algorithm tests, then actual sealed-OOF calculations.
Output: 72 amended statistical rows, 2 exploratory finalist rows, 36 descriptive
rows, policy JSON, operational receipt and artifact hashes.
Acceptance: all source files unchanged before/after; full coverage; patient,
image-identity and label alignment; exact original recall reconciliation;
explicit false locked-test/external selection flags. Synthetic tests alone are
not operational completion. See the workflow artifact for actual run status.

## Verified historical ensemble evidence

1. Within-model five-fold M07 ensemble is executed in the external report from
   query run 36486023072. Primary aggregation is the mean of
   logit(p_fold) - logit(frozen_validation_threshold_fold), thresholded at zero;
   majority vote is a separate secondary output. This is not a single fold.
2. M07 resolution ensemble (224/320/384) executed in run 36636935920,
   artifact 11064627434, file M07_RSNA_RESOLUTION_ENSEMBLE_V1.json, status PASS.
   It includes fixed equal-weight and majority-vote exploratory results on 1099
   images from 553 patients. Equal: TN=384 FP=194 FN=36 TP=485. Majority:
   TN=381 FP=197 FN=36 TP=485. Baseline R224: TN=400 FP=178 FN=37 TP=484.
3. Advanced M07 resolution ensemble executed in run 36865340637,
   artifact 11163415909, file M07_RSNA_ADVANCED_ENSEMBLE_V1.json, status
   PASS_ADVANCED_ENSEMBLE_ANALYSIS. Rank, median, confidence-gating and fallback
   methods are present, not merely implemented source code.
4. Historical files also contain EXTERNAL_CROSSFIT_ADAPTATION analyses that
   learned thresholds or weights from external labels. Preserve them as
   historical secondary/adaptation evidence only. They are not untouched
   external validation and must not choose or tune the final M01-M12 model.
   This amendment does not rerun those procedures.
5. A cross-model M09+M10 or final M01-M12 champion ensemble is not evidenced by
   the verified 36-unit OOF collection or the M07-only ensemble artifacts above.
   Do not report it as performed merely because M07 ensembles exist.

External positive endpoint is adjudicated lung opacity, not identical to the
internal pneumonia label. External cohort and internal OOF results remain
separate. External artifacts were read only to answer the ensemble-status
question; the new statistical script cannot read or fit those external outputs.
