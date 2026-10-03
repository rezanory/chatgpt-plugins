"""Additive, post-results OOF precision/recall analysis; no fitting or selection.

The original balanced-accuracy analysis remains unchanged. The user requested
both macro precision and macro recall after viewing the original results, so
this amendment and the finalist contrast are explicitly exploratory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
from datetime import UTC, datetime

import numpy as np
import phase2_paired_oof_stats_v1 as base

METRICS = ("macro_precision", "macro_recall")
CLASSIFICATION = "POST_RESULTS_EXPLORATORY_DUAL_METRIC_AMENDMENT"
SOURCE_RUN = "37118976126"
BOOTSTRAPS = 5000
SEED = 260915


def fail(message):
    raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(root, lock_path):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("source_run_id") != SOURCE_RUN:
        fail("SOURCE_RUN_MISMATCH")
    records = lock.get("files", {})
    if len(records) != 40:
        fail("SOURCE_LOCK_COVERAGE_INVALID")
    for relative, expected in records.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            fail("SOURCE_PATH_ESCAPE")
        if not path.is_file() or sha(path) != expected:
            fail("SOURCE_FILE_SHA_MISMATCH:" + relative)
    return lock


def check_pair(ref, cand):
    if ref["identity_sequence_sha256"] != cand["identity_sequence_sha256"]:
        fail("PAIR_IMAGE_IDENTITY_MISMATCH")
    if not np.array_equal(ref["patient_id"], cand["patient_id"]):
        fail("PAIR_PATIENT_ORDER_MISMATCH")
    if not np.array_equal(ref["label"], cand["label"]):
        fail("PAIR_LABEL_MISMATCH")
    n = len(ref["label"])
    for v in (ref, cand):
        if any(len(v[k]) != n for k in ("patient_id", "label", "prediction", "score")):
            fail("PAIR_LENGTH_MISMATCH")
        if not np.isin(v["label"], [0, 1]).all():
            fail("LABEL_NOT_BINARY")
        if not np.isin(v["prediction"], [0, 1]).all():
            fail("PREDICTION_NOT_BINARY")
        if v.get("locked_test_used_for_selection") is not False:
            fail("LOCKED_TEST_POLICY_NOT_FALSE")
        if v.get("external_used_for_selection") is not False:
            fail("EXTERNAL_POLICY_NOT_FALSE")
    if len(np.unique(ref["label"])) != 2:
        fail("BOTH_CLASSES_REQUIRED")


def pr_metrics(cm):
    cm = np.asarray(cm, dtype=float)
    if cm.ndim == 1:
        cm = cm[None, :]
    if cm.ndim != 2 or cm.shape[1] != 4 or not np.isfinite(cm).all() or (cm < 0).any():
        fail("CONFUSION_INPUT_INVALID")
    tn, fp, fn, tp = (cm[:, i] for i in range(4))

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)

    p0, p1 = divide(tn, tn + fn), divide(tp, tp + fp)
    r0, r1 = divide(tn, tn + fp), divide(tp, tp + fn)
    return {"macro_precision": (p0 + p1) / 2, "macro_recall": (r0 + r1) / 2}


def paired_pr(ref, cand, n_boot=BOOTSTRAPS, seed=SEED):
    check_pair(ref, cand)
    if int(n_boot) < 100:
        fail("TOO_FEW_REQUESTED_BOOTSTRAPS")
    y = np.asarray(ref["label"], dtype=int)
    patients, index = np.unique(ref["patient_id"], return_inverse=True)
    n = len(patients)
    if n < 2:
        fail("TOO_FEW_PATIENTS")
    contributions = [
        base._patient_confusion_contributions(index, y, v["prediction"], n) for v in (ref, cand)
    ]
    points = [pr_metrics(c.sum(axis=0)) for c in contributions]
    classes = np.stack(
        [np.bincount(index, weights=(y == c).astype(float), minlength=n) for c in (0, 1)], axis=1
    )
    rng = np.random.default_rng(int(seed))
    chunks = {m: [] for m in METRICS}
    block = max(8, min(128, 2_000_000 // n))
    remaining = int(n_boot)
    while remaining:
        size = min(block, remaining)
        sampled = rng.integers(0, n, size=(size, n), dtype=np.intp)
        offsets = np.arange(size, dtype=np.intp)[:, None] * n
        counts = np.bincount((sampled + offsets).reshape(-1), minlength=size * n).reshape(size, n)
        totals = counts @ classes
        counts = counts[(totals[:, 0] > 0) & (totals[:, 1] > 0)]
        if len(counts):
            a, b = (pr_metrics(counts @ c) for c in contributions)
            for metric in METRICS:
                chunks[metric].append(b[metric] - a[metric])
        remaining -= size
    rows = []
    for metric in METRICS:
        if not chunks[metric]:
            fail("NO_VALID_BOOTSTRAPS")
        delta = np.concatenate(chunks[metric])
        if len(delta) < max(100, int(n_boot) // 2):
            fail("TOO_FEW_VALID_BOOTSTRAPS")
        left = (np.sum(delta <= 0) + 1) / (len(delta) + 1)
        right = (np.sum(delta >= 0) + 1) / (len(delta) + 1)
        rows.append(
            {
                "metric": metric,
                "reference": float(points[0][metric][0]),
                "candidate": float(points[1][metric][0]),
                "delta_candidate_minus_reference": float(
                    points[1][metric][0] - points[0][metric][0]
                ),
                "ci95_low": float(np.quantile(delta, 0.025)),
                "ci95_high": float(np.quantile(delta, 0.975)),
                "p_raw_two_sided": float(min(1.0, 2 * min(left, right))),
                "bootstrap_unit": "PATIENT_CLUSTER_PAIRED",
                "n_patients": n,
                "n_bootstrap_valid": int(len(delta)),
                "bootstrap_seed": int(seed),
                "analysis_classification": CLASSIFICATION,
            }
        )
    return rows


def adjust(rows):
    for i, value in enumerate(base.holm_adjust([r["p_raw_two_sided"] for r in rows])):
        rows[i]["p_holm_dual_global"] = float(value)
        rows[i]["reject_holm_dual_global_0_05"] = bool(value < 0.05)
    for metric in METRICS:
        indices = [i for i, r in enumerate(rows) if r["metric"] == metric]
        for i, value in zip(
            indices, base.holm_adjust([rows[i]["p_raw_two_sided"] for i in indices]), strict=True
        ):
            rows[i]["p_holm_metric_global"] = float(value)
    return rows


def reconcile_recall(rows, original):
    original = {
        (r["reference_model"], r["candidate_model"], int(r["resolution"])): r
        for r in original
        if r["metric"] == "balanced_accuracy"
    }
    checked = 0
    for row in rows:
        if row["metric"] != "macro_recall":
            continue
        old = original[(row["reference_model"], row["candidate_model"], row["resolution"])]
        for key in (
            "reference",
            "candidate",
            "delta_candidate_minus_reference",
            "ci95_low",
            "ci95_high",
            "p_raw_two_sided",
        ):
            if abs(float(row[key]) - float(old[key])) > 1e-12:
                fail("ORIGINAL_RECALL_DRIFT:" + str((row["resolution"], row["contrast"], key)))
        if int(row["n_bootstrap_valid"]) != int(old["n_bootstrap_valid"]):
            fail("ORIGINAL_BOOTSTRAP_COUNT_DRIFT")
        checked += 1
    if checked != 36:
        fail("RECALL_RECONCILIATION_COVERAGE_INVALID")
    return checked


def descriptive_rows(vectors):
    rows = []
    for (model, resolution), v in sorted(vectors.items()):
        check_pair(v, v)
        y, p = v["label"], v["prediction"]
        cm = [int(np.sum((y == a) & (p == b))) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
        m = pr_metrics(cm)
        rows.append(
            {
                "model_id": model,
                "resolution": resolution,
                "n": len(y),
                "macro_precision": float(m["macro_precision"][0]),
                "macro_recall": float(m["macro_recall"][0]),
                **dict(zip(("tn", "fp", "fn", "tp"), cm, strict=True)),
            }
        )
    for row in rows:
        row["descriptive_nondominated"] = not any(
            other["macro_precision"] >= row["macro_precision"]
            and other["macro_recall"] >= row["macro_recall"]
            and (
                other["macro_precision"] > row["macro_precision"]
                or other["macro_recall"] > row["macro_recall"]
            )
            for other in rows
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=pathlib.Path, required=True)
    parser.add_argument("--source-lock", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root, out = args.baseline_root.resolve(), args.output_dir.resolve()
    if out == root or out.is_relative_to(root):
        fail("OUTPUT_MUST_NOT_MUTATE_SOURCE")
    if out.exists():
        fail("OUTPUT_ALREADY_EXISTS")
    lock = verify_source(root, args.source_lock)
    vectors = base.load_vectors(root / "PHASE2_OOF_VECTORS_V1")
    if len(vectors) != 36:
        fail("VECTOR_COUNT_INVALID")
    anchor = next(iter(vectors.values()))
    for v in vectors.values():
        check_pair(anchor, v)
    with (root / "PHASE2_OOF_PAIRED_STATS_V1/PHASE2_OOF_PAIRED_PATIENT_BOOTSTRAP_HOLM_V1.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        original = list(csv.DictReader(f))
    rows = []
    for resolution in base.EXPECTED_RESOLUTIONS:
        for reference, candidate, contrast in base.PREDECLARED_MODEL_COMPARISONS:
            results = paired_pr(
                vectors[(reference, resolution)],
                vectors[(candidate, resolution)],
                seed=SEED + resolution + sum(map(ord, reference + candidate)),
            )
            rows.extend(
                {
                    "dataset": "OOF",
                    "resolution": resolution,
                    "reference_model": reference,
                    "candidate_model": candidate,
                    "contrast": contrast,
                    **r,
                }
                for r in results
            )
    if len(rows) != 72:
        fail("DUAL_STATISTICS_ROW_COUNT_INVALID")
    reconcile_count = reconcile_recall(rows, original)
    adjust(rows)
    # This direct cross-resolution contrast was suggested by observed results.
    # It is not part of the original predeclared 36-comparison family.
    direct = paired_pr(vectors[("M09", 384)], vectors[("M10", 320)], seed=SEED + 90410320)
    for row in direct:
        row.update(
            reference_model="M09",
            reference_resolution=384,
            candidate_model="M10",
            candidate_resolution=320,
            contrast="POSTHOC_FINALISTS_NOT_CONFIRMATORY",
            selection_adjusted_inference=False,
        )
    direct = adjust(direct)
    table = descriptive_rows(vectors)
    # No output directory is created until source and old-estimator checks pass.
    out.mkdir(parents=True)
    base.write_csv(out / "DUAL_PR_PAIRED_HOLM_72.csv", rows)
    base.write_csv(out / "DUAL_PR_FINALISTS_EXPLORATORY_2.csv", direct)
    base.write_csv(out / "DUAL_PR_DESCRIPTIVE_36.csv", table)
    policy = {
        "analysis_classification": CLASSIFICATION,
        "request_utc": "2026-10-03T12:58:25Z",
        "source_run_id": SOURCE_RUN,
        "source_commit": lock["source_commit"],
        "metrics": list(METRICS),
        "original_analysis_preserved": True,
        "multiplicity_primary_for_amendment": "HOLM_ACROSS_72_DUAL_METRIC_COMPARISONS",
        "supplementary_multiplicity": "HOLM_WITHIN_EACH_36_COMPARISON_METRIC_FAMILY",
        "finalist_pair": "SEPARATE_POSTHOC_FAMILY_OF_2;_NOT_SELECTION_ADJUSTED;NOT_CONFIRMATORY",
        "p_value_method": "APPROXIMATE_TWO_SIDED_BOOTSTRAP_TAIL_NOT_EXACT_PERMUTATION",
        "bootstrap_unit": "PATIENT_CLUSTER_PAIRED",
        "bootstrap_count": BOOTSTRAPS,
        "zero_division": 0,
        "macro_average": "UNWEIGHTED_MEAN_OF_BOTH_CLASSES",
        "frozen_fold_thresholds_preserved": True,
        "training_performed": False,
        "inference_performed": False,
        "threshold_tuning_performed": False,
        "model_selection_or_freeze_performed": False,
        "ensemble_fitted_or_evaluated": False,
        "locked_test_used": False,
        "external_used": False,
    }
    (out / "DUAL_PR_POLICY.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8"
    )
    verify_source(root, args.source_lock)
    artifacts = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    receipt = {
        "schema": "pneumonia.phase2.dual_pr.operational.v1",
        "status": "PASS_DUAL_PR_OPERATIONAL",
        "created_utc": datetime.now(UTC).isoformat(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "git_commit": os.environ.get("GITHUB_SHA"),
        "source_files_verified_before_and_after": 40,
        "vectors_verified": 36,
        "patient_count": len(np.unique(anchor["patient_id"])),
        "image_count_per_vector": len(anchor["label"]),
        "dual_metric_rows": len(rows),
        "finalist_exploratory_rows": len(direct),
        "original_recall_rows_reconciled": reconcile_count,
        "descriptive_nondominated": [r for r in table if r["descriptive_nondominated"]],
        "reject_count_holm72": sum(r["reject_holm_dual_global_0_05"] for r in rows),
        "analysis_classification": CLASSIFICATION,
        "artifact_sha256": artifacts,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "locked_test_used": False,
        "external_used": False,
    }
    (out / "DUAL_PR_OPERATIONAL_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
