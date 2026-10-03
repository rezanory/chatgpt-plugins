from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import pathlib
import zlib
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

PAIRED_BOOTSTRAPS = 5000
PAIRED_BOOTSTRAP_SEED = 260915
PAIRED_PRIMARY_METRIC = "balanced_accuracy"
PAIRED_SECONDARY_METRICS = (
    "macro_f1",
    "mcc",
    "auroc",
    "recall_normal",
    "recall_pneumonia",
)
PAIRED_ALPHA = 0.05
EXPECTED_RESOLUTIONS = (224, 320, 384)
PREDECLARED_MODEL_COMPARISONS = (
    ("M01", "M02", "BCE_vs_weighted_BCE_sampling_bundle"),
    ("M03", "M04", "static_focal_vs_FLSD53"),
    ("M04", "M05", "CBAM_added_to_M04"),
    ("M04", "M06", "EdgeBlock_added_to_M04"),
    ("M04", "M07", "EdgeBlock_plus_CBAM_added_to_M04"),
    ("M05", "M07", "EdgeBlock_added_to_CBAM_branch"),
    ("M06", "M07", "CBAM_added_to_Edge_branch"),
    ("M07", "M08", "MixUp_added_to_M07"),
    ("M07", "M09", "CutMix_added_to_M07"),
    ("M07", "M10", "stochastic_MixUp_CutMix_added_to_M07"),
    ("M10", "M11", "EMA_added_to_M10"),
    ("M10", "M12", "SWA_added_to_M10"),
)


class PairedOofError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_vector(path: pathlib.Path) -> dict[str, Any]:
    outer = json.loads(path.read_text(encoding="utf-8"))
    if outer.get("schema") != "pneumonia.phase2.oof.vector.v1":
        raise PairedOofError(f"VECTOR_SCHEMA_INVALID:{path.name}")
    if outer.get("status") != "PASS":
        raise PairedOofError(f"VECTOR_STATUS_INVALID:{path.name}")
    if outer.get("payload_encoding") != "zlib+base64+json":
        raise PairedOofError(f"VECTOR_ENCODING_INVALID:{path.name}")
    raw = zlib.decompress(base64.b64decode(str(outer.get("payload_b64") or "")))
    if sha256_bytes(raw) != str(outer.get("payload_sha256") or ""):
        raise PairedOofError(f"VECTOR_PAYLOAD_SHA_MISMATCH:{path.name}")
    inner = json.loads(raw.decode("utf-8"))
    required = {"patient_id", "label", "prediction", "score"}
    if not required.issubset(inner):
        raise PairedOofError(f"VECTOR_FIELDS_MISSING:{path.name}")
    lengths = {len(inner[key]) for key in required}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise PairedOofError(f"VECTOR_LENGTH_INVALID:{path.name}")
    n = next(iter(lengths))
    if int(outer.get("row_count") or 0) != n:
        raise PairedOofError(f"VECTOR_ROW_COUNT_MISMATCH:{path.name}")

    patient = np.asarray([str(x) for x in inner["patient_id"]], dtype=object)
    y = np.asarray(inner["label"], dtype=int)
    pred = np.asarray(inner["prediction"], dtype=int)
    score = np.asarray(inner["score"], dtype=float)
    if any(not str(x).strip() for x in patient):
        raise PairedOofError(f"VECTOR_PATIENT_EMPTY:{path.name}")
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise PairedOofError(f"VECTOR_BINARY_INVALID:{path.name}")
    if not np.isfinite(score).all():
        raise PairedOofError(f"VECTOR_SCORE_NONFINITE:{path.name}")
    if len(np.unique(y)) != 2:
        raise PairedOofError(f"VECTOR_LABEL_CLASSES_INVALID:{path.name}")

    return {
        "model_id": str(outer["model_id"]),
        "resolution": int(outer["resolution"]),
        "row_count": n,
        "identity_sequence_sha256": str(outer["identity_sequence_sha256"]),
        "patient_id": patient,
        "label": y,
        "prediction": pred,
        "score": score,
        "source_dataset_ref": outer.get("source_dataset_ref"),
        "source_dataset_version": outer.get("source_dataset_version"),
        "source_archive_sha256": outer.get("source_archive_sha256"),
        "source_oof_sha256": outer.get("source_oof_sha256"),
        "locked_test_used_for_selection": outer.get("locked_test_used_for_selection"),
        "external_used_for_selection": outer.get("external_used_for_selection"),
        "vector_file_sha256": sha256_file(path),
    }


def _paired_metric_vector(y, pred, score):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) == 0 or not (len(y) == len(pred) == len(score)):
        raise PairedOofError("METRIC_VECTOR_LENGTH_INVALID")
    if len(np.unique(y)) < 2:
        raise PairedOofError("METRIC_VECTOR_REQUIRES_TWO_CLASSES")
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))

    def safe(num, den):
        return float(num / den) if den else 0.0

    r0 = safe(tn, tn + fp)
    r1 = safe(tp, tp + fn)
    p0 = safe(tn, tn + fn)
    p1 = safe(tp, tp + fp)
    f10 = safe(2.0 * p0 * r0, p0 + r0)
    f11 = safe(2.0 * p1 * r1, p1 + r1)
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float((tp * tn - fp * fn) / denom) if denom else 0.0
    auroc = float(roc_auc_score(y, score))
    return {
        "balanced_accuracy": float((r0 + r1) / 2.0),
        "macro_f1": float((f10 + f11) / 2.0),
        "mcc": mcc,
        "auroc": auroc,
        "recall_normal": r0,
        "recall_pneumonia": r1,
    }


def _safe_divide_batch(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def _patient_confusion_contributions(patient_index, y, pred, n_patients):
    patient_index = np.asarray(patient_index, dtype=np.intp)
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    masks = (
        (y == 0) & (pred == 0),
        (y == 0) & (pred == 1),
        (y == 1) & (pred == 0),
        (y == 1) & (pred == 1),
    )
    return np.stack(
        [
            np.bincount(
                patient_index,
                weights=mask.astype(np.float64),
                minlength=n_patients,
            )
            for mask in masks
        ],
        axis=1,
    )


def _auc_batch_plan(patient_index, y, score):
    patient_index = np.asarray(patient_index, dtype=np.intp)
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    group_starts = np.concatenate(
        (
            np.asarray([0], dtype=np.intp),
            np.flatnonzero(sorted_score[1:] != sorted_score[:-1]).astype(np.intp)
            + 1,
        )
    )
    return patient_index[order], y[order], group_starts


def _weighted_auc_batch(patient_counts, plan):
    patient_index_sorted, y_sorted, group_starts = plan
    row_weights = patient_counts[:, patient_index_sorted].astype(
        np.float64, copy=False
    )
    positive = row_weights * (y_sorted == 1)[None, :]
    negative = row_weights * (y_sorted == 0)[None, :]
    positive_by_group = np.add.reduceat(positive, group_starts, axis=1)
    negative_by_group = np.add.reduceat(negative, group_starts, axis=1)
    negative_before = np.cumsum(negative_by_group, axis=1) - negative_by_group
    numerator = np.sum(
        positive_by_group
        * (negative_before + 0.5 * negative_by_group),
        axis=1,
    )
    total_positive = np.sum(positive_by_group, axis=1)
    total_negative = np.sum(negative_by_group, axis=1)
    denominator = total_positive * total_negative
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(patient_counts), np.nan, dtype=float),
        where=denominator > 0,
    )


def _bootstrap_metric_batch(patient_counts, contributions, auc):
    cm = patient_counts @ contributions
    tn, fp, fn, tp = (cm[:, index] for index in range(4))
    recall_normal = _safe_divide_batch(tn, tn + fp)
    recall_pneumonia = _safe_divide_batch(tp, tp + fn)
    precision_normal = _safe_divide_batch(tn, tn + fn)
    precision_pneumonia = _safe_divide_batch(tp, tp + fp)
    f1_normal = _safe_divide_batch(
        2.0 * precision_normal * recall_normal,
        precision_normal + recall_normal,
    )
    f1_pneumonia = _safe_divide_batch(
        2.0 * precision_pneumonia * recall_pneumonia,
        precision_pneumonia + recall_pneumonia,
    )
    mcc_denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = _safe_divide_batch(tp * tn - fp * fn, mcc_denominator)
    return {
        "balanced_accuracy": (recall_normal + recall_pneumonia) / 2.0,
        "macro_f1": (f1_normal + f1_pneumonia) / 2.0,
        "mcc": mcc,
        "auroc": np.asarray(auc, dtype=float),
        "recall_normal": recall_normal,
        "recall_pneumonia": recall_pneumonia,
    }


def paired_patient_bootstrap_metrics(
    patient_ids,
    y,
    pred_reference,
    score_reference,
    pred_candidate,
    score_candidate,
    metrics,
    n_boot=PAIRED_BOOTSTRAPS,
    seed=PAIRED_BOOTSTRAP_SEED,
):
    patient_ids = np.asarray(patient_ids, dtype=object)
    y = np.asarray(y, dtype=int)
    pred_reference = np.asarray(pred_reference, dtype=int)
    pred_candidate = np.asarray(pred_candidate, dtype=int)
    score_reference = np.asarray(score_reference, dtype=float)
    score_candidate = np.asarray(score_candidate, dtype=float)
    if not (
        len(y)
        == len(patient_ids)
        == len(pred_reference)
        == len(pred_candidate)
        == len(score_reference)
        == len(score_candidate)
    ):
        raise PairedOofError("PAIRED_ARRAY_LENGTH_MISMATCH")

    metrics = tuple(metrics)
    known = set(PAIRED_SECONDARY_METRICS) | {PAIRED_PRIMARY_METRIC}
    unknown = set(metrics) - known
    if unknown:
        raise PairedOofError("PAIRED_UNKNOWN_METRICS:" + ",".join(sorted(unknown)))

    point_ref = _paired_metric_vector(y, pred_reference, score_reference)
    point_cand = _paired_metric_vector(y, pred_candidate, score_candidate)

    unique_patients, patient_index = np.unique(patient_ids, return_inverse=True)
    n_patients = len(unique_patients)
    if n_patients < 2:
        raise PairedOofError("PAIRED_TOO_FEW_PATIENTS")

    ref_contributions = _patient_confusion_contributions(
        patient_index, y, pred_reference, n_patients
    )
    cand_contributions = _patient_confusion_contributions(
        patient_index, y, pred_candidate, n_patients
    )
    class_contributions = np.stack(
        [
            np.bincount(
                patient_index,
                weights=(y == label).astype(np.float64),
                minlength=n_patients,
            )
            for label in (0, 1)
        ],
        axis=1,
    )
    ref_auc_plan = _auc_batch_plan(patient_index, y, score_reference)
    cand_auc_plan = _auc_batch_plan(patient_index, y, score_candidate)

    delta_chunks = {metric: [] for metric in metrics}
    rng = np.random.default_rng(int(seed))
    valid = 0
    remaining = int(n_boot)
    # Keep temporary sampling/row-weight matrices bounded while processing the
    # exact same patient-cluster bootstrap estimator in vectorized blocks.
    block_size = max(8, min(128, 2_000_000 // max(1, n_patients)))

    while remaining > 0:
        batch_size = min(block_size, remaining)
        sampled = rng.integers(
            0,
            n_patients,
            size=(batch_size, n_patients),
            dtype=np.intp,
        )
        offsets = (
            np.arange(batch_size, dtype=np.intp)[:, None] * n_patients
        )
        patient_counts = np.bincount(
            (sampled + offsets).reshape(-1),
            minlength=batch_size * n_patients,
        ).reshape(batch_size, n_patients)

        class_counts = patient_counts @ class_contributions
        valid_mask = (class_counts[:, 0] > 0) & (class_counts[:, 1] > 0)
        if np.any(valid_mask):
            valid_counts = patient_counts[valid_mask]
            ref_auc = _weighted_auc_batch(valid_counts, ref_auc_plan)
            cand_auc = _weighted_auc_batch(valid_counts, cand_auc_plan)
            if not np.isfinite(ref_auc).all() or not np.isfinite(cand_auc).all():
                raise PairedOofError("PAIRED_AUROC_NONFINITE")
            ref_metrics = _bootstrap_metric_batch(
                valid_counts, ref_contributions, ref_auc
            )
            cand_metrics = _bootstrap_metric_batch(
                valid_counts, cand_contributions, cand_auc
            )
            for metric in metrics:
                delta_chunks[metric].append(
                    np.asarray(cand_metrics[metric] - ref_metrics[metric])
                )
            valid += int(np.sum(valid_mask))
        remaining -= batch_size

    if valid < max(100, int(n_boot) // 2):
        raise PairedOofError("PAIRED_TOO_FEW_VALID_BOOTSTRAPS")

    rows = []
    for metric in metrics:
        if not delta_chunks[metric]:
            raise PairedOofError("PAIRED_METRIC_BOOTSTRAP_EMPTY:" + metric)
        deltas = np.concatenate(delta_chunks[metric]).astype(float, copy=False)
        left = (np.sum(deltas <= 0.0) + 1.0) / (len(deltas) + 1.0)
        right = (np.sum(deltas >= 0.0) + 1.0) / (len(deltas) + 1.0)
        rows.append(
            {
                "metric": metric,
                "reference": float(point_ref[metric]),
                "candidate": float(point_cand[metric]),
                "delta_candidate_minus_reference": float(
                    point_cand[metric] - point_ref[metric]
                ),
                "ci95_low": float(np.quantile(deltas, 0.025)),
                "ci95_high": float(np.quantile(deltas, 0.975)),
                "p_raw_two_sided": float(min(1.0, 2.0 * min(left, right))),
                "bootstrap_unit": "PATIENT_CLUSTER_PAIRED",
                "n_bootstrap_valid": int(len(deltas)),
                "n_patients": int(n_patients),
            }
        )
    return rows

def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise PairedOofError("HOLM_INPUT_INVALID")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise PairedOofError("HOLM_PVALUE_INVALID")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def load_vectors(vector_dir: pathlib.Path) -> dict[tuple[str, int], dict[str, Any]]:
    files = sorted(vector_dir.glob("PHASE2_OOF_VECTOR_*.json"))
    vectors = {}
    for path in files:
        vector = decode_vector(path)
        key = (vector["model_id"], vector["resolution"])
        if key in vectors:
            raise PairedOofError(f"VECTOR_DUPLICATE:{key}")
        if vector["locked_test_used_for_selection"] not in (None, False):
            raise PairedOofError(f"VECTOR_LOCKED_SELECTION_VIOLATION:{key}")
        if vector["external_used_for_selection"] not in (None, False):
            raise PairedOofError(f"VECTOR_EXTERNAL_SELECTION_VIOLATION:{key}")
        vectors[key] = vector

    needed = {
        (model, resolution)
        for comparison in PREDECLARED_MODEL_COMPARISONS
        for model in comparison[:2]
        for resolution in EXPECTED_RESOLUTIONS
    }
    missing = needed - set(vectors)
    if missing:
        raise PairedOofError("VECTOR_COVERAGE_MISSING:" + json.dumps(sorted(missing)))
    return vectors


def build_statistics(vectors: dict[tuple[str, int], dict[str, Any]]):
    metrics = (PAIRED_PRIMARY_METRIC,) + PAIRED_SECONDARY_METRICS
    rows = []
    for resolution in EXPECTED_RESOLUTIONS:
        for reference, candidate, contrast in PREDECLARED_MODEL_COMPARISONS:
            ref = vectors[(reference, resolution)]
            cand = vectors[(candidate, resolution)]
            if ref["identity_sequence_sha256"] != cand["identity_sequence_sha256"]:
                raise PairedOofError(
                    f"OOF_IDENTITY_MISMATCH:{reference}:{candidate}:R{resolution}"
                )
            if ref["row_count"] != cand["row_count"]:
                raise PairedOofError(
                    f"OOF_ROW_COUNT_MISMATCH:{reference}:{candidate}:R{resolution}"
                )
            if not np.array_equal(ref["label"], cand["label"]):
                raise PairedOofError(
                    f"OOF_LABEL_MISMATCH:{reference}:{candidate}:R{resolution}"
                )
            if not np.array_equal(ref["patient_id"], cand["patient_id"]):
                raise PairedOofError(
                    f"OOF_PATIENT_ORDER_MISMATCH:{reference}:{candidate}:R{resolution}"
                )

            results = paired_patient_bootstrap_metrics(
                ref["patient_id"],
                ref["label"],
                ref["prediction"],
                ref["score"],
                cand["prediction"],
                cand["score"],
                metrics,
                n_boot=PAIRED_BOOTSTRAPS,
                seed=(
                    PAIRED_BOOTSTRAP_SEED
                    + resolution
                    + sum(map(ord, reference + candidate))
                ),
            )
            for result in results:
                rows.append(
                    {
                        "dataset": "OOF",
                        "resolution": resolution,
                        "reference_model": reference,
                        "candidate_model": candidate,
                        "contrast": contrast,
                        **result,
                    }
                )

    primary_indexes = [
        i for i, row in enumerate(rows) if row["metric"] == PAIRED_PRIMARY_METRIC
    ]
    for resolution in EXPECTED_RESOLUTIONS:
        indexes = [
            i
            for i in primary_indexes
            if rows[i]["resolution"] == resolution
        ]
        adjusted = holm_adjust([rows[i]["p_raw_two_sided"] for i in indexes])
        for i, p_adj in zip(indexes, adjusted):
            rows[i]["p_holm_primary_within_resolution"] = float(p_adj)

    primary_adjusted = holm_adjust(
        [rows[i]["p_raw_two_sided"] for i in primary_indexes]
    )
    for i, p_adj in zip(primary_indexes, primary_adjusted):
        rows[i]["p_holm_primary_global"] = float(p_adj)
        rows[i]["reject_holm_primary_global_0_05"] = bool(p_adj < PAIRED_ALPHA)

    for metric in metrics:
        indexes = [i for i, row in enumerate(rows) if row["metric"] == metric]
        adjusted = holm_adjust([rows[i]["p_raw_two_sided"] for i in indexes])
        for i, p_adj in zip(indexes, adjusted):
            rows[i]["p_holm_metric_global"] = float(p_adj)
            rows[i]["reject_holm_metric_global_0_05"] = bool(p_adj < PAIRED_ALPHA)

    for row in rows:
        row.setdefault("p_holm_primary_global", None)
        row.setdefault("reject_holm_primary_global_0_05", False)
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    vector_dir = pathlib.Path(args.vector_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors = load_vectors(vector_dir)
    rows = build_statistics(vectors)
    expected_rows = (
        len(PREDECLARED_MODEL_COMPARISONS)
        * len(EXPECTED_RESOLUTIONS)
        * (1 + len(PAIRED_SECONDARY_METRICS))
    )
    if len(rows) != expected_rows:
        raise PairedOofError(f"PAIRED_ROW_COUNT_INVALID:{len(rows)}:{expected_rows}")

    stats_path = output_dir / "PHASE2_OOF_PAIRED_PATIENT_BOOTSTRAP_HOLM_V1.csv"
    write_csv(stats_path, rows)
    policy_path = output_dir / "PHASE2_OOF_PAIRED_POLICY_V1.json"
    policy = {
        "schema": "pneumonia.phase2.oof.paired.policy.v1",
        "status": "PASS_POLICY",
        "dataset": "OOF",
        "primary_metric": PAIRED_PRIMARY_METRIC,
        "secondary_metrics": list(PAIRED_SECONDARY_METRICS),
        "paired_bootstraps": PAIRED_BOOTSTRAPS,
        "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        "bootstrap_unit": "PATIENT_CLUSTER_PAIRED",
        "multiple_comparison_correction": "HOLM",
        "multiplicity_policy": (
            "CONFIRMATORY_PRIMARY_BALACC_GLOBAL_HOLM_PER_DATASET; "
            "SECONDARY_HOLM_SEPARATELY_WITHIN_EACH_METRIC_FAMILY_PER_DATASET; "
            "NO_SINGLE_216_TEST_ALL_METRICS_FAMILY"
        ),
        "p_value_method": "APPROXIMATE_TWO_SIDED_BOOTSTRAP_TAIL_NOT_EXACT_PERMUTATION",
        "estimator_changed_from_v1_6": False,
        "predeclared_comparisons": [list(x) for x in PREDECLARED_MODEL_COMPARISONS],
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "selection_performed": False,
    }
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8"
    )

    receipt = {
        "schema": "pneumonia.phase2.oof.paired.statistics.v1",
        "status": "PASS_PAIRED_OOF_STATISTICS",
        "dataset": "OOF",
        "statistical_rows": len(rows),
        "comparison_count": len(PREDECLARED_MODEL_COMPARISONS),
        "resolution_count": len(EXPECTED_RESOLUTIONS),
        "metric_count": 1 + len(PAIRED_SECONDARY_METRICS),
        "paired_bootstraps": PAIRED_BOOTSTRAPS,
        "primary_metric": PAIRED_PRIMARY_METRIC,
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "selection_performed": False,
        "artifact_sha256": {
            stats_path.name: sha256_file(stats_path),
            policy_path.name: sha256_file(policy_path),
        },
    }
    receipt_path = output_dir / "PHASE2_OOF_PAIRED_STATISTICS_RECEIPT_V1.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "PHASE2_OOF_PAIRED_STATS_PASS "
        + json.dumps(
            {
                "rows": len(rows),
                "comparisons": len(PREDECLARED_MODEL_COMPARISONS),
                "resolutions": len(EXPECTED_RESOLUTIONS),
                "bootstraps": PAIRED_BOOTSTRAPS,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
