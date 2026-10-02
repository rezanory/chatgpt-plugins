from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Iterable

import numpy as np
import pandas as pd
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


class PairedStatisticsError(RuntimeError):
    pass


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(tuple(p_values), dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Holm adjustment requires finite p-values in [0,1].")
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def paired_metric_vector(y, pred, score) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) == 0 or not (len(y) == len(pred) == len(score)):
        raise ValueError("Metric vector arrays must have equal nonzero length.")
    if len(np.unique(y)) < 2:
        raise ValueError("Metric vector requires both classes.")
    if not np.isfinite(score).all():
        raise ValueError("Metric score contains non-finite values.")
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
    return {
        "balanced_accuracy": float((r0 + r1) / 2.0),
        "macro_f1": float((f10 + f11) / 2.0),
        "mcc": mcc,
        "auroc": float(roc_auc_score(y, score)),
        "recall_normal": r0,
        "recall_pneumonia": r1,
    }


def paired_patient_bootstrap_metrics(
    frame: pd.DataFrame,
    pred_reference,
    score_reference,
    pred_candidate,
    score_candidate,
    metrics,
    *,
    n_boot: int = PAIRED_BOOTSTRAPS,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> list[dict]:
    required = {"patient_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Paired bootstrap requires patient_id and label.")
    if frame["patient_id"].isna().any():
        raise ValueError("Paired bootstrap requires non-missing patient IDs.")
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    if pd.Series(patient_ids).str.strip().eq("").any():
        raise ValueError("Paired bootstrap requires non-empty patient IDs.")
    y = frame["label"].to_numpy(dtype=int)
    pred_reference = np.asarray(pred_reference, dtype=int)
    pred_candidate = np.asarray(pred_candidate, dtype=int)
    score_reference = np.asarray(score_reference, dtype=float)
    score_candidate = np.asarray(score_candidate, dtype=float)
    if not (
        len(y)
        == len(pred_reference)
        == len(pred_candidate)
        == len(score_reference)
        == len(score_candidate)
        == len(patient_ids)
    ):
        raise ValueError("Paired arrays must have identical length.")

    metrics = tuple(metrics)
    allowed = set(PAIRED_SECONDARY_METRICS) | {PAIRED_PRIMARY_METRIC}
    unknown = set(metrics) - allowed
    if unknown:
        raise ValueError(f"Unknown paired metrics: {sorted(unknown)}")

    point_ref = paired_metric_vector(y, pred_reference, score_reference)
    point_cand = paired_metric_vector(y, pred_candidate, score_candidate)
    delta_store = {metric: [] for metric in metrics}
    rng = np.random.default_rng(int(seed))
    unique_patients = np.unique(patient_ids)
    patient_rows: dict[str, list[int]] = {}
    for row_index, patient_id in enumerate(patient_ids):
        patient_rows.setdefault(patient_id, []).append(row_index)
    patient_arrays = {
        patient_id: np.asarray(rows, dtype=np.intp)
        for patient_id, rows in patient_rows.items()
    }

    for _ in range(int(n_boot)):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([patient_arrays[patient_id] for patient_id in sampled])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        ref_vec = paired_metric_vector(yy, pred_reference[idx], score_reference[idx])
        cand_vec = paired_metric_vector(yy, pred_candidate[idx], score_candidate[idx])
        for metric in metrics:
            delta_store[metric].append(cand_vec[metric] - ref_vec[metric])

    valid = min(len(values) for values in delta_store.values())
    if valid < max(100, int(n_boot) // 2):
        raise RuntimeError("Too few valid paired patient-bootstrap replicates.")

    rows = []
    for metric in metrics:
        deltas = np.asarray(delta_store[metric], dtype=float)
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
                "n_patients": int(len(unique_patients)),
            }
        )
    return rows


def prediction_identity(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"relative_path", "patient_id", "label", "sha256"}
    if not required.issubset(frame.columns):
        raise PairedStatisticsError(
            "PREDICTION_IDENTITY_COLUMNS_MISSING="
            + ",".join(sorted(required - set(frame.columns)))
        )
    return (
        frame[["relative_path", "patient_id", "label", "sha256"]]
        .astype(
            {
                "relative_path": str,
                "patient_id": str,
                "label": int,
                "sha256": str,
            }
        )
        .reset_index(drop=True)
    )


def load_prediction(pred_root: pathlib.Path, model: str, resolution: int, dataset: str) -> pd.DataFrame:
    dataset = dataset.upper()
    path = pred_root / model / f"R{int(resolution)}" / f"{dataset}_PREDICTIONS.csv"
    if not path.is_file():
        raise PairedStatisticsError(f"PREDICTION_FILE_MISSING:{path}")
    frame = pd.read_csv(
        path,
        float_precision="round_trip",
        dtype={"patient_id": str, "sha256": str, "relative_path": str},
    )
    prediction_identity(frame)
    if dataset == "OOF":
        required = {"prediction_fold_threshold", "probability_pneumonia"}
    elif dataset == "LOCKED_TEST":
        if "prediction_primary" not in frame.columns:
            if "prediction_primary_normalized" not in frame.columns:
                raise PairedStatisticsError(f"PRIMARY_PREDICTION_COLUMN_MISSING:{path}")
            frame["prediction_primary"] = pd.to_numeric(
                frame["prediction_primary_normalized"], errors="raise"
            ).astype(int)
        elif "prediction_primary_normalized" in frame.columns:
            primary = pd.to_numeric(frame["prediction_primary"], errors="raise").astype(int)
            alias = pd.to_numeric(
                frame["prediction_primary_normalized"], errors="raise"
            ).astype(int)
            if not np.array_equal(primary.to_numpy(), alias.to_numpy()):
                raise PairedStatisticsError(f"PRIMARY_PREDICTION_ALIAS_MISMATCH:{path}")
            frame["prediction_primary"] = primary
        required = {"prediction_primary", "normalized_ensemble_score"}
    else:
        raise PairedStatisticsError(f"UNSUPPORTED_DATASET:{dataset}")
    missing = required - set(frame.columns)
    if missing:
        raise PairedStatisticsError(
            f"PREDICTION_COLUMNS_MISSING:{path}:{','.join(sorted(missing))}"
        )
    return frame


def aligned_pair(
    pred_root: pathlib.Path,
    reference: str,
    candidate: str,
    resolution: int,
    dataset: str,
):
    a = load_prediction(pred_root, reference, resolution, dataset)
    b = load_prediction(pred_root, candidate, resolution, dataset)
    identity = prediction_identity(a)
    if not identity.equals(prediction_identity(b)):
        raise PairedStatisticsError(
            f"PAIRED_IDENTITY_MISMATCH:{reference}:{candidate}:R{resolution}:{dataset}"
        )
    if dataset == "OOF":
        return (
            identity,
            a["prediction_fold_threshold"].to_numpy(dtype=int),
            a["probability_pneumonia"].to_numpy(dtype=float),
            b["prediction_fold_threshold"].to_numpy(dtype=int),
            b["probability_pneumonia"].to_numpy(dtype=float),
        )
    return (
        identity,
        a["prediction_primary"].to_numpy(dtype=int),
        a["normalized_ensemble_score"].to_numpy(dtype=float),
        b["prediction_primary"].to_numpy(dtype=int),
        b["normalized_ensemble_score"].to_numpy(dtype=float),
    )


def apply_holm(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["p_holm_primary_within_resolution"] = np.nan
    primary = table["metric"] == PAIRED_PRIMARY_METRIC
    for (_, _), idx in table[primary].groupby(["dataset", "resolution"]).groups.items():
        idx = list(idx)
        table.loc[idx, "p_holm_primary_within_resolution"] = holm_adjust(
            table.loc[idx, "p_raw_two_sided"].to_numpy(dtype=float)
        )

    table["p_holm_primary_global"] = np.nan
    for _, idx in table[primary].groupby("dataset").groups.items():
        idx = list(idx)
        table.loc[idx, "p_holm_primary_global"] = holm_adjust(
            table.loc[idx, "p_raw_two_sided"].to_numpy(dtype=float)
        )
    table["reject_holm_primary_global_0_05"] = (
        table["p_holm_primary_global"] < PAIRED_ALPHA
    ).fillna(False)

    table["p_holm_metric_global"] = np.nan
    for (_, _), idx in table.groupby(["dataset", "metric"]).groups.items():
        idx = list(idx)
        table.loc[idx, "p_holm_metric_global"] = holm_adjust(
            table.loc[idx, "p_raw_two_sided"].to_numpy(dtype=float)
        )
    table["reject_holm_metric_global_0_05"] = (
        table["p_holm_metric_global"] < PAIRED_ALPHA
    )
    return table


def build_statistics(pred_root: pathlib.Path, n_boot: int = PAIRED_BOOTSTRAPS) -> pd.DataFrame:
    metrics = (PAIRED_PRIMARY_METRIC,) + tuple(PAIRED_SECONDARY_METRICS)
    rows = []
    for dataset in ("OOF", "LOCKED_TEST"):
        for resolution in EXPECTED_RESOLUTIONS:
            for reference, candidate, contrast in PREDECLARED_MODEL_COMPARISONS:
                frame, pr, sr, pc, sc = aligned_pair(
                    pred_root, reference, candidate, resolution, dataset
                )
                result_rows = paired_patient_bootstrap_metrics(
                    frame,
                    pr,
                    sr,
                    pc,
                    sc,
                    metrics=metrics,
                    n_boot=n_boot,
                    seed=PAIRED_BOOTSTRAP_SEED
                    + int(resolution)
                    + sum(map(ord, reference + candidate)),
                )
                for result in result_rows:
                    rows.append(
                        {
                            "dataset": dataset,
                            "resolution": int(resolution),
                            "reference_model": reference,
                            "candidate_model": candidate,
                            "contrast": contrast,
                            **result,
                        }
                    )
    table = apply_holm(pd.DataFrame(rows))
    expected_rows = (
        2
        * len(EXPECTED_RESOLUTIONS)
        * len(PREDECLARED_MODEL_COMPARISONS)
        * (1 + len(PAIRED_SECONDARY_METRICS))
    )
    if len(table) != expected_rows:
        raise PairedStatisticsError(
            f"PAIRED_STATISTICS_ROW_COUNT_INVALID:{len(table)}:{expected_rows}"
        )
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstraps", type=int, default=PAIRED_BOOTSTRAPS)
    args = parser.parse_args()
    if args.bootstraps < 100:
        raise SystemExit("BOOTSTRAP_COUNT_TOO_LOW")

    pred_root = pathlib.Path(args.pred_root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_statistics(pred_root, n_boot=args.bootstraps)
    csv_path = out_dir / "MASTER_PAIRED_PATIENT_BOOTSTRAP_HOLM.csv"
    table.to_csv(csv_path, index=False, float_format="%.17g")

    policy = {
        "schema": "pneumonia.master.paired.statistics.v1.7",
        "primary_metric": PAIRED_PRIMARY_METRIC,
        "secondary_metrics": list(PAIRED_SECONDARY_METRICS),
        "paired_bootstraps": int(args.bootstraps),
        "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        "bootstrap_unit": "PATIENT_CLUSTER_PAIRED",
        "multiple_comparison_correction": "HOLM",
        "multiplicity_policy": (
            "CONFIRMATORY_PRIMARY_BALACC_GLOBAL_HOLM_PER_DATASET; "
            "SECONDARY_HOLM_SEPARATELY_WITHIN_EACH_METRIC_FAMILY_PER_DATASET; "
            "NO_SINGLE_216_TEST_ALL_METRICS_FAMILY"
        ),
        "predeclared_comparisons": [list(item) for item in PREDECLARED_MODEL_COMPARISONS],
        "selection_dataset": "OOF_ONLY",
        "locked_test_role": "CONFIRMATORY_REPORT_ONLY_NO_MODEL_OR_THRESHOLD_TUNING",
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "p_value_method": "APPROXIMATE_TWO_SIDED_BOOTSTRAP_TAIL_NOT_EXACT_PERMUTATION",
        "estimator_changed_from_v1_6": False,
    }
    (out_dir / "MASTER_STATISTICAL_COMPARISON_POLICY.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = {
        "schema": "pneumonia.phase2.master_paired_statistics.execution.v1",
        "status": "PASS",
        "rows": int(len(table)),
        "oof_rows": int((table["dataset"] == "OOF").sum()),
        "locked_test_rows": int((table["dataset"] == "LOCKED_TEST").sum()),
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "selection_dataset": "OOF_ONLY",
        "statistical_csv": csv_path.name,
        "policy_json": "MASTER_STATISTICAL_COMPARISON_POLICY.json",
    }
    (out_dir / "MASTER_PAIRED_STATISTICS_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("PHASE2_MASTER_PAIRED_STATISTICS_PASS=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
