from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import m07_rsna_age_audit_v1 as base

N_FOLDS = 5
SEED_TAG = "m07-rsna-r224-crossfit-calibration-v1"


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def patient_fold(patient_id: str) -> int:
    digest = hashlib.sha256((SEED_TAG + "|" + patient_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % N_FOLDS


def confusion(rows: list[dict], threshold: float) -> dict:
    tn = fp = fn = tp = 0
    for row in rows:
        pred = int(float(row["score"]) >= threshold)
        y = int(row["label"])
        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        else:
            tp += 1
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    balanced = (sensitivity + specificity) / 2.0
    accuracy = (tn + tp) / len(rows)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "accuracy": accuracy,
    }


def candidate_thresholds(rows: list[dict]) -> list[float]:
    values = sorted({float(row["score"]) for row in rows})
    if not values:
        raise ValueError("empty threshold source")
    candidates = [values[0] - 1e-9]
    candidates.extend((a + b) / 2.0 for a, b in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-9)
    return candidates


def choose_balanced_threshold(rows: list[dict]) -> tuple[float, dict]:
    best = None
    for threshold in candidate_thresholds(rows):
        stats = confusion(rows, threshold)
        key = (
            stats["balanced_accuracy"],
            stats["sensitivity"],
            stats["specificity"],
            -abs(threshold),
        )
        if best is None or key > best[0]:
            best = (key, threshold, stats)
    assert best is not None
    return float(best[1]), best[2]


def choose_sens90_threshold(rows: list[dict]) -> tuple[float, dict]:
    eligible = []
    for threshold in candidate_thresholds(rows):
        stats = confusion(rows, threshold)
        if stats["sensitivity"] >= 0.90:
            eligible.append(
                (
                    stats["specificity"],
                    stats["balanced_accuracy"],
                    stats["sensitivity"],
                    -abs(threshold),
                    threshold,
                    stats,
                )
            )
    if not eligible:
        return choose_balanced_threshold(rows)
    best = max(eligible)
    return float(best[4]), best[5]


def patient_weights(rows: list[dict]) -> list[float]:
    counts = Counter(str(row["patient_id"]) for row in rows)
    return [1.0 / counts[str(row["patient_id"])] for row in rows]


def fit_platt(rows: list[dict], l2: float = 1e-3) -> dict:
    xs = [float(row["score"]) for row in rows]
    ys = [float(row["label"]) for row in rows]
    weights = patient_weights(rows)
    wsum = sum(weights)
    mean = sum(w * x for w, x in zip(weights, xs)) / wsum
    variance = sum(w * (x - mean) ** 2 for w, x in zip(weights, xs)) / wsum
    scale = math.sqrt(max(variance, 1e-12))
    standardized = [(x - mean) / scale for x in xs]
    prevalence = min(1 - 1e-6, max(1e-6, sum(w * y for w, y in zip(weights, ys)) / wsum))
    a = 1.0
    b = math.log(prevalence / (1.0 - prevalence))
    for _ in range(80):
        g_a = l2 * a
        g_b = 0.0
        h_aa = l2
        h_ab = 0.0
        h_bb = 1e-12
        for w, x, y in zip(weights, standardized, ys):
            p = sigmoid(a * x + b)
            err = p - y
            curv = max(1e-9, p * (1.0 - p))
            g_a += w * err * x
            g_b += w * err
            h_aa += w * curv * x * x
            h_ab += w * curv * x
            h_bb += w * curv
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-12:
            break
        delta_a = (g_a * h_bb - g_b * h_ab) / det
        delta_b = (g_b * h_aa - g_a * h_ab) / det
        a = max(1e-6, a - delta_a)
        b -= delta_b
        if max(abs(delta_a), abs(delta_b)) < 1e-8:
            break
    return {"a": a, "b": b, "mean": mean, "scale": scale, "l2": l2}


def platt_probability(score: float, params: dict) -> float:
    x = (float(score) - params["mean"]) / params["scale"]
    return sigmoid(params["a"] * x + params["b"])


def ece15(rows: list[dict]) -> float:
    total = len(rows)
    error = 0.0
    for bin_index in range(15):
        lo = bin_index / 15.0
        hi = (bin_index + 1) / 15.0
        bucket = [
            row
            for row in rows
            if lo <= float(row["prob"]) < hi or (bin_index == 14 and float(row["prob"]) == 1.0)
        ]
        if not bucket:
            continue
        confidence = sum(float(row["prob"]) for row in bucket) / len(bucket)
        observed = sum(int(row["label"]) for row in bucket) / len(bucket)
        error += len(bucket) / total * abs(confidence - observed)
    return error


def evaluated(rows: list[dict]) -> dict:
    result = base.metrics(rows)
    result["ece15"] = ece15(rows)
    return result


def apply_threshold(rows: list[dict], threshold: float) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        item["pred"] = int(float(row["score"]) >= threshold)
        output.append(item)
    return output


def apply_platt(rows: list[dict], params: dict) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        probability = platt_probability(float(row["score"]), params)
        item["prob"] = probability
        item["pred"] = int(probability >= 0.5)
        output.append(item)
    return output


def validate_folds(rows: list[dict]) -> dict:
    patients_by_fold = {fold: set() for fold in range(N_FOLDS)}
    row_counts = {}
    class_counts = {}
    for fold in range(N_FOLDS):
        subset = [row for row in rows if patient_fold(str(row["patient_id"])) == fold]
        patients_by_fold[fold] = {str(row["patient_id"]) for row in subset}
        row_counts[str(fold)] = len(subset)
        class_counts[str(fold)] = {
            "negative": sum(int(row["label"]) == 0 for row in subset),
            "positive": sum(int(row["label"]) == 1 for row in subset),
        }
        if not subset or min(class_counts[str(fold)].values()) == 0:
            raise RuntimeError("crossfit fold lacks a class")
    for left in range(N_FOLDS):
        for right in range(left + 1, N_FOLDS):
            if patients_by_fold[left] & patients_by_fold[right]:
                raise RuntimeError("patient leakage across crossfit folds")
    return {"row_counts": row_counts, "class_counts": class_counts}


def crossfit(rows: list[dict]) -> dict:
    fold_audit = validate_folds(rows)
    outputs = {
        "balanced_threshold": [],
        "sensitivity90_threshold": [],
        "platt": [],
    }
    fold_receipts = []
    for fold in range(N_FOLDS):
        train = [row for row in rows if patient_fold(str(row["patient_id"])) != fold]
        test = [row for row in rows if patient_fold(str(row["patient_id"])) == fold]
        balanced_threshold, train_balanced = choose_balanced_threshold(train)
        sens90_threshold, train_sens90 = choose_sens90_threshold(train)
        platt = fit_platt(train)
        outputs["balanced_threshold"].extend(apply_threshold(test, balanced_threshold))
        outputs["sensitivity90_threshold"].extend(apply_threshold(test, sens90_threshold))
        outputs["platt"].extend(apply_platt(test, platt))
        fold_receipts.append(
            {
                "fold": fold,
                "train_patients": len({str(r["patient_id"]) for r in train}),
                "test_patients": len({str(r["patient_id"]) for r in test}),
                "balanced_threshold": balanced_threshold,
                "balanced_threshold_train_metrics": train_balanced,
                "sensitivity90_threshold": sens90_threshold,
                "sensitivity90_threshold_train_metrics": train_sens90,
                "platt": platt,
            }
        )
    patient_ids = [str(row["patient_id"]) for row in rows]
    if len({id(row) for method in outputs.values() for row in method}) != sum(
        len(method) for method in outputs.values()
    ):
        raise RuntimeError("unexpected row object reuse")
    baseline = evaluated([dict(row) for row in rows])
    methods = {name: evaluated(method_rows) for name, method_rows in outputs.items()}
    for name, method_rows in outputs.items():
        if sorted(str(row["patient_id"]) for row in method_rows) != sorted(patient_ids):
            raise RuntimeError(f"{name} did not produce one crossfit prediction per source row")
    return {
        "schema": "m07.external.rsna.r224.crossfit_calibration.v1",
        "status": "PASS_CROSSFIT_EXTERNAL_ADAPTATION_ANALYSIS",
        "source_kernel": base.DEFAULT_KERNEL,
        "source_resolution": 224,
        "n_folds": N_FOLDS,
        "split_unit": "PATIENT",
        "patient_leakage": False,
        "model_weights_changed": False,
        "training_performed": False,
        "raw_external_result_preserved": True,
        "analysis_is_pure_external_validation": False,
        "analysis_classification": "EXTERNAL_CROSSFIT_CALIBRATION_ADAPTATION_ANALYSIS",
        "fold_audit": fold_audit,
        "baseline_frozen": baseline,
        "methods": methods,
        "fold_receipts": fold_receipts,
        "improvements_pp": {
            name: {
                "accuracy": 100.0 * (metrics["accuracy"] - baseline["accuracy"]),
                "balanced_accuracy": 100.0
                * (metrics["balanced_accuracy"] - baseline["balanced_accuracy"]),
                "sensitivity": 100.0 * (metrics["sensitivity"] - baseline["sensitivity"]),
                "specificity": 100.0 * (metrics["specificity"] - baseline["specificity"]),
                "macro_f1": 100.0 * (metrics["macro_f1"] - baseline["macro_f1"]),
                "brier": 100.0 * (metrics["brier_mean_probability"] - baseline["brier_mean_probability"]),
                "ece15": 100.0 * (metrics["ece15"] - baseline["ece15"]),
            }
            for name, metrics in methods.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise SystemExit("CGP_READ_OIDC_TOKEN unavailable")
    url = base.find_prediction_url(base.DEFAULT_ACCOUNT, base.DEFAULT_KERNEL, token)
    rows = base.download_rows(url)
    report = crossfit(rows)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M07_RSNA_CROSSFIT_CALIBRATION=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
