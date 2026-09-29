from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import urllib.request
from collections import defaultdict

import m07_rsna_age_audit_v1 as base
from m07_rsna_resolution_reconcile_v1 import RUNS, output_inventory, safe_output_url

PREDICTIONS = "M07_RSNA_PEDIATRIC_EXTERNAL_PREDICTIONS.csv"
RESOLUTIONS = (224, 320, 384)
WEIGHT_STEP = 0.10
N_FOLDS = 5
SEED_TAG = "m07-rsna-resolution-ensemble-v1"


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def download_prediction_rows(url: str) -> list[dict]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "m07-rsna-resolution-ensemble-v1/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        text = response.read().decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    required = {
        "image_id",
        "patient_id",
        "age",
        "label",
        "normalized_ensemble_score",
        "prediction_primary_normalized",
    }
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        missing = sorted(required - set(reader.fieldnames or []))
        raise RuntimeError("prediction CSV schema mismatch: " + ",".join(missing))
    rows = []
    for raw in reader:
        rows.append(
            {
                "image_id": str(raw["image_id"]),
                "patient_id": str(raw["patient_id"]),
                "age": int(float(raw["age"])),
                "label": int(float(raw["label"])),
                "score": float(raw["normalized_ensemble_score"]),
                "pred": int(float(raw["prediction_primary_normalized"])),
            }
        )
    if len(rows) != 1099:
        raise RuntimeError(f"expected 1099 prediction rows, found {len(rows)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate image_id in prediction CSV")
    return rows


def fetch_all(token: str) -> dict[int, list[dict]]:
    outputs: dict[int, list[dict]] = {}
    for resolution in RESOLUTIONS:
        spec = RUNS[resolution]
        files = output_inventory(spec["account_id"], spec["kernel_ref"], token)
        url = safe_output_url(files, PREDICTIONS)
        outputs[resolution] = download_prediction_rows(url)
    return outputs


def align(outputs: dict[int, list[dict]]) -> list[dict]:
    indexed = {
        resolution: {row["image_id"]: row for row in rows}
        for resolution, rows in outputs.items()
    }
    ids = set(indexed[224])
    if any(set(indexed[r]) != ids for r in RESOLUTIONS):
        raise RuntimeError("resolution prediction image sets differ")
    merged = []
    for image_id in sorted(ids):
        refs = [indexed[r][image_id] for r in RESOLUTIONS]
        identity = {
            (row["patient_id"], row["age"], row["label"])
            for row in refs
        }
        if len(identity) != 1:
            raise RuntimeError(f"identity/label mismatch for {image_id}")
        patient_id, age, label = next(iter(identity))
        merged.append(
            {
                "image_id": image_id,
                "patient_id": patient_id,
                "age": age,
                "label": label,
                "score_224": refs[0]["score"],
                "score_320": refs[1]["score"],
                "score_384": refs[2]["score"],
                "pred_224": refs[0]["pred"],
                "pred_320": refs[1]["pred"],
                "pred_384": refs[2]["pred"],
            }
        )
    return merged


def to_metric_rows(rows: list[dict], *, score_key: str, pred_key: str) -> list[dict]:
    output = []
    for row in rows:
        score = float(row[score_key])
        output.append(
            {
                "patient_id": row["patient_id"],
                "age": row["age"],
                "label": row["label"],
                "score": score,
                "pred": int(row[pred_key]),
                "prob": sigmoid(score),
            }
        )
    return output


def metric_summary(rows: list[dict]) -> dict:
    metrics = base.metrics(rows)
    return {
        key: metrics[key]
        for key in (
            "n",
            "patients",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision_positive",
            "f1_positive",
            "macro_f1",
            "auroc",
            "auprc_positive",
            "tn",
            "fp",
            "fn",
            "tp",
        )
    }


def add_label_free_ensembles(rows: list[dict]) -> None:
    for row in rows:
        scores = [row["score_224"], row["score_320"], row["score_384"]]
        row["score_equal"] = sum(scores) / 3.0
        row["pred_equal"] = int(row["score_equal"] >= 0.0)
        votes = row["pred_224"] + row["pred_320"] + row["pred_384"]
        row["score_majority"] = float(votes)
        row["pred_majority"] = int(votes >= 2)


def fp_sets(rows: list[dict]) -> dict:
    sets = {}
    for resolution in RESOLUTIONS:
        sets[resolution] = {
            row["image_id"]
            for row in rows
            if row["label"] == 0 and row[f"pred_{resolution}"] == 1
        }
    all_three = sets[224] & sets[320] & sets[384]
    any_fp = sets[224] | sets[320] | sets[384]
    unique = {
        str(r): len(sets[r] - set().union(*(sets[x] for x in RESOLUTIONS if x != r)))
        for r in RESOLUTIONS
    }
    return {
        "fp_counts": {str(r): len(sets[r]) for r in RESOLUTIONS},
        "fp_all_three_overlap": len(all_three),
        "fp_any_resolution": len(any_fp),
        "fp_unique_to_resolution": unique,
        "r224_fp_rescued_by_equal_ensemble": sum(
            row["label"] == 0
            and row["pred_224"] == 1
            and row["pred_equal"] == 0
            for row in rows
        ),
        "r224_tp_lost_by_equal_ensemble": sum(
            row["label"] == 1
            and row["pred_224"] == 1
            and row["pred_equal"] == 0
            for row in rows
        ),
    }


def patient_fold(patient_id: str) -> int:
    import hashlib

    digest = hashlib.sha256((SEED_TAG + "|" + patient_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % N_FOLDS


def weight_grid() -> list[tuple[float, float, float]]:
    steps = int(round(1.0 / WEIGHT_STEP))
    out = []
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            c = steps - a - b
            out.append((a / steps, b / steps, c / steps))
    return out


def scored(rows: list[dict], weights: tuple[float, float, float]) -> list[dict]:
    output = []
    for row in rows:
        score = (
            weights[0] * row["score_224"]
            + weights[1] * row["score_320"]
            + weights[2] * row["score_384"]
        )
        output.append(
            {
                "patient_id": row["patient_id"],
                "age": row["age"],
                "label": row["label"],
                "score": float(score),
                "pred": 0,
                "prob": sigmoid(score),
            }
        )
    return output


def choose_precision_sens90(
    rows: list[dict], weights: tuple[float, float, float]
) -> tuple[float, dict] | None:
    metric_rows = scored(rows, weights)
    ordered = sorted(
        metric_rows,
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    positives = sum(int(row["label"]) == 1 for row in ordered)
    negatives = len(ordered) - positives
    if positives == 0 or negatives == 0:
        return None

    tp = fp = 0
    best = None
    index = 0
    while index < len(ordered):
        score = float(ordered[index]["score"])
        end = index
        while end < len(ordered) and float(ordered[end]["score"]) == score:
            if int(ordered[end]["label"]) == 1:
                tp += 1
            else:
                fp += 1
            end += 1

        fn = positives - tp
        tn = negatives - fp
        sensitivity = tp / positives
        specificity = tn / negatives
        precision = tp / (tp + fp) if tp + fp else 0.0
        balanced = (sensitivity + specificity) / 2.0
        accuracy = (tp + tn) / len(ordered)
        if sensitivity + 1e-12 >= 0.90:
            next_score = (
                float(ordered[end]["score"])
                if end < len(ordered)
                else score - 2e-9
            )
            threshold = (score + next_score) / 2.0
            metrics = {
                "n": len(ordered),
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "precision_positive": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
            key = (
                precision,
                specificity,
                balanced,
                accuracy,
                -abs(threshold),
            )
            if best is None or key > best[0]:
                best = (key, threshold, metrics)
        index = end

    if best is None:
        return None
    return float(best[1]), best[2]


def apply_weight_threshold(
    rows: list[dict], weights: tuple[float, float, float], threshold: float
) -> list[dict]:
    metric_rows = scored(rows, weights)
    for row in metric_rows:
        row["pred"] = int(row["score"] >= threshold)
    return metric_rows


def crossfit(rows: list[dict], *, optimize_weights: bool) -> dict:
    outputs = []
    receipts = []
    for fold in range(N_FOLDS):
        train = [r for r in rows if patient_fold(r["patient_id"]) != fold]
        test = [r for r in rows if patient_fold(r["patient_id"]) == fold]
        train_patients = {r["patient_id"] for r in train}
        test_patients = {r["patient_id"] for r in test}
        if train_patients & test_patients:
            raise RuntimeError("patient leakage across folds")
        candidates = weight_grid() if optimize_weights else [(1 / 3, 1 / 3, 1 / 3)]
        best = None
        for weights in candidates:
            selected = choose_precision_sens90(train, weights)
            if selected is None:
                continue
            threshold, train_metrics = selected
            key = (
                train_metrics["precision_positive"],
                train_metrics["specificity"],
                train_metrics["balanced_accuracy"],
                train_metrics["accuracy"],
            )
            if best is None or key > best[0]:
                best = (key, weights, threshold, train_metrics)
        if best is None:
            raise RuntimeError(f"no feasible sensitivity>=0.90 solution for fold {fold}")
        _, weights, threshold, train_metrics = best
        outputs.extend(apply_weight_threshold(test, weights, threshold))
        receipts.append(
            {
                "fold": fold,
                "train_patients": len(train_patients),
                "test_patients": len(test_patients),
                "weights": {
                    "224": weights[0],
                    "320": weights[1],
                    "384": weights[2],
                },
                "threshold": threshold,
                "train_precision_positive": train_metrics["precision_positive"],
                "train_sensitivity": train_metrics["sensitivity"],
                "train_specificity": train_metrics["specificity"],
            }
        )
    return {
        "metrics": metric_summary(outputs),
        "fold_receipts": receipts,
        "patient_leakage": False,
    }


def analyze(rows: list[dict]) -> dict:
    add_label_free_ensembles(rows)
    singles = {
        str(r): metric_summary(
            to_metric_rows(rows, score_key=f"score_{r}", pred_key=f"pred_{r}")
        )
        for r in RESOLUTIONS
    }
    equal = metric_summary(
        to_metric_rows(rows, score_key="score_equal", pred_key="pred_equal")
    )
    majority = metric_summary(
        to_metric_rows(rows, score_key="score_majority", pred_key="pred_majority")
    )
    equal_crossfit = crossfit(rows, optimize_weights=False)
    weighted_crossfit = crossfit(rows, optimize_weights=True)
    return {
        "schema": "m07.external.rsna.resolution_ensemble.v1",
        "status": "PASS",
        "source_resolutions": list(RESOLUTIONS),
        "n": len(rows),
        "patients": len({r["patient_id"] for r in rows}),
        "model_weights_changed": False,
        "retraining_performed": False,
        "raw_external_results_preserved": True,
        "label_free_exploratory": {
            "single_resolutions": singles,
            "equal_weight_threshold0": equal,
            "majority_vote": majority,
            "error_overlap": fp_sets(rows),
        },
        "external_crossfit_adaptation": {
            "classification": "SECONDARY_EXTERNAL_ADAPTATION_ANALYSIS",
            "split_unit": "PATIENT",
            "constraint": "train-fold sensitivity >= 0.90",
            "objective": "maximize train-fold positive precision, then specificity",
            "equal_weight_threshold_crossfit": equal_crossfit,
            "weight_and_threshold_crossfit": weighted_crossfit,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise SystemExit("CGP_READ_OIDC_TOKEN unavailable")
    outputs = fetch_all(token)
    rows = align(outputs)
    result = analyze(rows)
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M07_RSNA_RESOLUTION_ENSEMBLE=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
