from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import urllib.request
from collections import defaultdict

import m07_rsna_age_audit_v1 as base
from m07_rsna_resolution_reconcile_v1 import RUNS, output_inventory, safe_output_url

PREDICTIONS = "M07_RSNA_PEDIATRIC_EXTERNAL_PREDICTIONS.csv"
N_FOLDS = 5
SEED_TAG = "m07-rsna-subgroup-calibration-v1"
SENSITIVITY_FLOORS = (0.90, 0.92, 0.94)
GROUP_FAMILIES = ("global", "age", "view", "age_view")


def age_band(age: int) -> str:
    if age <= 5:
        return "age_1_5"
    if age <= 9:
        return "age_6_9"
    return "age_10_18"


def subgroup_key(row: dict, family: str) -> str:
    if family == "global":
        return "all"
    if family == "age":
        return age_band(int(row["age"]))
    if family == "view":
        return str(row["view"])
    if family == "age_view":
        return age_band(int(row["age"])) + "|" + str(row["view"])
    raise ValueError(f"unknown subgroup family {family}")


def download_rows(url: str) -> list[dict]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "m07-rsna-subgroup-calibration-v1/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        text = response.read().decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    required = {
        "image_id",
        "patient_id",
        "age",
        "sex",
        "view",
        "label",
        "normalized_ensemble_score",
        "prediction_primary_normalized",
    }
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        missing = sorted(required - set(reader.fieldnames or []))
        raise RuntimeError("subgroup prediction CSV schema mismatch: " + ",".join(missing))
    rows = []
    for raw in reader:
        score = float(raw["normalized_ensemble_score"])
        rows.append(
            {
                "image_id": str(raw["image_id"]),
                "patient_id": str(raw["patient_id"]),
                "age": int(float(raw["age"])),
                "sex": str(raw["sex"]).strip() or "UNKNOWN",
                "view": str(raw["view"]).strip() or "UNKNOWN",
                "label": int(float(raw["label"])),
                "score": score,
                "pred": int(float(raw["prediction_primary_normalized"])),
                "prob": 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, score)))),
            }
        )
    if len(rows) != 1099:
        raise RuntimeError(f"expected 1099 rows, found {len(rows)}")
    return rows


def fetch_r224(token: str) -> list[dict]:
    spec = RUNS[224]
    files = output_inventory(spec["account_id"], spec["kernel_ref"], token)
    return download_rows(safe_output_url(files, PREDICTIONS))


def compact(metrics: dict) -> dict:
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


def patient_fold(patient_id: str) -> int:
    digest = hashlib.sha256((SEED_TAG + "|" + patient_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % N_FOLDS


def operating_points(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    positives = sum(row["label"] == 1 for row in ordered)
    negatives = len(ordered) - positives
    if not ordered:
        return []
    points = [
        {
            "tp": 0,
            "fp": 0,
            "threshold": float(ordered[0]["score"]) + 1e-9,
            "positives": positives,
            "negatives": negatives,
        }
    ]
    tp = fp = 0
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
        next_score = (
            float(ordered[end]["score"])
            if end < len(ordered)
            else score - 2e-9
        )
        points.append(
            {
                "tp": tp,
                "fp": fp,
                "threshold": (score + next_score) / 2.0,
                "positives": positives,
                "negatives": negatives,
            }
        )
        index = end
    # For each achievable TP retain the point with the fewest FP.
    by_tp = {}
    for point in points:
        tp_value = point["tp"]
        current = by_tp.get(tp_value)
        if current is None or point["fp"] < current["fp"]:
            by_tp[tp_value] = point
    return [by_tp[key] for key in sorted(by_tp)]


def optimize_group_thresholds(
    rows: list[dict], family: str, sensitivity_floor: float
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[subgroup_key(row, family)].append(row)
    total_positive = sum(row["label"] == 1 for row in rows)
    total_negative = len(rows) - total_positive
    required_tp = math.ceil(sensitivity_floor * total_positive - 1e-12)

    # Dynamic program: total TP -> (minimum FP, thresholds, tp_by_group)
    states = {0: (0, {}, {})}
    for group in sorted(grouped):
        points = operating_points(grouped[group])
        if not points:
            continue
        new_states = {}
        for total_tp, (total_fp, thresholds, group_tps) in states.items():
            for point in points:
                candidate_tp = total_tp + int(point["tp"])
                candidate_fp = total_fp + int(point["fp"])
                current = new_states.get(candidate_tp)
                if current is None or candidate_fp < current[0]:
                    next_thresholds = dict(thresholds)
                    next_thresholds[group] = float(point["threshold"])
                    next_group_tps = dict(group_tps)
                    next_group_tps[group] = int(point["tp"])
                    new_states[candidate_tp] = (
                        candidate_fp,
                        next_thresholds,
                        next_group_tps,
                    )
        states = new_states

    feasible = []
    for tp, (fp, thresholds, group_tps) in states.items():
        if tp < required_tp:
            continue
        fn = total_positive - tp
        tn = total_negative - fp
        sensitivity = tp / total_positive
        specificity = tn / total_negative if total_negative else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        balanced = (sensitivity + specificity) / 2.0
        accuracy = (tp + tn) / len(rows)
        feasible.append(
            (
                (
                    precision,
                    specificity,
                    balanced,
                    accuracy,
                    -fp,
                ),
                {
                    "thresholds": thresholds,
                    "tp_by_group": group_tps,
                    "metrics": {
                        "n": len(rows),
                        "accuracy": accuracy,
                        "balanced_accuracy": balanced,
                        "sensitivity": sensitivity,
                        "specificity": specificity,
                        "precision_positive": precision,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                    },
                },
            )
        )
    if not feasible:
        raise RuntimeError(
            f"no feasible threshold solution family={family} floor={sensitivity_floor}"
        )
    feasible.sort(key=lambda item: item[0], reverse=True)
    return feasible[0][1]


def apply_thresholds(rows: list[dict], family: str, thresholds: dict) -> list[dict]:
    if not thresholds:
        raise RuntimeError("empty threshold map")
    fallback = sum(thresholds.values()) / len(thresholds)
    output = []
    for row in rows:
        group = subgroup_key(row, family)
        threshold = float(thresholds.get(group, fallback))
        item = dict(row)
        item["pred"] = int(float(row["score"]) >= threshold)
        output.append(item)
    return output


def crossfit(rows: list[dict], family: str, sensitivity_floor: float) -> dict:
    outputs = []
    receipts = []
    for fold in range(N_FOLDS):
        train = [row for row in rows if patient_fold(row["patient_id"]) != fold]
        test = [row for row in rows if patient_fold(row["patient_id"]) == fold]
        train_patients = {row["patient_id"] for row in train}
        test_patients = {row["patient_id"] for row in test}
        if train_patients & test_patients:
            raise RuntimeError("patient leakage")
        selected = optimize_group_thresholds(train, family, sensitivity_floor)
        outputs.extend(apply_thresholds(test, family, selected["thresholds"]))
        receipts.append(
            {
                "fold": fold,
                "family": family,
                "sensitivity_floor": sensitivity_floor,
                "train_patients": len(train_patients),
                "test_patients": len(test_patients),
                "thresholds": selected["thresholds"],
                "train_metrics": selected["metrics"],
            }
        )
    return {
        "family": family,
        "train_sensitivity_floor": sensitivity_floor,
        "metrics": compact(base.metrics(outputs)),
        "fold_receipts": receipts,
        "patient_leakage": False,
    }


def audit_subgroups(rows: list[dict]) -> dict:
    audit = {}
    for family in ("age", "view", "sex", "age_view"):
        grouped = defaultdict(list)
        for row in rows:
            if family == "sex":
                key = row["sex"]
            else:
                key = subgroup_key(row, family)
            grouped[key].append(row)
        audit[family] = {
            key: compact(base.metrics(group_rows))
            for key, group_rows in sorted(grouped.items())
            if sum(r["label"] == 1 for r in group_rows) > 0
            and sum(r["label"] == 0 for r in group_rows) > 0
        }
    return audit


def analyze(rows: list[dict]) -> dict:
    experiments = []
    for family in GROUP_FAMILIES:
        for floor in SENSITIVITY_FLOORS:
            experiments.append(crossfit(rows, family, floor))
    return {
        "schema": "m07.external.rsna.subgroup_calibration.v1",
        "status": "PASS",
        "source_resolution": 224,
        "n": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "model_weights_changed": False,
        "retraining_performed": False,
        "raw_external_results_preserved": True,
        "baseline_frozen": compact(base.metrics(rows)),
        "subgroup_audit": audit_subgroups(rows),
        "patient_level_crossfit_adaptation": experiments,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise SystemExit("CGP_READ_OIDC_TOKEN unavailable")
    rows = fetch_r224(token)
    result = analyze(rows)
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M07_RSNA_SUBGROUP_CALIBRATION=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
