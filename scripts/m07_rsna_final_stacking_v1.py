from __future__ import annotations

import argparse
import json
import math
import os
import pathlib

import m07_rsna_age_audit_v1 as base
import m07_rsna_resolution_ensemble_v1 as ens
import m07_rsna_r224_crossfit_calibration_v1 as cal
import m07_rsna_subgroup_calibration_v1 as subgroup

FLOORS = (0.90, 0.92, 0.94)
L2 = 0.10
FEATURE_NAMES = (
    "score_224",
    "score_320",
    "score_384",
    "view_is_ap",
    "age_years",
    "ap_x_age",
    "resolution_spread",
    "mean_abs_score",
)


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def merge_metadata(rows: list[dict], r224_rows: list[dict]) -> list[dict]:
    meta = {str(row["image_id"]): row for row in r224_rows}
    if len(meta) != len(r224_rows):
        raise RuntimeError("duplicate R224 metadata image_id")
    output = []
    for row in rows:
        key = str(row["image_id"])
        if key not in meta:
            raise RuntimeError("missing R224 metadata for " + key)
        ref = meta[key]
        if (
            str(ref["patient_id"]) != str(row["patient_id"])
            or int(ref["age"]) != int(row["age"])
            or int(ref["label"]) != int(row["label"])
        ):
            raise RuntimeError("identity mismatch for " + key)
        item = dict(row)
        item["view"] = str(ref["view"])
        item["sex"] = str(ref["sex"])
        output.append(item)
    return output


def raw_features(row: dict) -> list[float]:
    scores = [
        float(row["score_224"]),
        float(row["score_320"]),
        float(row["score_384"]),
    ]
    is_ap = 1.0 if str(row["view"]).upper() == "AP" else 0.0
    age = float(row["age"])
    return [
        scores[0],
        scores[1],
        scores[2],
        is_ap,
        age,
        is_ap * age,
        max(scores) - min(scores),
        sum(abs(value) for value in scores) / 3.0,
    ]


def patient_weights(rows: list[dict]) -> list[float]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["patient_id"])
        counts[key] = counts.get(key, 0) + 1
    return [1.0 / counts[str(row["patient_id"])] for row in rows]


def fit_scaler(rows: list[dict]) -> dict:
    matrix = [raw_features(row) for row in rows]
    weights = patient_weights(rows)
    total = sum(weights)
    means = []
    scales = []
    for column in range(len(FEATURE_NAMES)):
        mean = sum(w * x[column] for x, w in zip(matrix, weights)) / total
        var = sum(
            w * (x[column] - mean) ** 2
            for x, w in zip(matrix, weights)
        ) / total
        means.append(mean)
        scales.append(math.sqrt(max(var, 1e-8)))
    return {"means": means, "scales": scales}


def transform(row: dict, scaler: dict) -> list[float]:
    values = raw_features(row)
    return [
        (values[i] - scaler["means"][i]) / scaler["scales"][i]
        for i in range(len(values))
    ]


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    work = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            return None
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        for j in range(column, n + 1):
            work[column][j] /= scale
        for row in range(n):
            if row == column:
                continue
            scale = work[row][column]
            for j in range(column, n + 1):
                work[row][j] -= scale * work[column][j]
    return [work[i][n] for i in range(n)]


def objective(rows: list[dict], scaler: dict, theta: list[float]) -> float:
    weights = patient_weights(rows)
    total = 0.5 * L2 * sum(value * value for value in theta[1:])
    for row, weight in zip(rows, weights):
        x = [1.0] + transform(row, scaler)
        y = float(row["label"])
        p = min(1.0 - 1e-8, max(1e-8, sigmoid(sum(a * b for a, b in zip(theta, x)))))
        total += -weight * (y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total


def fit_logistic(rows: list[dict]) -> dict:
    scaler = fit_scaler(rows)
    dimension = len(FEATURE_NAMES) + 1
    theta = [0.0] * dimension
    weights = patient_weights(rows)

    prevalence = sum(
        w * float(row["label"]) for row, w in zip(rows, weights)
    ) / sum(weights)
    prevalence = min(1.0 - 1e-6, max(1e-6, prevalence))
    theta[0] = math.log(prevalence / (1.0 - prevalence))
    current = objective(rows, scaler, theta)

    for _ in range(80):
        gradient = [0.0] * dimension
        hessian = [[0.0] * dimension for _ in range(dimension)]
        for j in range(1, dimension):
            gradient[j] += L2 * theta[j]
            hessian[j][j] += L2

        for row, weight in zip(rows, weights):
            x = [1.0] + transform(row, scaler)
            y = float(row["label"])
            z = sum(a * b for a, b in zip(theta, x))
            p = sigmoid(z)
            error = p - y
            curvature = max(1e-8, p * (1.0 - p))
            for j in range(dimension):
                gradient[j] += weight * error * x[j]
                for k in range(dimension):
                    hessian[j][k] += weight * curvature * x[j] * x[k]

        delta = solve_linear(hessian, gradient)
        if delta is None:
            break
        if max(abs(value) for value in delta) < 1e-8:
            break

        step = 1.0
        accepted = False
        while step >= 1e-6:
            candidate = [
                theta[j] - step * delta[j]
                for j in range(dimension)
            ]
            value = objective(rows, scaler, candidate)
            if math.isfinite(value) and value < current - 1e-10:
                theta = candidate
                current = value
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    return {
        "theta": theta,
        "scaler": scaler,
        "l2": L2,
        "objective": current,
        "feature_names": list(FEATURE_NAMES),
        "solver": "damped_newton_backtracking",
    }


def score_rows(rows: list[dict], model: dict) -> list[dict]:
    output = []
    theta = model["theta"]
    scaler = model["scaler"]
    for row in rows:
        x = [1.0] + transform(row, scaler)
        logit = sum(a * b for a, b in zip(theta, x))
        p = sigmoid(logit)
        item = {
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "age": row["age"],
            "view": row["view"],
            "sex": row["sex"],
            "label": row["label"],
            "score": logit,
            "prob": p,
            "pred": 0,
        }
        output.append(item)
    return output


def choose_threshold(rows: list[dict], floor: float) -> tuple[float, dict]:
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    positives = sum(int(row["label"]) == 1 for row in ordered)
    negatives = len(ordered) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("threshold training fold lacks a class")

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

        sensitivity = tp / positives
        if sensitivity + 1e-12 >= floor:
            fn = positives - tp
            tn = negatives - fp
            specificity = tn / negatives
            precision = tp / (tp + fp) if tp + fp else 0.0
            balanced = (sensitivity + specificity) / 2.0
            next_score = (
                float(ordered[end]["score"])
                if end < len(ordered)
                else score - 2e-9
            )
            threshold = (score + next_score) / 2.0
            metrics = {
                "precision_positive": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
            key = (
                precision,
                specificity,
                balanced,
                -fp,
                -abs(threshold),
            )
            if best is None or key > best[0]:
                best = (key, threshold, metrics)
        index = end

    if best is None:
        raise RuntimeError("no threshold satisfies sensitivity floor")
    return float(best[1]), best[2]


def compact(metrics: dict) -> dict:
    keys = (
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
    return {key: metrics[key] for key in keys}


def crossfit(rows: list[dict], floor: float) -> dict:
    predictions = []
    receipts = []
    for fold in range(cal.N_FOLDS):
        train = [
            row for row in rows
            if cal.patient_fold(str(row["patient_id"])) != fold
        ]
        test = [
            row for row in rows
            if cal.patient_fold(str(row["patient_id"])) == fold
        ]
        train_patients = {str(row["patient_id"]) for row in train}
        test_patients = {str(row["patient_id"]) for row in test}
        if train_patients & test_patients:
            raise RuntimeError("patient leakage in final stacking")

        model = fit_logistic(train)
        train_scored = score_rows(train, model)
        threshold, train_metrics = choose_threshold(train_scored, floor)
        test_scored = score_rows(test, model)
        for row in test_scored:
            row["pred"] = int(float(row["score"]) >= threshold)
        predictions.extend(test_scored)

        receipts.append({
            "fold": fold,
            "train_patients": len(train_patients),
            "test_patients": len(test_patients),
            "sensitivity_floor": floor,
            "threshold": threshold,
            "train_metrics": train_metrics,
            "model": {
                "l2": model["l2"],
                "objective": model["objective"],
                "solver": model["solver"],
                "feature_names": model["feature_names"],
                "theta": model["theta"],
            },
        })

    if len(predictions) != len(rows):
        raise RuntimeError("stacking did not emit one prediction per row")
    return {
        "floor": floor,
        "metrics": compact(base.metrics(predictions)),
        "fold_receipts": receipts,
        "patient_leakage": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise SystemExit("CGP_READ_OIDC_TOKEN unavailable")

    resolution_rows = ens.align(ens.fetch_all(token))
    r224_rows = subgroup.fetch_r224(token)
    rows = merge_metadata(resolution_rows, r224_rows)

    baseline = ens.metric_summary(
        ens.to_metric_rows(rows, score_key="score_224", pred_key="pred_224")
    )
    experiments = [crossfit(rows, floor) for floor in FLOORS]

    report = {
        "schema": "m07.external.rsna.final_stacking.v1",
        "status": "PASS_FINAL_STACKING_ANALYSIS",
        "analysis_classification": "SECONDARY_EXTERNAL_CROSSFIT_ADAPTATION_ANALYSIS",
        "analysis_is_pure_external_validation": False,
        "raw_external_results_preserved": True,
        "model_weights_changed": False,
        "base_model_retraining_performed": False,
        "stacker_fit_performed": True,
        "split_unit": "PATIENT",
        "n_folds": cal.N_FOLDS,
        "feature_names": list(FEATURE_NAMES),
        "predeclared_l2": L2,
        "baseline_r224": baseline,
        "crossfit_experiments": experiments,
        "limitations": [
            "This is a secondary external adaptation analysis, not a new untouched external validation.",
            "All stacker fitting and operating-point selection are confined to the training side of each patient-level outer fold.",
            "No base-model weights are changed and the canonical frozen external result remains the primary reference.",
        ],
    }

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("M07_RSNA_FINAL_STACKING=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
