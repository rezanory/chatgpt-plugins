from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
from collections import Counter

import m07_rsna_age_audit_v1 as base
import m07_rsna_resolution_ensemble_v1 as ensemble
import m07_rsna_subgroup_calibration_v1 as subgroup

N_FOLDS = 5
SEED_TAG = "m07-rsna-final-combination-v1"
FEATURE_SPECS = ("scores", "scores_view", "scores_view_age", "scores_view_age_interactions")
L2_GRID = (0.01, 0.1, 1.0, 10.0)


def patient_fold(patient_id: str) -> int:
    digest = hashlib.sha256((SEED_TAG + "|" + patient_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % N_FOLDS


def patient_weights(rows: list[dict]) -> list[float]:
    counts = Counter(str(row["patient_id"]) for row in rows)
    return [1.0 / counts[str(row["patient_id"])] for row in rows]


def solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    work = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-10:
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


def fit_normalizer(rows: list[dict]) -> dict:
    keys = ("score_224", "score_320", "score_384", "age")
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        std = max(1e-6, math.sqrt(variance))
        result[key] = {"mean": mean, "std": std}
    return result


def z(row: dict, key: str, normalizer: dict) -> float:
    return (float(row[key]) - normalizer[key]["mean"]) / normalizer[key]["std"]


def features(row: dict, spec: str, normalizer: dict) -> list[float]:
    s224 = z(row, "score_224", normalizer)
    s320 = z(row, "score_320", normalizer)
    s384 = z(row, "score_384", normalizer)
    ap = 1.0 if str(row["view"]).upper() == "AP" else 0.0
    age = z(row, "age", normalizer)
    out = [1.0, s224, s320, s384]
    if spec in {"scores_view", "scores_view_age", "scores_view_age_interactions"}:
        out.append(ap)
    if spec in {"scores_view_age", "scores_view_age_interactions"}:
        out.append(age)
    if spec == "scores_view_age_interactions":
        out.extend((s224 * ap, s320 * ap, s384 * ap))
    return out


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def fit_logistic(rows: list[dict], spec: str, l2: float, normalizer: dict) -> list[float]:
    xs = [features(row, spec, normalizer) for row in rows]
    ys = [float(row["label"]) for row in rows]
    weights = patient_weights(rows)
    dim = len(xs[0])
    theta = [0.0] * dim

    def objective(params: list[float]) -> float:
        total = 0.5 * l2 * sum(value * value for value in params[1:])
        for x, y, weight in zip(xs, ys, weights):
            p = min(1.0 - 1e-8, max(1e-8, sigmoid(sum(a * b for a, b in zip(params, x)))))
            total += -weight * (y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        return total

    current = objective(theta)
    for _ in range(80):
        gradient = [0.0] * dim
        hessian = [[0.0] * dim for _ in range(dim)]
        for j in range(1, dim):
            gradient[j] += l2 * theta[j]
            hessian[j][j] += l2
        for x, y, weight in zip(xs, ys, weights):
            score = sum(a * b for a, b in zip(theta, x))
            p = sigmoid(score)
            error = p - y
            curvature = max(1e-7, p * (1.0 - p))
            for j in range(dim):
                gradient[j] += weight * error * x[j]
                for k in range(dim):
                    hessian[j][k] += weight * curvature * x[j] * x[k]
        delta = solve(hessian, gradient)
        if delta is None:
            break
        if max(abs(value) for value in delta) < 1e-8:
            break
        step = 1.0
        accepted = False
        while step >= 1e-6:
            candidate = [theta[j] - step * delta[j] for j in range(dim)]
            value = objective(candidate)
            if math.isfinite(value) and value < current - 1e-10:
                theta = candidate
                current = value
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return theta


def scored_rows(rows: list[dict], spec: str, normalizer: dict, theta: list[float]) -> list[dict]:
    output = []
    for row in rows:
        x = features(row, spec, normalizer)
        score = sum(a * b for a, b in zip(theta, x))
        item = {
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "age": row["age"],
            "label": row["label"],
            "score": float(score),
            "prob": sigmoid(score),
            "pred": 0,
        }
        output.append(item)
    return output


def choose_threshold(rows: list[dict]) -> tuple[float, dict] | None:
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
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
        accuracy = (tp + tn) / len(ordered)
        balanced = (sensitivity + specificity) / 2.0
        if sensitivity + 1e-12 >= 0.90:
            next_score = float(ordered[end]["score"]) if end < len(ordered) else score - 2e-9
            threshold = (score + next_score) / 2.0
            metrics = {
                "precision_positive": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
            key = (precision, specificity, balanced, accuracy, -fp)
            if best is None or key > best[0]:
                best = (key, threshold, metrics)
        index = end
    if best is None:
        return None
    return float(best[1]), best[2]


def align_with_metadata(token: str) -> list[dict]:
    rows = ensemble.align(ensemble.fetch_all(token))
    meta = {row["image_id"]: row for row in subgroup.fetch_r224(token)}
    if set(meta) != {row["image_id"] for row in rows}:
        raise RuntimeError("metadata and resolution image sets differ")
    output = []
    for row in rows:
        m = meta[row["image_id"]]
        if int(row["label"]) != int(m["label"]) or str(row["patient_id"]) != str(m["patient_id"]):
            raise RuntimeError("metadata identity mismatch")
        item = dict(row)
        item["view"] = m["view"]
        item["sex"] = m["sex"]
        output.append(item)
    return output


def crossfit(rows: list[dict]) -> dict:
    all_outputs = []
    receipts = []
    for fold in range(N_FOLDS):
        train = [row for row in rows if patient_fold(str(row["patient_id"])) != fold]
        test = [row for row in rows if patient_fold(str(row["patient_id"])) == fold]
        train_patients = {str(row["patient_id"]) for row in train}
        test_patients = {str(row["patient_id"]) for row in test}
        if train_patients & test_patients:
            raise RuntimeError("patient leakage")

        best = None
        for spec in FEATURE_SPECS:
            normalizer = fit_normalizer(train)
            for l2 in L2_GRID:
                theta = fit_logistic(train, spec, l2, normalizer)
                train_scored = scored_rows(train, spec, normalizer, theta)
                selected = choose_threshold(train_scored)
                if selected is None:
                    continue
                threshold, train_metrics = selected
                key = (
                    train_metrics["precision_positive"],
                    train_metrics["specificity"],
                    train_metrics["balanced_accuracy"],
                    train_metrics["accuracy"],
                    -len(theta),
                    -l2,
                )
                if best is None or key > best[0]:
                    best = (key, spec, l2, normalizer, theta, threshold, train_metrics)
        if best is None:
            raise RuntimeError(f"no feasible final-combination candidate for fold {fold}")

        _, spec, l2, normalizer, theta, threshold, train_metrics = best
        test_scored = scored_rows(test, spec, normalizer, theta)
        for row in test_scored:
            row["pred"] = int(float(row["score"]) >= threshold)
        all_outputs.extend(test_scored)
        receipts.append({
            "fold": fold,
            "feature_spec": spec,
            "l2": l2,
            "threshold": threshold,
            "train_patients": len(train_patients),
            "test_patients": len(test_patients),
            "train_metrics": train_metrics,
            "coefficients": theta,
        })

    return {
        "metrics": ensemble.metric_summary(all_outputs),
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
    rows = align_with_metadata(token)
    result = {
        "schema": "m07.external.rsna.final_combination.v1",
        "status": "PASS_FINAL_COMBINATION_ANALYSIS",
        "analysis_classification": "EXTERNAL_CROSSFIT_STACKING_ADAPTATION_ANALYSIS",
        "analysis_is_pure_external_validation": False,
        "raw_external_result_preserved": True,
        "model_weights_changed": False,
        "m07_retraining_performed": False,
        "split_unit": "PATIENT",
        "constraint": "train-fold sensitivity >= 0.90",
        "objective": "maximize train-fold precision, then specificity",
        "candidate_features": list(FEATURE_SPECS),
        "candidate_l2": list(L2_GRID),
        "baseline_r224": ensemble.metric_summary(
            ensemble.to_metric_rows(rows, score_key="score_224", pred_key="pred_224")
        ),
        "final_crossfit_combination": crossfit(rows),
    }
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M07_RSNA_FINAL_COMBINATION=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
