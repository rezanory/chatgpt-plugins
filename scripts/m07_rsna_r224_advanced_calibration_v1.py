from __future__ import annotations

import argparse
import json
import math
import os
import pathlib

import m07_rsna_age_audit_v1 as base
import m07_rsna_r224_crossfit_calibration_v1 as prior

N_FOLDS = prior.N_FOLDS


def clamp_prob(p: float) -> float:
    return min(1.0 - 1e-7, max(1e-7, float(p)))


def weighted_nll(rows: list[dict], temperature: float) -> float:
    weights = prior.patient_weights(rows)
    total = 0.0
    wsum = 0.0
    t = max(0.05, min(20.0, float(temperature)))
    for row, weight in zip(rows, weights):
        y = int(row["label"])
        p = clamp_prob(prior.sigmoid(float(row["score"]) / t))
        total += -weight * (y * math.log(p) + (1 - y) * math.log(1 - p))
        wsum += weight
    return total / wsum


def fit_temperature(rows: list[dict]) -> float:
    lo = math.log(0.05)
    hi = math.log(20.0)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    x1 = hi - (hi - lo) / phi
    x2 = lo + (hi - lo) / phi
    f1 = weighted_nll(rows, math.exp(x1))
    f2 = weighted_nll(rows, math.exp(x2))
    for _ in range(80):
        if f1 <= f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - (hi - lo) / phi
            f1 = weighted_nll(rows, math.exp(x1))
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + (hi - lo) / phi
            f2 = weighted_nll(rows, math.exp(x2))
    return math.exp((lo + hi) / 2.0)


def apply_temperature(rows: list[dict], temperature: float) -> list[dict]:
    output = []
    for row in rows:
        score = float(row["score"]) / temperature
        p = prior.sigmoid(score)
        item = dict(row)
        item["score"] = score
        item["prob"] = p
        item["pred"] = int(p >= 0.5)
        output.append(item)
    return output


def fit_isotonic(rows: list[dict]) -> list[dict]:
    weights = prior.patient_weights(rows)
    samples = sorted(
        (prior.sigmoid(float(row["score"])), float(row["label"]), weight)
        for row, weight in zip(rows, weights)
    )
    blocks: list[dict] = []
    for x, y, weight in samples:
        blocks.append({"lo": x, "hi": x, "sw": weight, "swy": weight * y})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left["swy"] / left["sw"] <= right["swy"] / right["sw"] + 1e-15:
                break
            blocks[-2:] = [{
                "lo": left["lo"],
                "hi": right["hi"],
                "sw": left["sw"] + right["sw"],
                "swy": left["swy"] + right["swy"],
            }]
    return [
        {"lo": block["lo"], "hi": block["hi"], "value": block["swy"] / block["sw"]}
        for block in blocks
    ]


def isotonic_predict(p: float, blocks: list[dict]) -> float:
    for block in blocks:
        if p <= float(block["hi"]) + 1e-15:
            return float(block["value"])
    return float(blocks[-1]["value"])


def apply_isotonic(rows: list[dict], blocks: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        p = isotonic_predict(prior.sigmoid(float(row["score"])), blocks)
        item = dict(row)
        item["prob"] = p
        item["score"] = p - 0.5
        item["pred"] = int(p >= 0.5)
        output.append(item)
    return output


def solve3(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    work = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-12:
            return None
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        for j in range(column, 4):
            work[column][j] /= scale
        for row in range(3):
            if row == column:
                continue
            scale = work[row][column]
            for j in range(column, 4):
                work[row][j] -= scale * work[column][j]
    return [work[i][3] for i in range(3)]


def beta_features(score: float) -> tuple[float, float, float]:
    p = clamp_prob(prior.sigmoid(float(score)))
    return math.log(p), -math.log(1.0 - p), 1.0


def fit_beta(rows: list[dict], l2: float = 1e-3) -> dict:
    weights = prior.patient_weights(rows)
    theta = [1.0, 1.0, 0.0]

    def objective(params: list[float]) -> float:
        total = 0.5 * l2 * (params[0] ** 2 + params[1] ** 2)
        for row, weight in zip(rows, weights):
            features = beta_features(float(row["score"]))
            y = float(row["label"])
            z = sum(params[j] * features[j] for j in range(3))
            p = clamp_prob(prior.sigmoid(z))
            total += -weight * (y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        return total

    current = objective(theta)
    for _ in range(100):
        gradient = [l2 * theta[0], l2 * theta[1], 0.0]
        hessian = [[0.0] * 3 for _ in range(3)]
        hessian[0][0] += l2
        hessian[1][1] += l2
        for row, weight in zip(rows, weights):
            features = beta_features(float(row["score"]))
            y = float(row["label"])
            z = sum(theta[j] * features[j] for j in range(3))
            p = prior.sigmoid(z)
            error = p - y
            curvature = max(1e-8, p * (1.0 - p))
            for j in range(3):
                gradient[j] += weight * error * features[j]
                for k in range(3):
                    hessian[j][k] += weight * curvature * features[j] * features[k]

        delta = solve3(hessian, gradient)
        if delta is None:
            break
        if max(abs(value) for value in delta) < 1e-8:
            break

        step = 1.0
        accepted = False
        while step >= 1e-6:
            candidate = [theta[j] - step * delta[j] for j in range(3)]
            value = objective(candidate)
            if math.isfinite(value) and value < current - 1e-12:
                theta = candidate
                current = value
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    return {
        "a": theta[0],
        "b": theta[1],
        "c": theta[2],
        "l2": l2,
        "objective": current,
        "solver": "damped_newton_backtracking",
    }


def apply_beta(rows: list[dict], params: dict) -> list[dict]:
    output = []
    for row in rows:
        features = beta_features(float(row["score"]))
        z = (
            params["a"] * features[0]
            + params["b"] * features[1]
            + params["c"]
        )
        p = prior.sigmoid(z)
        item = dict(row)
        item["score"] = z
        item["prob"] = p
        item["pred"] = int(p >= 0.5)
        output.append(item)
    return output


def crossfit(rows: list[dict]) -> dict:
    fold_audit = prior.validate_folds(rows)
    outputs = {"temperature": [], "isotonic": [], "beta": []}
    fold_receipts = []
    for fold in range(N_FOLDS):
        train = [
            row for row in rows
            if prior.patient_fold(str(row["patient_id"])) != fold
        ]
        test = [
            row for row in rows
            if prior.patient_fold(str(row["patient_id"])) == fold
        ]
        temperature = fit_temperature(train)
        isotonic = fit_isotonic(train)
        beta = fit_beta(train)
        outputs["temperature"].extend(apply_temperature(test, temperature))
        outputs["isotonic"].extend(apply_isotonic(test, isotonic))
        outputs["beta"].extend(apply_beta(test, beta))
        fold_receipts.append({
            "fold": fold,
            "train_patients": len({str(row["patient_id"]) for row in train}),
            "test_patients": len({str(row["patient_id"]) for row in test}),
            "temperature": temperature,
            "isotonic_blocks": len(isotonic),
            "beta": beta,
        })

    baseline = prior.evaluated([dict(row) for row in rows])
    methods = {name: prior.evaluated(method_rows) for name, method_rows in outputs.items()}
    return {
        "schema": "m07.external.rsna.r224.advanced_calibration.v1",
        "status": "PASS_ADVANCED_CROSSFIT_CALIBRATION_ANALYSIS",
        "analysis_classification": "EXTERNAL_CROSSFIT_CALIBRATION_ADAPTATION_ANALYSIS",
        "analysis_is_pure_external_validation": False,
        "raw_external_result_preserved": True,
        "training_performed": False,
        "model_weights_changed": False,
        "split_unit": "PATIENT",
        "patient_leakage": False,
        "n_folds": N_FOLDS,
        "fold_audit": fold_audit,
        "baseline_frozen": baseline,
        "methods": methods,
        "fold_receipts": fold_receipts,
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
    print("M07_RSNA_ADVANCED_CALIBRATION=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
