from __future__ import annotations

import argparse
import bisect
import json
import os
import pathlib

import m07_rsna_resolution_ensemble_v1 as prior
import m07_rsna_r224_crossfit_calibration_v1 as calibration

METHODS = (
    "rank_centered",
    "median_probability",
    "confidence_gated",
    "r224_fallback_disagreement",
)


def median3(a: float, b: float, c: float) -> float:
    return sorted((float(a), float(b), float(c)))[1]


def empirical_rank(sorted_values: list[float], value: float) -> float:
    left = bisect.bisect_left(sorted_values, float(value))
    right = bisect.bisect_right(sorted_values, float(value))
    return (left + right) / (2.0 * len(sorted_values))


def build_methods(rows: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    sorted_scores = {
        resolution: sorted(float(row[f"score_{resolution}"]) for row in rows)
        for resolution in prior.RESOLUTIONS
    }
    zero_rank = {
        resolution: empirical_rank(sorted_scores[resolution], 0.0)
        for resolution in prior.RESOLUTIONS
    }
    output = {method: [] for method in METHODS}

    for row in rows:
        scores = {
            resolution: float(row[f"score_{resolution}"])
            for resolution in prior.RESOLUTIONS
        }
        probabilities = {
            resolution: prior.sigmoid(scores[resolution])
            for resolution in prior.RESOLUTIONS
        }
        centered_ranks = {
            resolution: (
                empirical_rank(sorted_scores[resolution], scores[resolution])
                - zero_rank[resolution]
            )
            for resolution in prior.RESOLUTIONS
        }

        score_rank = sum(centered_ranks.values()) / 3.0
        score_median = median3(scores[224], scores[320], scores[384])
        selected_resolution = max(
            prior.RESOLUTIONS,
            key=lambda resolution: (abs(scores[resolution]), -resolution),
        )
        score_confidence = scores[selected_resolution]

        signs = [int(scores[resolution] >= 0.0) for resolution in prior.RESOLUTIONS]
        score_fallback = (
            sum(scores.values()) / 3.0
            if len(set(signs)) == 1
            else scores[224]
        )

        values = {
            "rank_centered": (score_rank, prior.sigmoid(8.0 * score_rank)),
            "median_probability": (
                score_median,
                median3(probabilities[224], probabilities[320], probabilities[384]),
            ),
            "confidence_gated": (score_confidence, prior.sigmoid(score_confidence)),
            "r224_fallback_disagreement": (
                score_fallback,
                prior.sigmoid(score_fallback),
            ),
        }

        for method, (score, probability) in values.items():
            output[method].append({
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "age": row["age"],
                "label": row["label"],
                "score": float(score),
                "prob": float(probability),
                "pred": int(score >= 0.0),
            })

    return output, {"zero_rank": zero_rank}


def crossfit_sens90(method_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    output = []
    fold_receipts = []
    for fold in range(calibration.N_FOLDS):
        train = [
            row for row in method_rows
            if calibration.patient_fold(str(row["patient_id"])) != fold
        ]
        test = [
            row for row in method_rows
            if calibration.patient_fold(str(row["patient_id"])) == fold
        ]
        threshold, train_metrics = calibration.choose_sens90_threshold(train)
        for row in test:
            item = dict(row)
            item["pred"] = int(float(item["score"]) >= threshold)
            output.append(item)
        fold_receipts.append({
            "fold": fold,
            "threshold": threshold,
            "train_metrics": train_metrics,
        })
    return output, fold_receipts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise SystemExit("CGP_READ_OIDC_TOKEN unavailable")

    rows = prior.align(prior.fetch_all(token))
    prior.add_label_free_ensembles(rows)
    methods, method_meta = build_methods(rows)

    label_free = {
        method: prior.metric_summary(method_rows)
        for method, method_rows in methods.items()
    }
    crossfit_adaptation = {}
    crossfit_receipts = {}
    for method, method_rows in methods.items():
        crossfit_rows, receipts = crossfit_sens90(method_rows)
        crossfit_adaptation[method] = prior.metric_summary(crossfit_rows)
        crossfit_receipts[method] = receipts

    baseline_r224 = prior.metric_summary(
        prior.to_metric_rows(rows, score_key="score_224", pred_key="pred_224")
    )

    report = {
        "schema": "m07.external.rsna.advanced_ensemble.v1",
        "status": "PASS_ADVANCED_ENSEMBLE_ANALYSIS",
        "raw_external_result_preserved": True,
        "model_weights_changed": False,
        "training_performed": False,
        "label_free_methods": label_free,
        "crossfit_sens90_adaptation": crossfit_adaptation,
        "crossfit_classification": "EXTERNAL_CROSSFIT_ENSEMBLE_ADAPTATION_ANALYSIS",
        "baseline_r224": baseline_r224,
        "method_meta": method_meta,
        "crossfit_receipts": crossfit_receipts,
    }

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M07_RSNA_ADVANCED_ENSEMBLE=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
