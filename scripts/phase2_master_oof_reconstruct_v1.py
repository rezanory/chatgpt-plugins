from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
from typing import Any

EXPECTED_MODELS = tuple(f"M{i:02d}" for i in range(1, 13))
EXPECTED_RESOLUTIONS = (224, 320, 384)
PRIMARY_METRIC = "balanced_accuracy"
SELECTION_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "auroc",
    "recall_normal",
    "recall_pneumonia",
    "precision_normal",
    "precision_pneumonia",
    "f1_normal",
    "f1_pneumonia",
    "f2",
)
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


class MasterAnalysisError(RuntimeError):
    pass


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _find_final_report_source(unit: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for source in unit.get("json_sources") or []:
        name = str(source.get("source") or "").replace("\\", "/").lower()
        schema = str(source.get("schema") or "")
        if name.endswith("final_report.json") or (
            schema in {
                "pneumonia.phase2.model_resolution.v1.6",
                "m07.gate.multires.model_resolution.v1.6",
            }
        ):
            candidates.append(source)
    if len(candidates) != 1:
        raise MasterAnalysisError(
            f"FINAL_REPORT_SOURCE_COUNT_INVALID:{unit.get('model_id')}:{unit.get('resolution')}:{len(candidates)}"
        )
    return candidates[0]


def _metric_block(source: dict[str, Any], path: str) -> dict[str, Any]:
    blocks = [
        block.get("values")
        for block in source.get("metric_blocks") or []
        if str(block.get("path") or "") == path
        and isinstance(block.get("values"), dict)
    ]
    if len(blocks) != 1:
        raise MasterAnalysisError(f"METRIC_BLOCK_COUNT_INVALID:{path}:{len(blocks)}")
    return blocks[0]


def _policy_scalar(source: dict[str, Any], suffix: str) -> Any:
    hits = [
        item.get("value")
        for item in source.get("policy_paths") or []
        if str(item.get("path") or "").endswith(suffix)
    ]
    if len(hits) != 1:
        raise MasterAnalysisError(f"POLICY_PATH_COUNT_INVALID:{suffix}:{len(hits)}")
    return hits[0]


def _normalize_oof(oof: dict[str, Any]) -> dict[str, float]:
    missing = [metric for metric in SELECTION_METRICS if metric not in oof]
    if missing:
        raise MasterAnalysisError("OOF_METRICS_MISSING=" + ",".join(missing))
    result: dict[str, float] = {}
    for metric in SELECTION_METRICS:
        value = oof[metric]
        if not _is_finite_number(value):
            raise MasterAnalysisError(f"OOF_METRIC_NONFINITE:{metric}:{value!r}")
        result[metric] = float(value)
    return result


def build_master_matrix(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema") != "pneumonia.phase2.final_evidence.matrix.v1":
        raise MasterAnalysisError("FINAL_EVIDENCE_MATRIX_SCHEMA_INVALID")
    if payload.get("status") != "PASS":
        raise MasterAnalysisError("FINAL_EVIDENCE_MATRIX_STATUS_INVALID")
    if payload.get("training_performed") is not False:
        raise MasterAnalysisError("FINAL_EVIDENCE_MATRIX_TRAINING_FLAG_INVALID")
    if payload.get("locked_test_executed") is not False:
        # The extractor itself must not execute test inference. Historical FINAL_REPORT
        # may contain prior confirmatory locked-test results; those are never used below.
        raise MasterAnalysisError("FINAL_EVIDENCE_EXTRACTOR_LOCKED_TEST_FLAG_INVALID")
    units = payload.get("units") or []
    expected = {(model, resolution) for model in EXPECTED_MODELS for resolution in EXPECTED_RESOLUTIONS}
    observed = {(str(unit.get("model_id")), int(unit.get("resolution"))) for unit in units}
    if len(units) != 36 or observed != expected:
        raise MasterAnalysisError(
            "FINAL_EVIDENCE_MATRIX_COVERAGE_INVALID:"
            + json.dumps(
                {
                    "count": len(units),
                    "missing": sorted(expected - observed),
                    "extra": sorted(observed - expected),
                },
                sort_keys=True,
            )
        )

    rows: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: (str(item["model_id"]), int(item["resolution"]))):
        final_report = _find_final_report_source(unit)
        if _policy_scalar(final_report, "locked_test_used_for_selection") is not False:
            raise MasterAnalysisError(
                f"LOCKED_TEST_SELECTION_POLICY_INVALID:{unit['model_id']}:R{unit['resolution']}"
            )
        if _policy_scalar(final_report, "external_used_for_selection") is not False:
            raise MasterAnalysisError(
                f"EXTERNAL_SELECTION_POLICY_INVALID:{unit['model_id']}:R{unit['resolution']}"
            )
        oof = _normalize_oof(_metric_block(final_report, "oof"))
        row = {
            "model_id": str(unit["model_id"]),
            "resolution": int(unit["resolution"]),
            "dataset": "OOF",
            "selection_role": "DEVELOPMENT_ONLY",
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
            "dataset_ref": str(unit.get("dataset_ref") or ""),
            "dataset_version_number": int(unit.get("dataset_version_number") or 0),
            **oof,
        }
        rows.append(row)
    return rows


def build_model_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in EXPECTED_MODELS}
    for row in rows:
        by_model[row["model_id"]].append(row)
    aggregate: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        model_rows = sorted(by_model[model], key=lambda item: item["resolution"])
        if [row["resolution"] for row in model_rows] != list(EXPECTED_RESOLUTIONS):
            raise MasterAnalysisError(f"MODEL_RESOLUTION_COVERAGE_INVALID:{model}")
        record: dict[str, Any] = {
            "model_id": model,
            "resolution_count": len(model_rows),
            "selection_role": "DESCRIPTIVE_DEVELOPMENT_ORDER_ONLY",
        }
        for metric in SELECTION_METRICS:
            values = [float(row[metric]) for row in model_rows]
            record[f"{metric}_mean"] = float(statistics.fmean(values))
            record[f"{metric}_sd"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
            record[f"{metric}_min"] = float(min(values))
            record[f"{metric}_max"] = float(max(values))
        aggregate.append(record)
    aggregate.sort(
        key=lambda item: (
            -float(item[f"{PRIMARY_METRIC}_mean"]),
            -float(item["macro_f1_mean"]),
            -float(item["mcc_mean"]),
            item["model_id"],
        )
    )
    for rank, row in enumerate(aggregate, 1):
        row["development_descriptive_rank"] = rank
    return aggregate


def build_predeclared_point_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["model_id"], row["resolution"]): row for row in rows}
    contrasts: list[dict[str, Any]] = []
    for resolution in EXPECTED_RESOLUTIONS:
        for reference, candidate, contrast in PREDECLARED_MODEL_COMPARISONS:
            ref = lookup[(reference, resolution)]
            cand = lookup[(candidate, resolution)]
            for metric in (PRIMARY_METRIC, "macro_f1", "mcc", "auroc", "recall_normal", "recall_pneumonia"):
                contrasts.append(
                    {
                        "dataset": "OOF",
                        "resolution": resolution,
                        "reference_model": reference,
                        "candidate_model": candidate,
                        "contrast": contrast,
                        "metric": metric,
                        "reference": float(ref[metric]),
                        "candidate": float(cand[metric]),
                        "delta_candidate_minus_reference": float(cand[metric] - ref[metric]),
                        "inference_status": "POINT_ESTIMATE_ONLY_PENDING_PAIRED_PATIENT_BOOTSTRAP_HOLM",
                    }
                )
    return contrasts


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise MasterAnalysisError(f"EMPTY_CSV:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source_path = pathlib.Path(args.input)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = build_master_matrix(payload)
    aggregate = build_model_aggregate(rows)
    contrasts = build_predeclared_point_contrasts(rows)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "MASTER_OOF_RESULTS_MATRIX.csv", rows)
    write_csv(out_dir / "MASTER_OOF_MODEL_AGGREGATE.csv", aggregate)
    write_csv(out_dir / "MASTER_PREDECLARED_OOF_POINT_CONTRASTS.csv", contrasts)

    report = {
        "schema": "pneumonia.phase2.master_oof_reconstruction.v1",
        "status": "PASS_DESCRIPTIVE_OOF_RECONSTRUCTION",
        "selection_dataset": "OOF_ONLY",
        "primary_metric": PRIMARY_METRIC,
        "model_resolution_rows": len(rows),
        "models": len(aggregate),
        "resolutions": list(EXPECTED_RESOLUTIONS),
        "predeclared_contrast_rows": len(contrasts),
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "development_order_is_confirmatory_champion_selection": False,
        "paired_statistics_status": "PENDING_PREDICTION_LEVEL_RECONSTRUCTION",
        "development_descriptive_order": [row["model_id"] for row in aggregate],
        "predeclared_model_comparisons": [list(item) for item in PREDECLARED_MODEL_COMPARISONS],
        "next_action": "RECONSTRUCT_ALIGNED_OOF_PREDICTIONS_AND_RUN_PAIRED_PATIENT_BOOTSTRAP_HOLM",
    }
    (out_dir / "MASTER_OOF_RECONSTRUCTION_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("PHASE2_MASTER_OOF_RECONSTRUCTION_PASS=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
