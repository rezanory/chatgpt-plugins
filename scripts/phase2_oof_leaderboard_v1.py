from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
from typing import Any

PRIMARY_METRIC = "balanced_accuracy"
SECONDARY_METRICS = (
    "macro_f1",
    "mcc",
    "auroc",
    "recall_normal",
    "recall_pneumonia",
)
DISPLAY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mcc",
    "auroc",
    "precision_normal",
    "recall_normal",
    "precision_pneumonia",
    "recall_pneumonia",
    "f1_normal",
    "f1_pneumonia",
    "f2",
)
EXPECTED_MODELS = tuple(f"M{i:02d}" for i in range(1, 13))
EXPECTED_RESOLUTIONS = (224, 320, 384)


class LeaderboardError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LeaderboardError(f"NON_NUMERIC_METRIC:{name}:{value!r}") from exc
    if not math.isfinite(number):
        raise LeaderboardError(f"NON_FINITE_METRIC:{name}:{value!r}")
    return number


def _source_priority(source_name: str, path: str) -> int:
    source_upper = source_name.upper()
    path_lower = path.lower()
    if "M07_OOF_PRIMARY_METRICS.JSON" in source_upper and path in {"$", ""}:
        # Legacy M07/R224 pre-registered its primary OOF rule as the
        # fold-specific-threshold metrics file. The global-threshold file is
        # explicitly secondary and must never displace this source.
        return 0
    if "OOF_METRICS.JSON" in source_upper and path in {"$", ""}:
        return 0
    if "FINAL_REPORT.JSON" in source_upper and path_lower == "oof":
        return 1
    if path_lower.endswith(".oof") or path_lower == "oof":
        return 2
    return 99


def _policy_value(source: dict[str, Any], suffix: str) -> Any:
    found = []
    for row in source.get("policy_paths") or []:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "")
        if path.lower().endswith(suffix.lower()):
            found.append(row.get("value"))
    if len(found) > 1 and len({json.dumps(v, sort_keys=True) for v in found}) > 1:
        raise LeaderboardError(f"POLICY_VALUE_CONFLICT:{suffix}:{found}")
    return found[0] if found else None


def extract_oof_metrics(unit: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    candidates: list[tuple[int, str, str, dict[str, Any], dict[str, Any]]] = []
    for source in unit.get("json_sources") or []:
        if not isinstance(source, dict):
            continue
        source_name = str(source.get("source") or "")
        for block in source.get("metric_blocks") or []:
            if not isinstance(block, dict) or not isinstance(block.get("values"), dict):
                continue
            path = str(block.get("path") or "")
            priority = _source_priority(source_name, path)
            if priority >= 99:
                continue
            values = block["values"]
            if PRIMARY_METRIC not in values:
                continue
            if not all(metric in values for metric in SECONDARY_METRICS):
                continue
            candidates.append((priority, source_name, path, values, source))

    if not candidates:
        raise LeaderboardError(
            f"OOF_METRICS_NOT_FOUND:{unit.get('model_id')}:{unit.get('resolution')}"
        )
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    best_priority = candidates[0][0]
    best = [row for row in candidates if row[0] == best_priority]
    if len(best) > 1:
        canonical = {
            json.dumps(
                {k: row[3].get(k) for k in (PRIMARY_METRIC,) + SECONDARY_METRICS},
                sort_keys=True,
            )
            for row in best
        }
        if len(canonical) != 1:
            raise LeaderboardError(
                f"OOF_METRIC_SOURCE_CONFLICT:{unit.get('model_id')}:{unit.get('resolution')}"
            )
    _, source_name, path, values, source = best[0]

    metrics: dict[str, float] = {}
    for metric in DISPLAY_METRICS:
        if metric in values:
            metrics[metric] = _finite_number(
                values[metric],
                name=f"{unit.get('model_id')}.R{unit.get('resolution')}.{metric}",
            )
    for metric in (PRIMARY_METRIC,) + SECONDARY_METRICS:
        if metric not in metrics:
            metrics[metric] = _finite_number(
                values.get(metric),
                name=f"{unit.get('model_id')}.R{unit.get('resolution')}.{metric}",
            )

    for metric, value in metrics.items():
        if metric == "mcc":
            if not -1.0 <= value <= 1.0:
                raise LeaderboardError(f"METRIC_RANGE_INVALID:{metric}:{value}")
        elif not 0.0 <= value <= 1.0:
            raise LeaderboardError(f"METRIC_RANGE_INVALID:{metric}:{value}")

    locked_flag = _policy_value(source, "locked_test_used_for_selection")
    external_flag = _policy_value(source, "external_used_for_selection")
    if locked_flag not in (None, False):
        raise LeaderboardError(
            f"LOCKED_TEST_SELECTION_POLICY_VIOLATION:{unit.get('model_id')}:{unit.get('resolution')}"
        )
    if external_flag not in (None, False):
        raise LeaderboardError(
            f"EXTERNAL_SELECTION_POLICY_VIOLATION:{unit.get('model_id')}:{unit.get('resolution')}"
        )

    evidence = {
        "source": source_name,
        "path": path,
        "locked_test_used_for_selection": locked_flag,
        "external_used_for_selection": external_flag,
    }
    return metrics, evidence


def build_unit_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if matrix.get("schema") != "pneumonia.phase2.final_evidence.matrix.v1":
        raise LeaderboardError("MATRIX_SCHEMA_INVALID")
    if matrix.get("status") != "PASS":
        raise LeaderboardError("MATRIX_STATUS_NOT_PASS")
    units = matrix.get("units")
    if not isinstance(units, list) or len(units) != 36:
        raise LeaderboardError(f"MATRIX_UNIT_COUNT_INVALID:{len(units) if isinstance(units, list) else 'NA'}")

    expected = {
        (model_id, resolution)
        for model_id in EXPECTED_MODELS
        for resolution in EXPECTED_RESOLUTIONS
    }
    observed = {
        (str(unit.get("model_id")), int(unit.get("resolution")))
        for unit in units
    }
    if observed != expected:
        raise LeaderboardError(
            "MATRIX_COVERAGE_INVALID:"
            + json.dumps(
                {
                    "missing": sorted(expected - observed),
                    "extra": sorted(observed - expected),
                },
                sort_keys=True,
            )
        )

    rows: list[dict[str, Any]] = []
    for unit in units:
        model_id = str(unit["model_id"])
        resolution = int(unit["resolution"])
        metrics, evidence = extract_oof_metrics(unit)
        row = {
            "model_id": model_id,
            "resolution": resolution,
            **metrics,
            "oof_evidence_source": evidence["source"],
            "oof_evidence_path": evidence["path"],
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row[PRIMARY_METRIC]),
            -float(row.get("macro_f1", -1.0)),
            -float(row.get("mcc", -2.0)),
            -float(row.get("auroc", -1.0)),
            row["model_id"],
            row["resolution"],
        )
    )
    for index, row in enumerate(rows, start=1):
        row["descriptive_oof_rank"] = index
    return rows


def build_model_rows(unit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in EXPECTED_MODELS:
        subset = [row for row in unit_rows if row["model_id"] == model_id]
        if len(subset) != 3:
            raise LeaderboardError(f"MODEL_RESOLUTION_COUNT_INVALID:{model_id}:{len(subset)}")
        result: dict[str, Any] = {
            "model_id": model_id,
            "resolution_count": 3,
            "resolutions": "224,320,384",
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
        }
        for metric in (PRIMARY_METRIC,) + SECONDARY_METRICS:
            values = [float(row[metric]) for row in subset]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_sd_across_resolutions"] = statistics.pstdev(values)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        rows.append(result)

    rows.sort(
        key=lambda row: (
            -float(row[f"{PRIMARY_METRIC}_mean"]),
            float(row[f"{PRIMARY_METRIC}_sd_across_resolutions"]),
            -float(row["macro_f1_mean"]),
            -float(row["mcc_mean"]),
            row["model_id"],
        )
    )
    for index, row in enumerate(rows, start=1):
        row["descriptive_oof_model_rank"] = index
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise LeaderboardError(f"EMPTY_CSV:{path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    matrix_path = pathlib.Path(args.matrix)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    unit_rows = build_unit_rows(matrix)
    model_rows = build_model_rows(unit_rows)

    unit_path = out_dir / "PHASE2_OOF_UNIT_LEADERBOARD_V1.csv"
    model_path = out_dir / "PHASE2_OOF_MODEL_SUMMARY_V1.csv"
    receipt_path = out_dir / "PHASE2_OOF_LEADERBOARD_RECEIPT_V1.json"
    write_csv(unit_path, unit_rows)
    write_csv(model_path, model_rows)

    receipt = {
        "schema": "pneumonia.phase2.oof_leaderboard.v1",
        "status": "PASS_OOF_LEADERBOARD",
        "models": 12,
        "resolutions_per_model": 3,
        "model_resolution_units": 36,
        "folds_represented": 180,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
        "ordering_role": "DESCRIPTIVE_DEVELOPMENT_OOF_ONLY",
        "selection_performed": False,
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "paired_significance_performed": False,
        "next_action": "RUN_PREDECLARED_PAIRED_PATIENT_BOOTSTRAP_HOLM",
        "top_descriptive_unit": {
            "model_id": unit_rows[0]["model_id"],
            "resolution": unit_rows[0]["resolution"],
            PRIMARY_METRIC: unit_rows[0][PRIMARY_METRIC],
        },
        "top_descriptive_model_by_mean_primary": {
            "model_id": model_rows[0]["model_id"],
            f"{PRIMARY_METRIC}_mean": model_rows[0][f"{PRIMARY_METRIC}_mean"],
            f"{PRIMARY_METRIC}_sd_across_resolutions": model_rows[0][
                f"{PRIMARY_METRIC}_sd_across_resolutions"
            ],
        },
        "artifact_sha256": {
            "input_matrix": sha256_file(matrix_path),
            unit_path.name: sha256_file(unit_path),
            model_path.name: sha256_file(model_path),
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "PHASE2_OOF_LEADERBOARD_PASS "
        + json.dumps(
            {
                "units": len(unit_rows),
                "models": len(model_rows),
                "primary_metric": PRIMARY_METRIC,
                "selection_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
