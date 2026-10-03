from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import zipfile


EXPECTED_CANDIDATES = {
    ("M09", 384),
    ("M10", 320),
}
EXTERNAL_SCHEMA = "pneumonia.phase2.finalist.external.rsna_pediatric.terminal.v1"
HANDOFF_SCHEMA = "pneumonia.phase2.finalist.external.github_handoff.v1"


class ClosureError(RuntimeError):
    pass


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClosureError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def verify_selection_freeze(path: pathlib.Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ClosureError("SELECTION_FREEZE_INVALID")
    expected = {
        "schema": "pneumonia.phase2.selection_freeze.v1",
        "status": "FROZEN_FOR_REPORT_ONLY_CONFIRMATION",
        "single_winner_declared": False,
        "confirmatory_superiority_claim": False,
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "locked_test_may_reselect": False,
        "external_may_reselect": False,
        "thresholds_refit_after_freeze": False,
        "cross_model_ensemble_selected": False,
        "cross_model_ensemble_fitted": False,
    }
    bad = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if bad:
        raise ClosureError(
            "SELECTION_FREEZE_POLICY_INVALID=" + json.dumps(bad, sort_keys=True)
        )
    metrics = ((value.get("selection_basis") or {}).get("requested_metrics") or [])
    if metrics != ["macro_precision", "macro_recall"]:
        raise ClosureError("SELECTION_FREEZE_METRICS_INVALID")
    candidates = {
        (str(row.get("model_id")), int(row.get("resolution") or 0))
        for row in value.get("frozen_candidate_set") or []
        if isinstance(row, dict)
    }
    if candidates != EXPECTED_CANDIDATES:
        raise ClosureError(
            "SELECTION_FREEZE_CANDIDATES_INVALID="
            + json.dumps(sorted(candidates), sort_keys=True)
        )
    return value, sha256_bytes(raw)


def _unit_key(unit: dict) -> tuple[str, int]:
    return str(unit.get("model_id")), int(unit.get("resolution") or 0)


def verify_full_results(path: pathlib.Path) -> tuple[dict, dict]:
    value = load_json(path)
    expected = {
        "schema": "pneumonia.phase2.full_results.matrix.v1",
        "status": "PASS",
        "model_resolution_units": 36,
        "folds_represented": 180,
        "validation_metric_folds": 180,
        "extraction_only": True,
        "training_performed": False,
        "inference_performed": False,
        "threshold_tuning_performed": False,
        "locked_test_executed_by_this_run": False,
        "external_validation_executed_by_this_run": False,
    }
    bad = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if bad:
        raise ClosureError(
            "FULL_RESULTS_CONTRACT_INVALID=" + json.dumps(bad, sort_keys=True)
        )
    if int(value.get("train_metric_folds") or 0) != 180:
        raise ClosureError(
            "FULL_RESULTS_TRAIN_METRICS_INCOMPLETE:"
            + str(value.get("train_metric_folds"))
        )
    units = value.get("units") or []
    if len(units) != 36:
        raise ClosureError("FULL_RESULTS_UNIT_LIST_INVALID")
    index = {_unit_key(row): row for row in units if isinstance(row, dict)}
    expected_units = {
        (f"M{model:02d}", resolution)
        for model in range(1, 13)
        for resolution in (224, 320, 384)
    }
    if set(index) != expected_units:
        raise ClosureError("FULL_RESULTS_UNIT_COVERAGE_INVALID")
    locked = {}
    for key in EXPECTED_CANDIDATES:
        unit = index[key]
        recovery = unit.get("locked_test_recovery") or {}
        if recovery.get("status") != "PRESENT":
            raise ClosureError(
                f"FINALIST_LOCKED_TEST_NOT_RECOVERED:{key[0]}:{key[1]}"
            )
        primary = recovery.get("primary")
        if not isinstance(primary, dict):
            raise ClosureError(
                f"FINALIST_LOCKED_TEST_PRIMARY_MISSING:{key[0]}:{key[1]}"
            )
        metrics = primary.get("metrics")
        if not isinstance(metrics, dict):
            raise ClosureError(
                f"FINALIST_LOCKED_TEST_METRICS_MISSING:{key[0]}:{key[1]}"
            )
        required_metrics = (
            "macro_precision",
            "macro_recall",
            "accuracy",
            "balanced_accuracy",
        )
        missing = [name for name in required_metrics if name not in metrics]
        if missing:
            raise ClosureError(
                f"FINALIST_LOCKED_TEST_METRICS_INCOMPLETE:{key[0]}:{key[1]}:"
                + ",".join(missing)
            )
        locked[key] = {
            "source": primary.get("source"),
            "path": primary.get("path"),
            "metrics": metrics,
        }
    return value, locked


def _find_exactly_one(root: pathlib.Path, pattern: str) -> pathlib.Path:
    rows = [path for path in root.rglob(pattern) if path.is_file()]
    if len(rows) != 1:
        raise ClosureError(
            f"ARTIFACT_FILE_COUNT_INVALID:{pattern}:{len(rows)}"
        )
    return rows[0]


def _read_report_from_zip(zip_path: pathlib.Path, model_id: str) -> tuple[dict, str]:
    expected_name = f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_REPORT.json"
    with zipfile.ZipFile(zip_path) as archive:
        matches = [
            name for name in archive.namelist()
            if pathlib.PurePosixPath(name).name == expected_name
        ]
        if len(matches) != 1:
            raise ClosureError(
                f"EXTERNAL_REPORT_MEMBER_COUNT_INVALID:{model_id}:{len(matches)}"
            )
        raw = archive.read(matches[0])
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ClosureError(f"EXTERNAL_REPORT_INVALID:{model_id}")
    return value, sha256_bytes(raw)


def verify_external_root(
    root: pathlib.Path,
    selection_freeze_sha256: str,
) -> dict:
    rows = {}
    for model_id, resolution in sorted(EXPECTED_CANDIDATES):
        handoff_path = _find_exactly_one(
            root, f"{model_id}_R{resolution}_RSNA_HANDOFF.json"
        )
        receipt_path = _find_exactly_one(
            root, f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_TERMINAL_RECEIPT.json"
        )
        zip_path = _find_exactly_one(
            root, f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_R{resolution}_V1_COMPLETE.zip"
        )
        handoff = load_json(handoff_path)
        receipt = load_json(receipt_path)

        expected_handoff = {
            "schema": HANDOFF_SCHEMA,
            "status": "SCIENTIFIC_RECEIPT_PASS",
            "model_id": model_id,
            "resolution": resolution,
            "selection_freeze_sha256": selection_freeze_sha256,
            "training_performed": False,
            "hpo_performed": False,
            "external_threshold_tuning": False,
            "external_adaptation": False,
            "external_calibration_fitting": False,
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
            "external_may_reselect": False,
        }
        bad = {
            key: {"expected": wanted, "actual": handoff.get(key)}
            for key, wanted in expected_handoff.items()
            if handoff.get(key) != wanted
        }
        if bad:
            raise ClosureError(
                f"EXTERNAL_HANDOFF_INVALID:{model_id}:"
                + json.dumps(bad, sort_keys=True)
            )

        expected_receipt = {
            "schema": EXTERNAL_SCHEMA,
            "status": "SCIENTIFIC_RECEIPT_PASS",
            "model_id": model_id,
            "resolution": resolution,
            "selection_freeze_sha256": selection_freeze_sha256,
            "training_performed": False,
            "hpo_performed": False,
            "external_threshold_tuning": False,
            "external_adaptation": False,
            "external_calibration_fitting": False,
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
            "external_may_reselect": False,
        }
        bad = {
            key: {"expected": wanted, "actual": receipt.get(key)}
            for key, wanted in expected_receipt.items()
            if receipt.get(key) != wanted
        }
        if bad:
            raise ClosureError(
                f"EXTERNAL_RECEIPT_INVALID:{model_id}:"
                + json.dumps(bad, sort_keys=True)
            )
        claimed = str(receipt.get("receipt_sha256") or "")
        body = dict(receipt)
        body.pop("receipt_sha256", None)
        if claimed != sha256_bytes(canonical_json(body)):
            raise ClosureError(f"EXTERNAL_RECEIPT_SHA_MISMATCH:{model_id}")
        if handoff.get("terminal_receipt_sha256") != claimed:
            raise ClosureError(f"EXTERNAL_HANDOFF_RECEIPT_SHA_MISMATCH:{model_id}")
        zip_sha = sha256_file(zip_path)
        if handoff.get("complete_zip_sha256") != zip_sha:
            raise ClosureError(f"EXTERNAL_ZIP_SHA_MISMATCH:{model_id}")

        report, report_sha = _read_report_from_zip(zip_path, model_id)
        expected_report = {
            "schema": "pneumonia.phase2.finalist.external.rsna_pediatric.resolution.v1",
            "status": "SCIENTIFIC_RECEIPT_PASS",
            "model_id": model_id,
            "resolution": resolution,
            "selection_freeze_sha256": selection_freeze_sha256,
            "scientific_classification": "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION",
            "locked_test_used_for_selection": False,
            "external_used_for_selection": False,
            "external_may_reselect": False,
            "training_performed": False,
            "hpo_performed": False,
            "external_threshold_tuning": False,
            "external_adaptation": False,
            "external_calibration_fitting": False,
        }
        bad = {
            key: {"expected": wanted, "actual": report.get(key)}
            for key, wanted in expected_report.items()
            if report.get(key) != wanted
        }
        if bad:
            raise ClosureError(
                f"EXTERNAL_REPORT_POLICY_INVALID:{model_id}:"
                + json.dumps(bad, sort_keys=True)
            )

        primary = ((report.get("primary_pediatric_lt10") or {}).get("metrics"))
        expanded = ((report.get("expanded_pediatric_le18") or {}).get("metrics"))
        if not isinstance(primary, dict) or not isinstance(expanded, dict):
            raise ClosureError(f"EXTERNAL_METRICS_MISSING:{model_id}")
        for cohort_name, metrics in (("primary", primary), ("expanded", expanded)):
            for metric in ("macro_precision", "macro_recall", "accuracy"):
                if metric not in metrics:
                    raise ClosureError(
                        f"EXTERNAL_METRIC_MISSING:{model_id}:{cohort_name}:{metric}"
                    )

        rows[(model_id, resolution)] = {
            "handoff_sha256": sha256_file(handoff_path),
            "receipt_sha256": claimed,
            "complete_zip_sha256": zip_sha,
            "report_sha256": report_sha,
            "kernel_ref": handoff.get("kernel_ref"),
            "compute_mode": handoff.get("compute_mode"),
            "external_dataset": handoff.get("external_dataset"),
            "external_manifest_sha256": handoff.get("external_manifest_sha256"),
            "primary_pediatric_lt10": primary,
            "expanded_pediatric_le18": expanded,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-freeze", required=True)
    parser.add_argument("--full-results", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full-results-run-id", required=True)
    parser.add_argument("--external-run-id", required=True)
    args = parser.parse_args()

    if not str(args.full_results_run_id).isdigit():
        raise SystemExit("FULL_RESULTS_RUN_ID_INVALID")
    if not str(args.external_run_id).isdigit():
        raise SystemExit("EXTERNAL_RUN_ID_INVALID")

    selection_path = pathlib.Path(args.selection_freeze)
    full_path = pathlib.Path(args.full_results)
    external_root = pathlib.Path(args.external_root)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    freeze, freeze_sha = verify_selection_freeze(selection_path)
    full, locked = verify_full_results(full_path)
    external = verify_external_root(external_root, freeze_sha)

    candidate_rows = []
    for item in freeze["frozen_candidate_set"]:
        key = (str(item["model_id"]), int(item["resolution"]))
        candidate_rows.append(
            {
                "model_id": key[0],
                "resolution": key[1],
                "development_oof": {
                    "n": item["n"],
                    "macro_precision": item["macro_precision"],
                    "macro_recall": item["macro_recall"],
                    "tn": item["tn"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "tp": item["tp"],
                },
                "locked_test_report_only": locked[key],
                "external_report_only": external[key],
            }
        )

    closure = {
        "schema": "pneumonia.phase2.final_scientific_closure.v1",
        "status": "CLOSED_PASS",
        "scientific_classification": "POST_RESULTS_EXPLORATORY_DUAL_METRIC_FINAL_CLOSURE",
        "models": 12,
        "resolutions_per_model": 3,
        "model_resolution_units": 36,
        "training_folds": 180,
        "training_metrics_recovered_folds": int(full["train_metric_folds"]),
        "validation_metrics_recovered_folds": int(full["validation_metric_folds"]),
        "selection_metric_pair": ["macro_precision", "macro_recall"],
        "selection_rule": freeze["freeze_rule"],
        "frozen_candidate_count": len(candidate_rows),
        "single_winner_declared": False,
        "confirmatory_superiority_claim": False,
        "locked_test_role": "REPORT_ONLY_AFTER_SELECTION_FREEZE",
        "external_validation_role": "REPORT_ONLY_AFTER_SELECTION_FREEZE",
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "reselection_after_locked_test": False,
        "reselection_after_external_validation": False,
        "threshold_refit_after_freeze": False,
        "training_after_freeze": False,
        "hpo_after_freeze": False,
        "cross_model_ensemble_selected": False,
        "source_runs": {
            "dual_metric_oof": str(
                (freeze.get("selection_basis") or {}).get("source_dual_run_id")
            ),
            "full_results_recovery": str(args.full_results_run_id),
            "finalist_external_validation": str(args.external_run_id),
        },
        "source_hashes": {
            "selection_freeze_sha256": freeze_sha,
            "full_results_matrix_sha256": sha256_file(full_path),
        },
        "frozen_candidates": candidate_rows,
        "remaining_gate": "NONE",
    }
    body = canonical_json(closure)
    closure["closure_sha256"] = sha256_bytes(body)
    closure_path = out / "PHASE2_FINAL_SCIENTIFIC_CLOSURE_V1.json"
    closure_path.write_text(
        json.dumps(closure, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": "pneumonia.phase2.final_scientific_closure.manifest.v1",
        "status": "PASS",
        "closure_sha256": closure["closure_sha256"],
        "files": {
            closure_path.name: sha256_file(closure_path),
            selection_path.name: sha256_file(selection_path),
            full_path.name: sha256_file(full_path),
        },
    }
    manifest_path = out / "PHASE2_FINAL_SCIENTIFIC_CLOSURE_MANIFEST_V1.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "PHASE2_FINAL_SCIENTIFIC_CLOSURE_PASS",
        json.dumps(
            {
                "status": closure["status"],
                "training_folds": closure["training_folds"],
                "frozen_candidates": [
                    [row["model_id"], row["resolution"]]
                    for row in candidate_rows
                ],
                "remaining_gate": closure["remaining_gate"],
                "closure_sha256": closure["closure_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
