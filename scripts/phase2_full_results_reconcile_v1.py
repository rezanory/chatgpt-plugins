from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib

import phase2_final_evidence_extract_v1 as base
import phase2_full_results_recovery_v1 as recovery

SOURCE_RUN_ID = "37133485951"

PINNED_OUTPUTS = {
    "master": (
        "azadka/phase2-final-evidence-master-37133485951",
        "PHASE2_FULL_RESULTS_MASTER.json",
    ),
    "kg-02": (
        "radlinaradlina/phase2-final-evidence-kg-02-37133485951",
        "PHASE2_FULL_RESULTS_KG_02.json",
    ),
    "kg-03": (
        "rezanory/phase2-final-evidence-kg-03-37133485951",
        "PHASE2_FULL_RESULTS_KG_03.json",
    ),
    "kg-04": (
        "reyhanehazad/phase2-final-evidence-kg-04-37133485951",
        "PHASE2_FULL_RESULTS_KG_04.json",
    ),
    "kg-05": (
        "trickermark/phase2-final-evidence-kg-05-37133485951",
        "PHASE2_FULL_RESULTS_KG_05.json",
    ),
    "kg-06": (
        "msdenis/phase2-final-evidence-kg-06-37133485951",
        "PHASE2_FULL_RESULTS_KG_06.json",
    ),
    "kg-07": (
        "nisabulutmark/phase2-final-evidence-kg-07-37133485951",
        "PHASE2_FULL_RESULTS_KG_07.json",
    ),
    "kg-08": (
        "azadkk/phase2-final-evidence-kg-08-37133485951",
        "PHASE2_FULL_RESULTS_KG_08.json",
    ),
    "kg-09": (
        "mylovevpn1/phase2-final-evidence-kg-09-37133485951",
        "PHASE2_FULL_RESULTS_KG_09.json",
    ),
    "kg-10": (
        "computstu1/phase2-final-evidence-kg-10-37133485951",
        "PHASE2_FULL_RESULTS_KG_10.json",
    ),
    "kg-11": (
        "jobreza1/phase2-final-evidence-kg-11-37133485951",
        "PHASE2_FULL_RESULTS_KG_11.json",
    ),
}

M07_R224_LEGACY_DATASET_REF = "rezanory/m07-final-5fold-fix2-d260914d"
M07_R224_LEGACY_DATASET_VERSION = 108
M07_R224_EXPECTED_SPLIT = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
M07_R224_EXPECTED_RECIPE = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"
M07_R224_LEGACY_FILES = [
    f"M07_FOLD_{fold}_RECOVERY/FINAL_5FOLD/fold_{fold}/COMPLETED.json"
    for fold in range(1, 6)
]


def fetch_account(read_token: str, account_id: str) -> dict:
    kernel_ref, file_name = PINNED_OUTPUTS[account_id]
    response = base.post_json(
        base.READ_ENDPOINT,
        read_token,
        {
            "action": "output_json_files",
            "account_id": account_id,
            "kernel_ref": kernel_ref,
            "file_names": [file_name],
            "max_bytes_per_file": 262144,
        },
        timeout=180,
    )
    if not response.get("ok"):
        raise RuntimeError(
            f"RECONCILE_OUTPUT_FETCH_FAILED:{account_id}:{response}"
        )
    result = base._read_payload(response)
    files = result.get("files") or []
    if (
        len(files) != 1
        or not isinstance(files[0], dict)
        or not isinstance(files[0].get("json"), dict)
    ):
        raise RuntimeError(f"RECONCILE_OUTPUT_SHAPE_INVALID:{account_id}")
    value = files[0]["json"]
    if (
        value.get("schema") != "pneumonia.phase2.full_results.account.v1"
        or value.get("status") != "PASS"
        or value.get("account_id") != account_id
    ):
        raise RuntimeError(f"RECONCILE_ACCOUNT_RECEIPT_INVALID:{account_id}")
    return value


def fetch_m07_r224_legacy_folds(read_token: str) -> list[dict]:
    response = base.post_json(
        base.READ_ENDPOINT,
        read_token,
        {
            "action": "dataset_json_files",
            "account_id": "kg-03",
            "dataset_ref": M07_R224_LEGACY_DATASET_REF,
            "dataset_version_number": M07_R224_LEGACY_DATASET_VERSION,
            "file_names": M07_R224_LEGACY_FILES,
            "max_bytes_per_file": 262144,
        },
        timeout=180,
    )
    if not response.get("ok"):
        raise RuntimeError("M07_R224_LEGACY_FETCH_FAILED:" + str(response))
    result = base._read_payload(response)
    files = result.get("files") or []
    if len(files) != 5:
        raise RuntimeError(f"M07_R224_LEGACY_FILE_COUNT_INVALID:{len(files)}")

    by_fold: dict[int, dict] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("json"), dict):
            raise RuntimeError("M07_R224_LEGACY_FILE_SHAPE_INVALID")
        receipt = item["json"]
        fold_id = int(receipt.get("fold_id") or 0)
        contract = receipt.get("run_contract") or {}
        artifacts = receipt.get("artifact_sha256") or {}
        metrics = receipt.get("metrics")
        if (
            receipt.get("schema") != "m07.final.fold.v1.6"
            or receipt.get("status") != "COMPLETED"
            or fold_id not in range(1, 6)
            or receipt.get("split_fingerprint") != M07_R224_EXPECTED_SPLIT
            or receipt.get("hpo_recipe_fingerprint") != M07_R224_EXPECTED_RECIPE
            or receipt.get("locked_test_used") is not False
            or contract.get("stage") != "M07_FINAL_FOLD"
            or contract.get("model_id") != "M07"
            or int(contract.get("resolution") or 0) != 224
            or int(contract.get("fold_id") or 0) != fold_id
            or int(contract.get("head_epochs") or 0) != 3
            or int(contract.get("finetune_epochs") or 0) != 25
            or not isinstance(metrics, dict)
            or int(metrics.get("n") or 0) < 1
        ):
            raise RuntimeError(f"M07_R224_LEGACY_RECEIPT_INVALID:{fold_id}")
        required_artifacts = (
            "validation_predictions",
            "history",
            "weights",
            "run_result",
        )
        bad_hashes = [
            name
            for name in required_artifacts
            if not isinstance(artifacts.get(name), str)
            or len(artifacts[name]) != 64
        ]
        if bad_hashes:
            raise RuntimeError(
                f"M07_R224_LEGACY_ARTIFACT_HASH_INVALID:{fold_id}:"
                + ",".join(bad_hashes)
            )
        if fold_id in by_fold:
            raise RuntimeError(f"M07_R224_LEGACY_DUPLICATE_FOLD:{fold_id}")
        by_fold[fold_id] = {
            "fold_id": fold_id,
            "schema": receipt["schema"],
            "status": receipt["status"],
            "receipt_sha256": receipt.get("receipt_sha256"),
            "run_fingerprint": receipt.get("run_fingerprint"),
            "locked_test_used_for_training": False,
            "external_used_for_training": False,
            "train_metrics": None,
            "validation_metrics": metrics,
            "training_evidence": {
                "evidence_class": "LEGACY_TRAINING_HISTORY_AND_FROZEN_WEIGHTS",
                "train_classification_metrics_persisted": False,
                "head_epochs_budget": 3,
                "finetune_epochs_budget": 25,
                "history_sha256": artifacts["history"],
                "weights_sha256": artifacts["weights"],
                "run_result_sha256": artifacts["run_result"],
                "validation_predictions_sha256": artifacts["validation_predictions"],
            },
            "source_file_name": item.get("file_name"),
            "source_file_sha256": item.get("sha256"),
        }
    if set(by_fold) != {1, 2, 3, 4, 5}:
        raise RuntimeError("M07_R224_LEGACY_FOLD_COVERAGE_INVALID")
    return [by_fold[fold] for fold in range(1, 6)]


def build_matrix(
    account_results: list[dict],
    run_id: str,
    legacy_m07_folds: list[dict],
) -> dict:
    if len(account_results) != 11:
        raise RuntimeError(
            f"RECONCILE_ACCOUNT_COUNT_INVALID:{len(account_results)}"
        )
    accounts = [str(row.get("account_id")) for row in account_results]
    if set(accounts) != set(PINNED_OUTPUTS) or len(set(accounts)) != 11:
        raise RuntimeError("RECONCILE_ACCOUNT_COVERAGE_INVALID")

    units = []
    for row in account_results:
        units.extend(row.get("targets") or [])

    expected = {
        (f"M{index:02d}", resolution)
        for index in range(1, 13)
        for resolution in (224, 320, 384)
    }
    observed = {
        (str(row.get("model_id")), int(row.get("resolution") or 0))
        for row in units
        if isinstance(row, dict)
    }
    if observed != expected or len(units) != 36:
        raise RuntimeError(
            "RECONCILE_UNIT_COVERAGE_INVALID="
            + json.dumps(
                {
                    "missing": sorted(expected - observed),
                    "extra": sorted(observed - expected),
                    "count": len(units),
                },
                sort_keys=True,
            )
        )

    fold_count = 0
    train_metric_folds = 0
    validation_metric_folds = 0
    training_evidence_folds = 0
    legacy_training_history_folds = 0
    locked_present = 0
    locked_missing = []
    enriched_units = []

    for unit in sorted(
        units,
        key=lambda row: (str(row["model_id"]), int(row["resolution"])),
    ):
        unit_key = (str(unit["model_id"]), int(unit["resolution"]))
        folds = recovery._fold_receipts(unit)
        if unit_key == ("M07", 224):
            if folds:
                raise RuntimeError("M07_R224_UNEXPECTED_STANDARD_FOLD_RECEIPTS")
            if len(legacy_m07_folds) != 5:
                raise RuntimeError(
                    f"M07_R224_LEGACY_FOLD_COUNT_INVALID:{len(legacy_m07_folds)}"
                )
            folds = legacy_m07_folds

        fold_count += len(folds)
        train_metric_folds += sum(
            isinstance(row.get("train_metrics"), dict) for row in folds
        )
        validation_metric_folds += sum(
            isinstance(row.get("validation_metrics"), dict) for row in folds
        )
        for row in folds:
            if isinstance(row.get("train_metrics"), dict) or isinstance(
                row.get("training_evidence"), dict
            ):
                training_evidence_folds += 1
            if (
                (row.get("training_evidence") or {}).get("evidence_class")
                == "LEGACY_TRAINING_HISTORY_AND_FROZEN_WEIGHTS"
            ):
                legacy_training_history_folds += 1

        locked = recovery._locked_test_summary(unit)
        if locked["status"] == "PRESENT":
            locked_present += 1
        else:
            locked_missing.append(
                {
                    "model_id": str(unit["model_id"]),
                    "resolution": int(unit["resolution"]),
                }
            )
        enriched = dict(unit)
        enriched["recovered_folds"] = folds
        enriched["locked_test_recovery"] = locked
        enriched_units.append(enriched)

    if fold_count != 180:
        raise RuntimeError(f"RECONCILE_FOLD_COUNT_INVALID:{fold_count}")
    if training_evidence_folds != 180:
        raise RuntimeError(
            f"RECONCILE_TRAINING_EVIDENCE_INCOMPLETE:{training_evidence_folds}"
        )
    if train_metric_folds + legacy_training_history_folds != 180:
        raise RuntimeError(
            "RECONCILE_TRAIN_METRIC_AND_LEGACY_EVIDENCE_COVERAGE_INVALID:"
            f"{train_metric_folds}:{legacy_training_history_folds}"
        )
    if legacy_training_history_folds != 5:
        raise RuntimeError(
            f"RECONCILE_LEGACY_TRAINING_HISTORY_COUNT_INVALID:"
            f"{legacy_training_history_folds}"
        )
    if validation_metric_folds != 180:
        raise RuntimeError(
            f"RECONCILE_VALIDATION_METRICS_INCOMPLETE:{validation_metric_folds}"
        )

    finalists = {("M09", 384), ("M10", 320)}
    index = {
        (str(row["model_id"]), int(row["resolution"])): row
        for row in enriched_units
    }
    for key in finalists:
        locked = index[key]["locked_test_recovery"]
        if locked.get("status") != "PRESENT":
            raise RuntimeError(
                f"RECONCILE_FINALIST_LOCKED_TEST_MISSING:{key[0]}:{key[1]}"
            )
        primary = locked.get("primary")
        metrics = primary.get("metrics") if isinstance(primary, dict) else None
        if not isinstance(metrics, dict):
            raise RuntimeError(
                f"RECONCILE_FINALIST_LOCKED_TEST_METRICS_MISSING:{key[0]}:{key[1]}"
            )
        required = {
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
        }
        missing = sorted(required - set(metrics))
        if missing:
            raise RuntimeError(
                f"RECONCILE_FINALIST_LOCKED_TEST_METRICS_INCOMPLETE:"
                f"{key[0]}:{key[1]}:" + ",".join(missing)
            )

    launches = [
        {
            "account_id": account_id,
            "kernel_ref": PINNED_OUTPUTS[account_id][0],
            "status": "COMPLETE",
            "reused": True,
            "source_run_id": SOURCE_RUN_ID,
        }
        for account_id in sorted(PINNED_OUTPUTS)
    ]

    return {
        "schema": "pneumonia.phase2.full_results.matrix.v1",
        "status": "PASS",
        "github_run_id": str(run_id),
        "source_extractor_run_id": SOURCE_RUN_ID,
        "accounts_complete": 11,
        "model_resolution_units": 36,
        "folds_represented": 180,
        "train_metric_folds": train_metric_folds,
        "legacy_training_history_folds": legacy_training_history_folds,
        "training_evidence_folds": training_evidence_folds,
        "validation_metric_folds": validation_metric_folds,
        "locked_test_units_recovered": locked_present,
        "locked_test_units_not_recovered": locked_missing,
        "extraction_only": True,
        "training_performed": False,
        "inference_performed": False,
        "threshold_tuning_performed": False,
        "locked_test_executed_by_this_run": False,
        "external_validation_executed_by_this_run": False,
        "reconciled_from_existing_outputs": True,
        "legacy_training_metric_limitation": {
            "model_id": "M07",
            "resolution": 224,
            "folds": [1, 2, 3, 4, 5],
            "train_classification_metrics_persisted": False,
            "available_training_evidence": "HASHED_HISTORY_AND_FROZEN_WEIGHTS",
            "posthoc_train_inference_performed": False,
        },
        "launches": launches,
        "units": enriched_units,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not str(args.run_id).isdigit():
        raise SystemExit("RUN_ID_INVALID")
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(read_token) < 100:
        raise SystemExit("OIDC_TOKEN_INVALID")

    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
        futures = {
            pool.submit(fetch_account, read_token, account_id): account_id
            for account_id in PINNED_OUTPUTS
        }
        account_results = []
        for future in concurrent.futures.as_completed(futures):
            account_id = futures[future]
            value = future.result()
            account_results.append(value)
            print(
                "PHASE2_FULL_RESULTS_RECONCILE_ACCOUNT_PASS",
                json.dumps({"account_id": account_id}, sort_keys=True),
                flush=True,
            )

    account_results.sort(key=lambda row: str(row.get("account_id")))
    legacy_m07_folds = fetch_m07_r224_legacy_folds(read_token)
    print(
        "PHASE2_M07_R224_LEGACY_TRAINING_EVIDENCE_PASS",
        json.dumps({"folds": len(legacy_m07_folds)}, sort_keys=True),
        flush=True,
    )
    matrix = build_matrix(account_results, str(args.run_id), legacy_m07_folds)
    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PHASE2_FULL_RESULTS_RECONCILE_PASS",
        json.dumps(
            {
                "accounts": matrix["accounts_complete"],
                "units": matrix["model_resolution_units"],
                "folds": matrix["folds_represented"],
                "train_metric_folds": matrix["train_metric_folds"],
                "legacy_training_history_folds": matrix[
                    "legacy_training_history_folds"
                ],
                "training_evidence_folds": matrix["training_evidence_folds"],
                "validation_metric_folds": matrix["validation_metric_folds"],
                "locked_test_units_recovered": matrix[
                    "locked_test_units_recovered"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
