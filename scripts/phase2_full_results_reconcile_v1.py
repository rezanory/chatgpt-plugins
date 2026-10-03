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


def build_matrix(account_results: list[dict], run_id: str) -> dict:
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
    locked_present = 0
    locked_missing = []
    enriched_units = []

    for unit in sorted(
        units,
        key=lambda row: (str(row["model_id"]), int(row["resolution"])),
    ):
        folds = recovery._fold_receipts(unit)
        fold_count += len(folds)
        train_metric_folds += sum(
            isinstance(row.get("train_metrics"), dict) for row in folds
        )
        validation_metric_folds += sum(
            isinstance(row.get("validation_metrics"), dict) for row in folds
        )
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
    if train_metric_folds != 180:
        raise RuntimeError(
            f"RECONCILE_TRAIN_METRICS_INCOMPLETE:{train_metric_folds}"
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
        required = {"n", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall"}
        missing = sorted(required - set(metrics))
        if missing:
            raise RuntimeError(
                f"RECONCILE_FINALIST_LOCKED_TEST_METRICS_INCOMPLETE:{key[0]}:{key[1]}:"
                + ",".join(missing)
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
    matrix = build_matrix(account_results, str(args.run_id))
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
                "validation_metric_folds": matrix["validation_metric_folds"],
                "locked_test_units_recovered": matrix["locked_test_units_recovered"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
