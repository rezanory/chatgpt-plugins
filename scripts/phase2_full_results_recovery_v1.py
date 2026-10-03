from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import time

import phase2_final_evidence_extract_v1 as base

EXTRACTOR_GENERATION = "full-results-v1"
_BASE_KERNEL_SCRIPT = base.kernel_script


def _output_name(account_id: str) -> str:
    return "PHASE2_FULL_RESULTS_" + account_id.replace("-", "_").upper() + ".json"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"PATCH_BOUNDARY_INVALID:{label}:{source.count(old)}")
    return source.replace(old, new, 1)


def kernel_script(account_id: str, targets: list[dict]) -> str:
    source = _BASE_KERNEL_SCRIPT(account_id, targets)

    source = _replace_once(
        source,
        '''def candidate_name(name):
    base=Path(name).name.upper()
    if not base.endswith(".JSON"):
        return False
    return base=="FINAL_REPORT.JSON" or ("OOF" in base and "METRIC" in base)
''',
        '''def candidate_name(name):
    base=Path(name).name.upper()
    if not base.endswith(".JSON"):
        return False
    if base=="FINAL_REPORT.JSON":
        return True
    if "OOF" in base and "METRIC" in base:
        return True
    if "LOCKED_TEST" in base and ("METRIC" in base or "BOOTSTRAP_CI95" in base):
        return True
    if "GENERALIZATION_GAP" in base:
        return True
    return False
''',
        "candidate_name",
    )

    source = _replace_once(
        source,
        'if is_oof and len(hit)>=2 and len(metric_blocks)<4:',
        'if len(hit)>=2 and len(metric_blocks)<10:',
        "metric_block_scope",
    )

    source = _replace_once(
        source,
        '            ][:8]',
        '            ][:16]',
        "archive_member_bound",
    )

    source = _replace_once(
        source,
        '''                "external_used_for_training":receipt.get("external_used_for_training"),
            }
''',
        '''                "external_used_for_training":receipt.get("external_used_for_training"),
                "train_metrics":receipt.get("train_metrics"),
                "validation_metrics":receipt.get("validation_metrics"),
                "training_evidence":receipt.get("training_evidence"),
            }
''',
        "fold_metric_receipt",
    )

    direct_scan = '''    direct_by_fold={}
    for path in files:
        if path.name!="COMPLETED.json":
            continue
        try:
            receipt=read_json_file(path)
        except Exception:
            continue
        if (
            receipt.get("schema")!="pneumonia.phase2.fold.v1.7"
            or receipt.get("status")!="COMPLETED"
            or str(receipt.get("model_id") or "")!=model_id
            or int(receipt.get("resolution") or 0)!=resolution
        ):
            continue
        fold_id=int(receipt.get("fold_id") or 0)
        if fold_id not in (1,2,3,4,5):
            continue
        record={
            "fold_id":fold_id,
            "path":path.relative_to(state_root).as_posix(),
            "sha256":sha256_file(path),
            "receipt_sha256":receipt.get("receipt_sha256"),
            "run_fingerprint":receipt.get("run_fingerprint"),
            "locked_test_used_for_training":receipt.get("locked_test_used_for_training"),
            "external_used_for_training":receipt.get("external_used_for_training"),
            "train_metrics":receipt.get("train_metrics"),
            "validation_metrics":receipt.get("validation_metrics"),
            "training_evidence":receipt.get("training_evidence"),
        }
        previous=direct_by_fold.get(fold_id)
        if previous is not None:
            if (
                previous.get("receipt_sha256")!=record.get("receipt_sha256")
                or previous.get("run_fingerprint")!=record.get("run_fingerprint")
            ):
                raise RuntimeError("DIRECT_FOLD_RECEIPT_CONFLICT:"+dataset_ref+":"+str(fold_id))
            if len(record["path"])<len(previous["path"]):
                direct_by_fold[fold_id]=record
        else:
            direct_by_fold[fold_id]=record
    row["direct_fold_receipts"]=[direct_by_fold[k] for k in sorted(direct_by_fold)]

'''
    source = _replace_once(
        source,
        '''    for path in files:
        if candidate_name(path.name) and path.name!="CAMPAIGN_STATE.json":
''',
        direct_scan + '''    for path in files:
        if candidate_name(path.name) and path.name!="CAMPAIGN_STATE.json":
''',
        "direct_fold_scan",
    )

    source = _replace_once(
        source,
        '''        for path in files:
            if path.suffix.lower()==".json" and ("REPORT" in path.name.upper() or "OOF" in path.name.upper()):
''',
        '''        for path in files:
            if path.suffix.lower()==".json" and (
                "REPORT" in path.name.upper()
                or "OOF" in path.name.upper()
                or "LOCKED" in path.name.upper()
            ):
''',
        "legacy_m07_source_scan",
    )

    source = _replace_once(
        source,
        '''    for source_name,value in sources[:6]:
''',
        '''    def source_priority(item):
        label=item[0].upper()
        if "LOCKED_TEST_PRIMARY_METRICS" in label:
            return (0,label)
        if "FINAL_REPORT" in label:
            return (1,label)
        if "GENERALIZATION_GAP" in label:
            return (2,label)
        if "OOF" in label and "METRIC" in label:
            return (3,label)
        if "LOCKED_TEST" in label:
            return (4,label)
        return (5,label)
    sources.sort(key=source_priority)
    for source_name,value in sources[:16]:
''',
        "source_priority",
    )

    source = _replace_once(
        source,
        '"schema":"pneumonia.phase2.final_evidence.extract.account.v1",',
        '"schema":"pneumonia.phase2.full_results.account.v1",',
        "account_schema",
    )
    source = _replace_once(
        source,
        'name="PHASE2_FINAL_EVIDENCE_"+ACCOUNT_ID.replace("-","_").upper()+".json"',
        'name="PHASE2_FULL_RESULTS_"+ACCOUNT_ID.replace("-","_").upper()+".json"',
        "output_name",
    )
    source = _replace_once(
        source,
        'print("PHASE2_FINAL_EVIDENCE_EXTRACT_PASS "+json.dumps({',
        'print("PHASE2_FULL_RESULTS_RECOVERY_PASS "+json.dumps({',
        "pass_marker",
    )
    return source


def _configure_base() -> None:
    base.EXTRACTOR_GENERATION = EXTRACTOR_GENERATION
    base.kernel_script = kernel_script
    base._extractor_output_name = _output_name


def _fold_receipts(unit: dict) -> list[dict]:
    by_fold: dict[int, dict] = {}
    for archive in unit.get("fold_archives") or []:
        receipt = archive.get("receipt") if isinstance(archive, dict) else None
        if not isinstance(receipt, dict):
            continue
        fold_id = int(receipt.get("fold_id") or 0)
        if fold_id in range(1, 6):
            by_fold[fold_id] = receipt
    for receipt in unit.get("direct_fold_receipts") or []:
        if not isinstance(receipt, dict):
            continue
        fold_id = int(receipt.get("fold_id") or 0)
        if fold_id in range(1, 6):
            current = by_fold.get(fold_id)
            if current is None or not isinstance(current.get("train_metrics"), dict):
                by_fold[fold_id] = receipt
    return [by_fold[k] for k in sorted(by_fold)]


def _metric_block_has_locked_test(source: dict) -> bool:
    source_name = str(source.get("source") or "").upper()
    if "LOCKED_TEST" in source_name:
        return True
    for block in source.get("metric_blocks") or []:
        if "LOCKED" in str(block.get("path") or "").upper():
            return True
    return False


def _locked_test_summary(unit: dict) -> dict:
    candidates = []
    for source in unit.get("json_sources") or []:
        if not isinstance(source, dict) or not _metric_block_has_locked_test(source):
            continue
        for block in source.get("metric_blocks") or []:
            values = block.get("values") if isinstance(block, dict) else None
            if not isinstance(values, dict):
                continue
            if int(values.get("n") or 0) > 0 and (
                "macro_precision" in values or "balanced_accuracy" in values
            ):
                candidates.append(
                    {
                        "source": source.get("source"),
                        "path": block.get("path"),
                        "metrics": values,
                    }
                )
    primary = None
    for row in candidates:
        label = (str(row.get("source") or "") + " " + str(row.get("path") or "")).upper()
        if "PRIMARY" in label:
            primary = row
            break
    if primary is None and candidates:
        primary = candidates[0]
    return {
        "status": "PRESENT" if primary else "NOT_RECOVERED",
        "primary": primary,
        "candidate_count": len(candidates),
    }


def main() -> None:
    _configure_base()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not str(args.run_id).isdigit():
        raise SystemExit("RUN_ID_INVALID")
    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(action_token) < 100 or len(read_token) < 100:
        raise SystemExit("OIDC_TOKEN_INVALID")

    launches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
        futures = {
            pool.submit(
                base.launch_account,
                action_token,
                read_token,
                args.run_id,
                account_id,
                plan,
            ): account_id
            for account_id, plan in base.ACCOUNT_PLANS.items()
        }
        for future in concurrent.futures.as_completed(futures):
            account_id = futures[future]
            row = future.result()
            launches.append(row)
            print(
                "PHASE2_FULL_RESULTS_LAUNCH",
                json.dumps(
                    {
                        "account_id": account_id,
                        "kernel_ref": row["kernel_ref"],
                        "status": row["status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    launches.sort(key=lambda row: row["account_id"])

    terminal: dict[str, str] = {}
    deadline = time.monotonic() + 75 * 60
    while time.monotonic() < deadline and len(terminal) < len(launches):
        active_rows = [row for row in launches if row["account_id"] not in terminal]
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
            futures = {
                pool.submit(
                    base.probe_extractor_status,
                    read_token,
                    row["account_id"],
                    row["kernel_ref"],
                ): row
                for row in active_rows
            }
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                status = future.result()
                print(
                    "PHASE2_FULL_RESULTS_STATUS",
                    json.dumps(
                        {"account_id": row["account_id"], "status": status or "UNKNOWN"},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if status in base.TERMINAL:
                    terminal[row["account_id"]] = status
        if len(terminal) < len(launches):
            time.sleep(20)

    if len(terminal) != len(launches):
        raise SystemExit("FULL_RESULTS_TERMINAL_TIMEOUT")
    failed = {
        account: status
        for account, status in terminal.items()
        if status != "COMPLETE"
    }
    if failed:
        raise SystemExit(
            "FULL_RESULTS_TERMINAL_FAILURE=" + json.dumps(failed, sort_keys=True)
        )

    account_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as pool:
        futures = {
            pool.submit(base.fetch_account_result, read_token, row): row["account_id"]
            for row in launches
        }
        for future in concurrent.futures.as_completed(futures):
            account_results.append(future.result())
    account_results.sort(key=lambda row: row["account_id"])

    units = []
    for account_result in account_results:
        if account_result.get("status") != "PASS":
            raise RuntimeError(
                "FULL_RESULTS_ACCOUNT_RECEIPT_INVALID:"
                + str(account_result.get("account_id"))
            )
        units.extend(account_result.get("targets") or [])

    expected = {
        (f"M{index:02d}", resolution)
        for index in range(1, 13)
        for resolution in (224, 320, 384)
    }
    observed = {
        (str(row.get("model_id")), int(row.get("resolution")))
        for row in units
    }
    if observed != expected or len(units) != 36:
        raise RuntimeError(
            "FULL_RESULTS_MATRIX_COVERAGE_INVALID="
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
    for unit in sorted(units, key=lambda row: (row["model_id"], row["resolution"])):
        folds = _fold_receipts(unit)
        fold_count += len(folds)
        train_metric_folds += sum(
            isinstance(row.get("train_metrics"), dict) for row in folds
        )
        validation_metric_folds += sum(
            isinstance(row.get("validation_metrics"), dict) for row in folds
        )
        locked = _locked_test_summary(unit)
        if locked["status"] == "PRESENT":
            locked_present += 1
        else:
            locked_missing.append(
                {"model_id": unit["model_id"], "resolution": unit["resolution"]}
            )
        enriched = dict(unit)
        enriched["recovered_folds"] = folds
        enriched["locked_test_recovery"] = locked
        enriched_units.append(enriched)

    if fold_count != 180:
        raise RuntimeError(f"FULL_RESULTS_FOLD_COUNT_INVALID:{fold_count}")
    if validation_metric_folds != 180:
        raise RuntimeError(
            f"FULL_RESULTS_VALIDATION_METRICS_INCOMPLETE:{validation_metric_folds}"
        )

    output = {
        "schema": "pneumonia.phase2.full_results.matrix.v1",
        "status": "PASS",
        "github_run_id": str(args.run_id),
        "accounts_complete": len(account_results),
        "model_resolution_units": len(enriched_units),
        "folds_represented": fold_count,
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
        "launches": launches,
        "units": enriched_units,
    }
    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "PHASE2_FULL_RESULTS_MATRIX_PASS",
        json.dumps(
            {
                "units": len(enriched_units),
                "folds": fold_count,
                "train_metric_folds": train_metric_folds,
                "validation_metric_folds": validation_metric_folds,
                "locked_test_units_recovered": locked_present,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
