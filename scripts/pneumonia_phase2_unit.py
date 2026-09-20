#!/usr/bin/env python3
"""Build and validate one resumable Phase-2 model/resolution Kaggle unit.

The campaign deliberately assigns exactly one non-M07 model to each Kaggle
account.  Every unit restores its own sealed dataset, trains at most one new
fold, and only performs OOF/locked-test reporting after all five folds exist.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import pathlib
import re
from typing import Any

SCHEMA = "pneumonia.phase2.unit.v1"
SPLIT_FINGERPRINT = (
    "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
)
RESOLUTIONS = (224, 320, 384)
FROZEN_M07_RECIPE_FINGERPRINT = (
    "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"
)
FROZEN_M07_RECIPE_RECEIPT_SHA256 = (
    "81d7165c247d27d867d3e0f82239dcb7f71a360775105c969f422cd69269ae54"
)
FROZEN_M07_RECIPE_SOURCE_RUN_ID = "35505225105"
FROZEN_M07_SHARED_PARAMS = {
    "batch_size": 12,
    "head_lr": 0.00013819845844024556,
    "finetune_lr": 5.4346062237001635e-05,
    "weight_decay": 0.00018992423414663237,
    "dropout": 0.2,
    "patience": 4,
    "lr_schedule": "cosine",
    "min_lr": 3e-07,
    "head_warmup_epochs": 1,
    "finetune_warmup_epochs": 0,
    "edge_filters": 16,
    "edge_max_gate": 0.25,
    "cbam_reduction": 32,
}
MODEL_ASSIGNMENTS = {
    "M01": {"account_id": "master", "owner": "azadka"},
    "M02": {"account_id": "kg-02", "owner": "radlinaradlina"},
    "M03": {"account_id": "kg-03", "owner": "rezanory"},
    "M04": {"account_id": "kg-04", "owner": "reyhanehazad"},
    "M05": {"account_id": "kg-05", "owner": "trickermark"},
    "M06": {"account_id": "kg-06", "owner": "msdenis"},
    "M08": {"account_id": "kg-07", "owner": "nisabulutmark"},
    "M09": {"account_id": "kg-08", "owner": "azadkk"},
    "M10": {"account_id": "kg-09", "owner": "mylovevpn1"},
    "M11": {"account_id": "kg-10", "owner": "computstu1"},
    "M12": {"account_id": "kg-11", "owner": "jobreza1"},
}
TOKEN_RE = re.compile(
    r"PHASE2_UNIT_(M(?:0[1-6]|0[8-9]|1[0-2]))_R(224|320|384)_A([0-9]{2})"
)


class Phase2ContractError(ValueError):
    """Raised when a Phase-2 identity or notebook contract is invalid."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_token(token: str) -> dict[str, Any]:
    match = TOKEN_RE.fullmatch(str(token).strip())
    if match is None:
        raise Phase2ContractError("PHASE2_UNIT_TOKEN_INVALID")
    model_id, resolution_text, attempt_text = match.groups()
    attempt = int(attempt_text)
    if not 1 <= attempt <= 99:
        raise Phase2ContractError("PHASE2_UNIT_ATTEMPT_OUT_OF_RANGE")
    assignment = MODEL_ASSIGNMENTS[model_id]
    return {
        "token": token,
        "model_id": model_id,
        "resolution": int(resolution_text),
        "attempt": attempt,
        **assignment,
    }


def campaign_manifest() -> dict[str, Any]:
    units = [
        {
            "model_id": model_id,
            "resolution": resolution,
            **assignment,
            "folds": [1, 2, 3, 4, 5],
        }
        for model_id, assignment in MODEL_ASSIGNMENTS.items()
        for resolution in RESOLUTIONS
    ]
    manifest = {
        "schema": "pneumonia.phase2.campaign.v1",
        "split_fingerprint": SPLIT_FINGERPRINT,
        "shared_recipe": {
            "fingerprint": FROZEN_M07_RECIPE_FINGERPRINT,
            "receipt_sha256": FROZEN_M07_RECIPE_RECEIPT_SHA256,
            "source_run_id": FROZEN_M07_RECIPE_SOURCE_RUN_ID,
            "params": FROZEN_M07_SHARED_PARAMS,
        },
        "models": list(MODEL_ASSIGNMENTS),
        "resolutions": list(RESOLUTIONS),
        "units": units,
    }
    manifest["manifest_sha256"] = sha256_text(canonical_json(manifest))
    return manifest


def unit_contract(token: str, run_id: str, run_date: str | None = None) -> dict[str, Any]:
    parsed = parse_token(token)
    run_id = str(run_id).strip()
    if not run_id.isdigit():
        raise Phase2ContractError("PHASE2_UNIT_RUN_ID_INVALID")
    if run_date is None:
        run_date = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    if re.fullmatch(r"20[0-9]{6}", run_date) is None:
        raise Phase2ContractError("PHASE2_UNIT_RUN_DATE_INVALID")
    slug = (
        f"p17-p2-{parsed['model_id'].lower()}-r{parsed['resolution']}-"
        f"a{parsed['attempt']:02d}-{run_date}-{run_id}"
    )
    title = (
        f"Pneumonia V1.7 Phase2 {parsed['model_id']} R{parsed['resolution']} "
        f"A{parsed['attempt']:02d} {run_id}"
    )
    contract = {
        "schema": SCHEMA,
        **parsed,
        "run_id": run_id,
        "run_date": run_date,
        "kernel_ref": f"{parsed['owner']}/{slug}",
        "kernel_title": title,
        "state_dataset_handle": (
            f"{parsed['owner']}/pneumonia-{parsed['model_id'].lower()}-"
            f"r{parsed['resolution']}-state-v1-7"
        ),
        "split_fingerprint": SPLIT_FINGERPRINT,
        "max_new_folds": 1,
        "expected_folds": [1, 2, 3, 4, 5],
        "locked_test_policy": "RUN_ONLY_AFTER_FIVE_VALIDATED_FOLDS",
        "campaign_manifest_sha256": campaign_manifest()["manifest_sha256"],
    }
    contract["contract_sha256"] = sha256_text(canonical_json(contract))
    contract["request_id"] = (
        f"phase2-{parsed['model_id'].lower()}-r{parsed['resolution']}-"
        f"a{parsed['attempt']:02d}-{run_id}-{contract['contract_sha256'][:12]}"
    )
    return contract


def next_token(token: str, terminal_status: str) -> str | None:
    parsed = parse_token(token)
    attempt = int(parsed["attempt"])
    if terminal_status == "PARTIAL":
        if attempt >= 99:
            raise Phase2ContractError("PHASE2_UNIT_ATTEMPT_SPACE_EXHAUSTED")
        return (
            f"PHASE2_UNIT_{parsed['model_id']}_R{parsed['resolution']}_"
            f"A{attempt + 1:02d}"
        )
    if terminal_status != "COMPLETE":
        raise Phase2ContractError("PHASE2_UNIT_SUCCESSOR_STATUS_INVALID")
    index = RESOLUTIONS.index(parsed["resolution"])
    if index == len(RESOLUTIONS) - 1:
        return None
    return f"PHASE2_UNIT_{parsed['model_id']}_R{RESOLUTIONS[index + 1]}_A01"


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def _set_source(cell: dict[str, Any], source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)
    cell["execution_count"] = None
    cell["outputs"] = []


def _validate_input_notebook(notebook: dict[str, Any]) -> None:
    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) <= 26:
        raise Phase2ContractError("PHASE2_SOURCE_NOTEBOOK_CELL_TOPOLOGY_INVALID")
    joined = "\n".join(_source(cell) for cell in cells)
    required = (
        SPLIT_FINGERPRINT,
        "PHASE2_MODEL_ORDER",
        "M07_CONTINUATION_PRECHECK_PASS",
        "def phase2_restore(",
        "def run_phase2_model_resolution(",
        "FINAL_REPORT.json",
    )
    missing = [marker for marker in required if marker not in joined]
    if missing:
        raise Phase2ContractError(
            "PHASE2_SOURCE_NOTEBOOK_MARKERS_MISSING=" + ",".join(missing)
        )


def _patch_run_cell(source: str) -> str:
    restored_return = '''    if (out / "FINAL_REPORT.json").exists():
        restored_report, _ = phase2_validate_report(model_id, resolution, out)
        phase2_ensure_final_persisted(model_id, resolution, out)
        return restored_report
'''
    restored_replacement = '''    if (out / "FINAL_REPORT.json").exists():
        restored_report, _ = phase2_validate_report(model_id, resolution, out)
        phase2_ensure_final_persisted(model_id, resolution, out)
        globals()["CGP_PHASE2_EXECUTION_STATS"] = {
            "restored_folds": [1, 2, 3, 4, 5],
            "new_folds_completed": 0,
        }
        return restored_report
'''
    if source.count(restored_return) != 1:
        raise Phase2ContractError("PHASE2_RUN_CELL_RESTORED_RETURN_NOT_FOUND")
    source = source.replace(restored_return, restored_replacement, 1)
    start = source.find("    fold_results = []")
    end_marker = "    # --------------------------------------------------------\n    # OOF"
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise Phase2ContractError("PHASE2_RUN_CELL_FOLD_LOOP_NOT_FOUND")
    bounded_loop = '''    fold_results = []
    oof_parts = []
    thresholds = {}
    restored_folds = [
        fold for fold in range(1, 6)
        if (folds_root / f"fold_{fold}" / "COMPLETED.json").is_file()
    ]
    new_folds_completed = 0

    # --------------------------------------------------------
    # Five final folds — at most one new fold per Kaggle run.
    # --------------------------------------------------------
    for fold in range(1, 6):
        fold_dir = folds_root / f"fold_{fold}"
        completed = fold_dir / "COMPLETED.json"
        pred_path = fold_dir / "validation_predictions.csv"

        if (
            completed.exists()
            or pred_path.exists()
            or (fold_dir / "final_selected.weights.h5").exists()
        ):
            payload, pred = phase2_validate_fold(model_id, resolution, fold, fold_dir)
            phase2_ensure_fold_persisted(model_id, resolution, fold, fold_dir)
            print(f"[{model_id} R{resolution} F{fold}] validated completion and persistence — SKIP")
        else:
            if new_folds_completed >= int(CGP_PHASE2_MAX_NEW_FOLDS):
                raise RuntimeError("PHASE2_UNIT_NEW_FOLD_BOUND_EXCEEDED")
            payload, pred = phase2_train_one_fold(model_id, resolution, fold, fold_dir)
            phase2_save_fold_history_figures(fold_dir, PHASE2_FINAL_HEAD_EPOCHS)
            phase2_persist_fold(model_id, resolution, fold)
            new_folds_completed += 1

        fold_results.append(payload)
        pred["fold_id"] = int(fold)
        oof_parts.append(pred)
        thresholds[fold] = float(payload["validation_metrics"]["threshold"])

        completed_folds = [
            candidate for candidate in range(1, 6)
            if (folds_root / f"fold_{candidate}" / "COMPLETED.json").is_file()
        ]
        if new_folds_completed >= int(CGP_PHASE2_MAX_NEW_FOLDS) and len(completed_folds) < 5:
            globals()["CGP_PHASE2_EXECUTION_STATS"] = {
                "restored_folds": restored_folds,
                "new_folds_completed": int(new_folds_completed),
            }
            partial = {
                "schema": "pneumonia.phase2.unit.terminal.v1",
                "status": "PARTIAL",
                "scientific_pass": False,
                "model_id": model_id,
                "resolution": int(resolution),
                "restored_folds": restored_folds,
                "completed_folds": completed_folds,
                "completed_fold": int(fold),
                "new_folds_completed": int(new_folds_completed),
                "five_fold_ready": False,
                "locked_test_started": False,
                "next_action": "DISPATCH_NEXT_IMMUTABLE_ATTEMPT_AFTER_TERMINAL_VERIFICATION",
            }
            atomic_write_json(WORK / "PHASE2_UNIT_TERMINAL_RECEIPT.json", partial)
            print("PHASE2_UNIT_TERMINAL_RECEIPT=" + json.dumps(partial, sort_keys=True))
            return partial

'''
    patched = source[:start] + bounded_loop + source[end:]
    final_return = "    return summary\n"
    final_replacement = '''    globals()["CGP_PHASE2_EXECUTION_STATS"] = {
        "restored_folds": restored_folds,
        "new_folds_completed": int(new_folds_completed),
    }
    return summary
'''
    if patched.count(final_return) != 1:
        raise Phase2ContractError("PHASE2_RUN_CELL_FINAL_RETURN_NOT_FOUND")
    patched = patched.replace(final_return, final_replacement, 1)
    if patched.count("PHASE2_UNIT_NEW_FOLD_BOUND_EXCEEDED") != 1:
        raise Phase2ContractError("PHASE2_RUN_CELL_PATCH_COUNT_INVALID")
    return patched


def _patch_legacy_persistence_cell(source: str) -> str:
    """Keep shared helpers but remove the executable M07 persistence gate.

    Phase-2 units own a per-model/per-resolution state dataset through the
    phase2 persistence helpers.  Running the legacy M07 gate first targets the
    immutable rezanory M07 dataset from a different account and fails before
    the unit can restore or train its own fold.
    """
    start_markers = (
        "RESTORE_SUMMARY = _try_restore_persisted_state_compat()\n",
        "RESTORE_SUMMARY = _try_restore_persisted_state()\n",
    )
    end_marker = 'print("✅ PERSISTENCE GATE PASSED")'
    matched_starts = [marker for marker in start_markers if source.count(marker) == 1]
    if len(matched_starts) != 1 or source.count(end_marker) != 1:
        raise Phase2ContractError("PHASE2_LEGACY_PERSISTENCE_GATE_BOUNDARY_INVALID")
    start_marker = matched_starts[0]
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    replacement = '''# Phase-2 unit: do not restore or mutate the legacy M07 persistence dataset.
PERSIST_RESTORE_VERIFIED = False
RESTORE_SUMMARY = {
    "folds": 0,
    "light_state": 0,
    "restore_status": "SKIPPED_FOR_EXACT_PHASE2_UNIT",
}
print("PHASE2_UNIT_LEGACY_M07_PERSISTENCE_SKIPPED")'''
    patched = source[:start] + replacement + source[end:]
    if any(marker in patched for marker in start_markers) or (
        "M07 run-safe persistence healthcheck before training" in patched
    ):
        raise Phase2ContractError("PHASE2_LEGACY_PERSISTENCE_GATE_STILL_EXECUTABLE")
    return patched


def build_notebook(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    token: str,
    run_id: str,
    run_date: str | None = None,
) -> dict[str, Any]:
    contract = unit_contract(token, run_id, run_date)
    notebook = json.loads(input_path.read_text(encoding="utf-8"))
    _validate_input_notebook(notebook)
    source_cells = notebook["cells"]
    def find_unique(marker: str) -> int:
        matches = [
            index for index, cell in enumerate(source_cells) if marker in _source(cell)
        ]
        if len(matches) != 1:
            raise Phase2ContractError(
                f"PHASE2_SOURCE_MARKER_COUNT_INVALID={marker}:{len(matches)}"
            )
        return matches[0]

    legacy_persistence_index = find_unique("PERSISTENCE GATE — MUST PASS BEFORE TRAINING")
    registry_index = find_unique("PHASE-2 CANONICAL MODEL REGISTRY")
    persistence_index = find_unique("def phase2_restore(")
    run_index = find_unique("def run_phase2_model_resolution(")
    if not registry_index < persistence_index < run_index:
        raise Phase2ContractError("PHASE2_SOURCE_DEFINITION_ORDER_INVALID")
    # Cells 11 and 12 are the M07 runtime smoke test and M07 continuation
    # precheck.  They are executable, expensive, and unrelated to the exact
    # non-M07 Phase-2 unit, so they must not enter the generated notebook.
    selected_indexes = [*range(0, 11), registry_index, persistence_index, run_index]
    cells = [copy.deepcopy(source_cells[index]) for index in selected_indexes]
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    config_source = _source(cells[1])
    config_append = f'''

# Exact Phase-2 unit identity injected by pneumonia_phase2_unit.py.
PHASE2_UNIT_TOKEN = {contract['token']!r}
PHASE2_UNIT_MODEL_ID = {contract['model_id']!r}
PHASE2_UNIT_RESOLUTION = {contract['resolution']!r}
PHASE2_UNIT_ATTEMPT = {contract['attempt']!r}
PHASE2_UNIT_CONTRACT_SHA256 = {contract['contract_sha256']!r}
PHASE2_PERSIST_OWNER = {contract['owner']!r}
UNLOCK_REMAINING_MODELS = True
CGP_PHASE2_MAX_NEW_FOLDS = 1
'''
    _set_source(cells[1], config_source + config_append)
    legacy_cell = selected_indexes.index(legacy_persistence_index)
    _set_source(cells[legacy_cell], _patch_legacy_persistence_cell(_source(cells[legacy_cell])))

    # Cell 12 performs Confirmation HPO and must never execute in a distributed
    # comparator unit.  Inject only the exact recipe already proven by the M07
    # R384 A12 terminal receipt, so every account has the same scientific input
    # without repeating HPO or depending on cross-account M07 persistence.
    frozen_recipe_source = f'''# Exact frozen M07 recipe for Phase-2 comparator units.
shared_params = {FROZEN_M07_SHARED_PARAMS!r}
recipe = {{
    "schema": "m07.final.confirmed.recipe.v1.6",
    "recipe_fingerprint_sha256": {FROZEN_M07_RECIPE_FINGERPRINT!r},
    "shared_training_params": dict(shared_params),
    "source_run_id": {FROZEN_M07_RECIPE_SOURCE_RUN_ID!r},
    "source_receipt_sha256": {FROZEN_M07_RECIPE_RECEIPT_SHA256!r},
}}
EDGE_FILTERS = int(shared_params["edge_filters"])
EDGE_MAX_GATE = float(shared_params["edge_max_gate"])
CBAM_REDUCTION = int(shared_params["cbam_reduction"])
print("PHASE2_UNIT_FROZEN_M07_RECIPE_BOUND", recipe["recipe_fingerprint_sha256"])
'''
    cells.insert(
        11,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"cgp_phase2_frozen_recipe": True},
            "outputs": [],
            "source": frozen_recipe_source.splitlines(keepends=True),
        },
    )
    _set_source(cells[-1], _patch_run_cell(_source(cells[-1])))

    terminal_tail = f'''# Exact bounded Phase-2 unit dispatch.
if not UNLOCK_REMAINING_MODELS:
    raise RuntimeError("PHASE2_UNIT_NOT_UNLOCKED")
if (
    PHASE2_UNIT_MODEL_ID != {contract['model_id']!r}
    or int(PHASE2_UNIT_RESOLUTION) != {contract['resolution']!r}
):
    raise RuntimeError("PHASE2_UNIT_IDENTITY_DRIFT")
phase2_unit_result = run_phase2_model_resolution(
    PHASE2_UNIT_MODEL_ID,
    PHASE2_UNIT_RESOLUTION,
)
if phase2_unit_result.get("status") == "COMPLETE":
    phase2_execution_stats = globals().get("CGP_PHASE2_EXECUTION_STATS", {{}})
    phase2_terminal = {{
        "schema": "pneumonia.phase2.unit.terminal.v1",
        "status": "COMPLETE",
        "scientific_pass": True,
        "model_id": PHASE2_UNIT_MODEL_ID,
        "resolution": int(PHASE2_UNIT_RESOLUTION),
        "completed_folds": [1, 2, 3, 4, 5],
        "completed_fold": 5,
        "restored_folds": phase2_execution_stats.get("restored_folds", [1, 2, 3, 4, 5]),
        "new_folds_completed": int(phase2_execution_stats.get("new_folds_completed", 0)),
        "five_fold_ready": True,
        "locked_test_started": True,
        "final_report_status": "COMPLETE",
        "next_action": "UNIT_COMPLETE",
    }}
    atomic_write_json(WORK / "PHASE2_UNIT_TERMINAL_RECEIPT.json", phase2_terminal)
    print("PHASE2_UNIT_TERMINAL_RECEIPT=" + json.dumps(phase2_terminal, sort_keys=True))
'''
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"cgp_phase2_unit": True},
            "outputs": [],
            "source": terminal_tail.splitlines(keepends=True),
        }
    )
    notebook["cells"] = cells
    generated_source = "\n".join(_source(cell) for cell in cells)
    forbidden_executable_markers = (
        "RESTORE_SUMMARY = _try_restore_persisted_state_compat()",
        "RESTORE_SUMMARY = _try_restore_persisted_state()",
        "M07_CONTINUATION_PRECHECK_PASS",
        "M07 RUNTIME PRE-FLIGHT PASS",
    )
    leaked = [marker for marker in forbidden_executable_markers if marker in generated_source]
    if leaked:
        raise Phase2ContractError("PHASE2_LEGACY_EXECUTION_LEAK=" + ",".join(leaked))
    required_frozen_markers = (
        "PHASE2_UNIT_FROZEN_M07_RECIPE_BOUND",
        FROZEN_M07_RECIPE_FINGERPRINT,
        FROZEN_M07_RECIPE_RECEIPT_SHA256,
        "shared_params =",
    )
    missing_frozen = [
        marker for marker in required_frozen_markers if marker not in generated_source
    ]
    if missing_frozen:
        raise Phase2ContractError(
            "PHASE2_FROZEN_RECIPE_BINDING_MISSING=" + ",".join(missing_frozen)
        )
    forbidden_hpo_markers = (
        "generate_confirmation_candidates()",
        "run_or_restore_hpo_candidate_fold(",
        "confirmation_ranked = sorted(",
    )
    leaked_hpo = [marker for marker in forbidden_hpo_markers if marker in generated_source]
    if leaked_hpo:
        raise Phase2ContractError("PHASE2_HPO_EXECUTION_LEAK=" + ",".join(leaked_hpo))
    metadata = notebook.setdefault("metadata", {})
    metadata["cgp_phase2_unit"] = contract
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        **contract,
        "notebook_path": str(output_path),
        "notebook_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "selected_source_cells": selected_indexes,
        "output_cell_count": len(cells),
    }


def validate_terminal_receipt(receipt: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != "pneumonia.phase2.unit.terminal.v1":
        raise Phase2ContractError("PHASE2_TERMINAL_SCHEMA_INVALID")
    if receipt.get("model_id") != contract["model_id"]:
        raise Phase2ContractError("PHASE2_TERMINAL_MODEL_ID_MISMATCH")
    if int(receipt.get("resolution", -1)) != contract["resolution"]:
        raise Phase2ContractError("PHASE2_TERMINAL_RESOLUTION_MISMATCH")
    folds = receipt.get("completed_folds")
    if not isinstance(folds, list) or folds != sorted(set(folds)):
        raise Phase2ContractError("PHASE2_TERMINAL_COMPLETED_FOLDS_INVALID")
    if any(fold not in contract["expected_folds"] for fold in folds):
        raise Phase2ContractError("PHASE2_TERMINAL_FOLD_OUT_OF_RANGE")
    new_folds = int(receipt.get("new_folds_completed", -1))
    if new_folds not in (0, 1):
        raise Phase2ContractError("PHASE2_TERMINAL_NEW_FOLD_BOUND_BROKEN")
    status = receipt.get("status")
    if status == "PARTIAL":
        if len(folds) >= 5 or receipt.get("five_fold_ready") is not False:
            raise Phase2ContractError("PHASE2_PARTIAL_RECEIPT_CONTRADICTORY")
        if receipt.get("locked_test_started") is not False:
            raise Phase2ContractError("PHASE2_PARTIAL_LOCKED_TEST_POLICY_BROKEN")
    elif status == "COMPLETE":
        if folds != contract["expected_folds"]:
            raise Phase2ContractError("PHASE2_COMPLETE_FOLD_SET_INVALID")
        if receipt.get("five_fold_ready") is not True:
            raise Phase2ContractError("PHASE2_COMPLETE_READY_FLAG_INVALID")
        if receipt.get("scientific_pass") is not True:
            raise Phase2ContractError("PHASE2_COMPLETE_SCIENTIFIC_PASS_MISSING")
    else:
        raise Phase2ContractError("PHASE2_TERMINAL_STATUS_INVALID")
    return receipt


def _write_identity_files(directory: pathlib.Path, result: dict[str, Any]) -> None:
    values = {
        "m07_kernel_slug.txt": result["kernel_ref"],
        "m07_kernel_title.txt": result["kernel_title"],
        "m07_request_id.txt": result["request_id"],
        "m07_account_id.txt": result["account_id"],
        "m07_owner.txt": result["owner"],
        "phase2_unit_contract.json": json.dumps(result, ensure_ascii=False, indent=2),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (directory / name).write_text(str(value).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", type=pathlib.Path)

    contract_parser = subparsers.add_parser("contract")
    contract_parser.add_argument("--token", required=True)
    contract_parser.add_argument("--run-id", required=True)
    contract_parser.add_argument("--run-date")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--input", required=True, type=pathlib.Path)
    build_parser.add_argument("--output", required=True, type=pathlib.Path)
    build_parser.add_argument("--token", required=True)
    build_parser.add_argument("--run-id", required=True)
    build_parser.add_argument("--run-date")
    build_parser.add_argument("--identity-directory", type=pathlib.Path)

    receipt_parser = subparsers.add_parser("validate-receipt")
    receipt_parser.add_argument("--receipt", required=True, type=pathlib.Path)
    receipt_parser.add_argument("--token", required=True)
    receipt_parser.add_argument("--run-id", required=True)
    receipt_parser.add_argument("--run-date")

    args = parser.parse_args(argv)
    if args.command == "manifest":
        result = campaign_manifest()
        if args.output:
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command == "contract":
        result = unit_contract(args.token, args.run_id, args.run_date)
    elif args.command == "build":
        result = build_notebook(
            args.input, args.output, args.token, args.run_id, args.run_date
        )
        if args.identity_directory:
            _write_identity_files(args.identity_directory, result)
    else:
        contract = unit_contract(args.token, args.run_id, args.run_date)
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        result = validate_terminal_receipt(receipt, contract)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
