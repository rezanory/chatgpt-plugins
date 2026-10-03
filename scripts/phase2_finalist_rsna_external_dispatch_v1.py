from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time

import m07_rsna_pediatric_external_dispatch_v1 as legacy

DATASET_REF = legacy.DATASET_REF
EXPECTED_MANIFEST_SHA256 = legacy.EXPECTED_MANIFEST_SHA256
EXPECTED_SPLIT = legacy.EXPECTED_SPLIT
EXPECTED_RECIPE = legacy.EXPECTED_RECIPE
EXPECTED_COUNTS = legacy.EXPECTED_COUNTS

SELECTION_FREEZE_REL = pathlib.Path(
    "evidence/phase2_final_closure_v1/SELECTION_FREEZE.json"
)
EXPECTED_CANDIDATES = {
    ("M09", 384),
    ("M10", 320),
}

CONTRACTS = {
    "PHASE2_FINALIST_RSNA_M09_R384_V1": {
        "model_id": "M09",
        "resolution": 384,
        "account_id": "kg-08",
        "owner": "azadkk",
        "state_handle": "azadkk/pneumonia-m09-r384-state-v1-7",
        "batch_size": 4,
    },
    "PHASE2_FINALIST_RSNA_M10_R320_V1": {
        "model_id": "M10",
        "resolution": 320,
        "account_id": "kg-09",
        "owner": "mylovevpn1",
        "state_handle": "mylovevpn1/pneumonia-m10-r320-state-v1-7",
        "batch_size": 6,
    },
}


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_and_verify_selection_freeze(root: pathlib.Path) -> tuple[dict, str]:
    path = root / SELECTION_FREEZE_REL
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    candidates = {
        (str(row.get("model_id")), int(row.get("resolution") or 0))
        for row in payload.get("frozen_candidate_set") or []
        if isinstance(row, dict)
    }
    required = {
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
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "SELECTION_FREEZE_POLICY_MISMATCH="
            + json.dumps(mismatches, sort_keys=True)
        )
    metrics = ((payload.get("selection_basis") or {}).get("requested_metrics") or [])
    if metrics != ["macro_precision", "macro_recall"]:
        raise RuntimeError("SELECTION_FREEZE_METRICS_DRIFT")
    if candidates != EXPECTED_CANDIDATES:
        raise RuntimeError(
            "SELECTION_FREEZE_CANDIDATES_DRIFT="
            + json.dumps(sorted(candidates), sort_keys=True)
        )
    return payload, sha256_bytes(raw)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"FINALIST_EXTERNAL_PATCH_BOUNDARY:{label}:{count}")
    return source.replace(old, new, 1)


def build_kernel_script(
    source: str,
    contract: dict,
    *,
    manifest_b64: str,
    selection_freeze_sha256: str,
) -> str:
    model_id = str(contract["model_id"])
    resolution = int(contract["resolution"])
    state_handle = str(contract["state_handle"])
    batch_size = int(contract["batch_size"])

    if (model_id, resolution) not in EXPECTED_CANDIDATES:
        raise RuntimeError("FINALIST_EXTERNAL_CONTRACT_NOT_FROZEN")
    if resolution not in {320, 384}:
        raise RuntimeError("FINALIST_EXTERNAL_RESOLUTION_UNSUPPORTED")
    if not (
        len(selection_freeze_sha256) == 64
        and all(ch in "0123456789abcdef" for ch in selection_freeze_sha256)
    ):
        raise RuntimeError("FINALIST_EXTERNAL_FREEZE_SHA_INVALID")

    future = "from __future__ import annotations\n"
    if not source.startswith(future):
        raise RuntimeError("FINALIST_EXTERNAL_SOURCE_FUTURE_IMPORT_DRIFT")

    preamble = (
        future
        + f"MODEL_ID={model_id!r}\n"
        + f"SELECTION_FREEZE_SHA256={selection_freeze_sha256!r}\n"
        + f"EMBEDDED_MANIFEST_B64={manifest_b64!r}\n"
        + "import os\n"
        + f"os.environ['M07_EXTERNAL_RESOLUTION']={str(resolution)!r}\n"
        + f"os.environ['M07_EXTERNAL_STATE_HANDLE']={state_handle!r}\n"
        + f"os.environ['M07_EXTERNAL_BATCH_SIZE']={str(batch_size)!r}\n"
    )
    patched = preamble + source[len(future):]

    patched = _replace_once(
        patched,
        'OUT = WORK / f"M07_RSNA_PEDIATRIC_EXTERNAL_R{IMAGE_SIZE}_V1"',
        'OUT = WORK / f"{MODEL_ID}_RSNA_PEDIATRIC_EXTERNAL_R{IMAGE_SIZE}_V1"',
        "out_root",
    )

    old_handles = '''STATE_HANDLES = {
    224: "rezanory/m07-final-5fold-fix2-d260914d",
    320: "trickermark/m07-gate-r320-state-v1-7",
    384: "trickermark/m07-gate-r384-state-v1-7",
}
'''
    new_handles = f'''STATE_HANDLES = {{
    224: {state_handle!r},
    320: {state_handle!r},
    384: {state_handle!r},
}}
'''
    patched = _replace_once(
        patched, old_handles, new_handles, "state_handles"
    )

    patched = _replace_once(
        patched,
        "import base64, gc, hashlib, json, math, os, re, shutil, zlib",
        "import base64, gc, hashlib, json, math, os, re, shutil, zipfile, zlib",
        "zipfile_import",
    )
    patched = _replace_once(
        patched,
        "and set(artifacts) == expected_artifacts",
        "and expected_artifacts.issubset(set(artifacts))",
        "campaign_artifact_subset",
    )

    old_state_fold_presence = """            root = marker.parent.resolve()
            if all(
                (
                    root
                    / f"FOLD_{fold}_RECOVERY"
                    / "FOLDS"
                    / f"fold_{fold}"
                    / "COMPLETED.json"
                ).is_file()
                and (
                    root
                    / f"FOLD_{fold}_RECOVERY"
                    / "FOLDS"
                    / f"fold_{fold}"
                    / "final_selected.weights.h5"
                ).is_file()
                for fold in range(1, 6)
            ):
                roots.append(root)
"""
    new_state_fold_presence = """            root = marker.parent.resolve()
            if all(
                (root / f"FOLD_{fold}_RECOVERY.cgpzip").is_file()
                and sha256_file(root / f"FOLD_{fold}_RECOVERY.cgpzip")
                    == artifacts[f"FOLD_{fold}_RECOVERY.cgpzip"]
                for fold in range(1, 6)
            ):
                roots.append(root)
"""
    patched = _replace_once(
        patched,
        old_state_fold_presence,
        new_state_fold_presence,
        "standard_state_archive_presence",
    )

    patched = patched.replace(
        'payload.get("model_id") == "M07"',
        'payload.get("model_id") == MODEL_ID',
    )
    patched = patched.replace(
        'receipt.get("model_id") == "M07"',
        'receipt.get("model_id") == MODEL_ID',
    )

    materializer = r"""
def materialize_standard_fold(state_root, fold):
    archive = state_root / f"FOLD_{fold}_RECOVERY.cgpzip"
    if not archive.is_file():
        raise RuntimeError(f"FOLD_{fold}_ARCHIVE_MISSING_R{IMAGE_SIZE}")
    dest = WORK / f"{MODEL_ID}_STATE_R{IMAGE_SIZE}_MATERIALIZED" / f"fold_{fold}"
    dest.mkdir(parents=True, exist_ok=True)
    completed_target = dest / "COMPLETED.json"
    weight_target = dest / "final_selected.weights.h5"
    with zipfile.ZipFile(archive) as z:
        names = [name for name in z.namelist() if not name.endswith("/")]
        fold_token = f"/fold_{fold}/".lower()
        completed = [
            name for name in names
            if Path(name).name == "COMPLETED.json"
            and fold_token in ("/" + name.replace("\\", "/")).lower()
        ]
        weights = [
            name for name in names
            if Path(name).name == "final_selected.weights.h5"
            and fold_token in ("/" + name.replace("\\", "/")).lower()
        ]
        if len(completed) != 1 or len(weights) != 1:
            raise RuntimeError(
                f"FOLD_{fold}_ARCHIVE_MEMBER_COUNT_INVALID_R{IMAGE_SIZE}:"
                f"{len(completed)}:{len(weights)}"
            )
        cinfo = z.getinfo(completed[0])
        winfo = z.getinfo(weights[0])
        if cinfo.file_size < 1 or cinfo.file_size > 2_000_000:
            raise RuntimeError(f"FOLD_{fold}_COMPLETED_SIZE_INVALID_R{IMAGE_SIZE}")
        if winfo.file_size < 1 or winfo.file_size > 1_000_000_000:
            raise RuntimeError(f"FOLD_{fold}_WEIGHT_SIZE_INVALID_R{IMAGE_SIZE}")
        for member, target in (
            (completed[0], completed_target),
            (weights[0], weight_target),
        ):
            part = target.with_suffix(target.suffix + ".part")
            with z.open(member, "r") as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            part.replace(target)
    return dest

"""
    patched = _replace_once(
        patched,
        "def load_folds(state_root):",
        materializer + "\ndef load_folds(state_root):",
        "standard_fold_materializer",
    )
    patched = _replace_once(
        patched,
        '            root = state_root / f"FOLD_{fold}_RECOVERY/FOLDS/fold_{fold}"',
        "            root = materialize_standard_fold(state_root, fold)",
        "standard_fold_root",
    )

    if patched.count('== MODEL_ID') < 2:
        raise RuntimeError("FINALIST_EXTERNAL_MODEL_ID_PATCH_INCOMPLETE")

    patched = _replace_once(
        patched,
        'oof_files = list(state_root.rglob("M07_OOF_PREDICTIONS.csv"))',
        '''oof_files = list(state_root.rglob("OOF_PREDICTIONS.csv"))
    if not oof_files:
        oof_files = list(state_root.rglob("M07_OOF_PREDICTIONS.csv"))''',
        "oof_predictions",
    )

    patched = _replace_once(
        patched,
        '    return tf.keras.Model(inputs, out, name="M07")',
        '    return tf.keras.Model(inputs, out, name=MODEL_ID)',
        "keras_model_name",
    )

    file_literals = (
        "M07_RSNA_PEDIATRIC_INFERENCE_MANIFEST.csv",
        "M07_RSNA_PEDIATRIC_PRIMARY_LT10_MANIFEST.csv",
        "M07_RSNA_PREINFERENCE_INTEGRITY_RECEIPT.json",
        "M07_RSNA_PEDIATRIC_EXTERNAL_PREDICTIONS.csv",
        "M07_RSNA_PEDIATRIC_PRIMARY_LT10_PREDICTIONS.csv",
        "M07_RSNA_PEDIATRIC_EXPANDED_LE18_PREDICTIONS.csv",
        "M07_RSNA_PEDIATRIC_EXTERNAL_REPORT.json",
        "M07_INTERNAL_RSNA_EXTERNAL_COMPARISON.json",
        "M07_MODEL_CARD_RSNA_EXTERNAL_ADDENDUM.md",
        "M07_RSNA_PEDIATRIC_EXTERNAL_TERMINAL_RECEIPT.json",
    )
    for literal in file_literals:
        replacement = literal.replace("M07_", "{MODEL_ID}_", 1)
        patched = patched.replace(
            f'"{literal}"',
            f'f"{replacement}"',
        )

    patched = _replace_once(
        patched,
        'oof_files = list(state_root.rglob("M07_OOF_PRIMARY_METRICS.json"))',
        '''oof_files = list(state_root.rglob("OOF_METRICS.json"))
if not oof_files:
    oof_files = list(state_root.rglob("M07_OOF_PRIMARY_METRICS.json"))''',
        "oof_metrics",
    )

    patched = _replace_once(
        patched,
        '"schema": "m07.external.rsna_pediatric.resolution.v1",',
        (
            '"schema": "pneumonia.phase2.finalist.external.rsna_pediatric.resolution.v1",\n'
            '    "model_id": MODEL_ID,\n'
            '    "selection_freeze_sha256": SELECTION_FREEZE_SHA256,\n'
            '    "scientific_classification": "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION",\n'
            '    "locked_test_used_for_selection": False,\n'
            '    "external_used_for_selection": False,\n'
            '    "external_may_reselect": False,'
        ),
        "report_schema",
    )

    patched = _replace_once(
        patched,
        '"model": "M07 Final Gate - ConvNeXt-Tiny + EdgeBlock + CBAM + FLSD-53",',
        (
            '"model": f"{MODEL_ID} - ConvNeXt-Tiny + EdgeBlock + CBAM + '
            'shared frozen Phase-2 recipe",'
        ),
        "report_model",
    )

    patched = _replace_once(
        patched,
        '"schema": "m07.external.rsna_pediatric.terminal.v1",',
        (
            '"schema": "pneumonia.phase2.finalist.external.rsna_pediatric.terminal.v1",\n'
            '    "model_id": MODEL_ID,\n'
            '    "selection_freeze_sha256": SELECTION_FREEZE_SHA256,\n'
            '    "scientific_classification": "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION",'
        ),
        "receipt_schema",
    )

    patched = _replace_once(
        patched,
        '"external_calibration_fitting": False,\n    "source_state": STATE_HANDLE,',
        (
            '"external_calibration_fitting": False,\n'
            '    "locked_test_used_for_selection": False,\n'
            '    "external_used_for_selection": False,\n'
            '    "external_may_reselect": False,\n'
            '    "source_state": STATE_HANDLE,'
        ),
        "receipt_selection_policy",
    )

    old_archive = '''str(WORK / f"M07_RSNA_PEDIATRIC_EXTERNAL_R{IMAGE_SIZE}_V1_COMPLETE"),'''
    new_archive = '''str(WORK / f"{MODEL_ID}_RSNA_PEDIATRIC_EXTERNAL_R{IMAGE_SIZE}_V1_COMPLETE"),'''
    patched = _replace_once(
        patched, old_archive, new_archive, "complete_archive"
    )

    # The evaluator is inference-only. Keep these assertions as generated-code
    # tripwires so later source drift fails before any Kaggle submission.
    forbidden = (
        "model.fit(",
        "optimizer.apply_gradients",
        "external_threshold_tuning\": True",
        "external_adaptation\": True",
        "external_calibration_fitting\": True",
    )
    found = [marker for marker in forbidden if marker in patched]
    if found:
        raise RuntimeError(
            "FINALIST_EXTERNAL_FORBIDDEN_EXECUTION_MARKER="
            + ",".join(found)
        )

    compile(patched, "<phase2-finalist-rsna-external>", "exec")
    return patched


def verify_receipt(
    receipt: dict,
    contract: dict,
    *,
    selection_freeze_sha256: str,
) -> None:
    expected = {
        "schema": "pneumonia.phase2.finalist.external.rsna_pediatric.terminal.v1",
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "model_id": contract["model_id"],
        "resolution": int(contract["resolution"]),
        "primary_n": EXPECTED_COUNTS["primary_n"],
        "primary_normal": EXPECTED_COUNTS["primary_normal"],
        "primary_lung_opacity": EXPECTED_COUNTS["primary_lung_opacity"],
        "expanded_n": EXPECTED_COUNTS["expanded_n"],
        "expanded_normal": EXPECTED_COUNTS["expanded_normal"],
        "expanded_lung_opacity": EXPECTED_COUNTS["expanded_lung_opacity"],
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
        "external_calibration_fitting": False,
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "external_may_reselect": False,
        "source_state": contract["state_handle"],
        "split_fingerprint": EXPECTED_SPLIT,
        "hpo_recipe_fingerprint": EXPECTED_RECIPE,
        "external_dataset": DATASET_REF,
        "external_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "selection_freeze_sha256": selection_freeze_sha256,
        "scientific_classification": "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION",
    }
    mismatches = {
        key: {"expected": expected_value, "actual": receipt.get(key)}
        for key, expected_value in expected.items()
        if receipt.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "FINALIST_RSNA_TERMINAL_RECEIPT_MISMATCH="
            + json.dumps(mismatches, sort_keys=True)
        )
    claimed = str(receipt.get("receipt_sha256") or "")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    actual = sha256_bytes(canonical_json(body))
    if claimed != actual:
        raise RuntimeError("FINALIST_RSNA_TERMINAL_RECEIPT_SHA_MISMATCH")
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("FINALIST_RSNA_TERMINAL_ARTIFACT_MANIFEST_MISSING")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.token not in CONTRACTS:
        raise SystemExit("PHASE2_FINALIST_RSNA_TOKEN_INVALID")
    if not str(args.run_id).isdigit():
        raise SystemExit("PHASE2_FINALIST_RSNA_RUN_ID_INVALID")

    contract = CONTRACTS[args.token]
    root = pathlib.Path(__file__).resolve().parents[1]
    _, selection_freeze_sha256 = load_and_verify_selection_freeze(root)

    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(action_token) < 100 or len(read_token) < 100:
        raise SystemExit("PHASE2_FINALIST_RSNA_OIDC_TOKEN_INVALID")

    source_path = root / "scripts" / "m07_rsna_pediatric_external_v1.py"
    source = source_path.read_text(encoding="utf-8")
    source_sha256 = sha256_bytes(source.encode("utf-8"))

    manifest_path = (
        root / "evidence" / "m07_external" / "rsna_pediatric_v1" / "manifest.json"
    )
    manifest = manifest_path.read_bytes()
    rows = json.loads(manifest.decode("utf-8"))
    if sha256_bytes(canonical_json(rows)) != EXPECTED_MANIFEST_SHA256:
        raise SystemExit("PHASE2_FINALIST_RSNA_MANIFEST_SHA_DRIFT")

    import base64
    import zlib

    manifest_b64 = base64.b64encode(zlib.compress(manifest, 9)).decode("ascii")
    code = build_kernel_script(
        source,
        contract,
        manifest_b64=manifest_b64,
        selection_freeze_sha256=selection_freeze_sha256,
    )

    model_id = str(contract["model_id"])
    resolution = int(contract["resolution"])
    account_id = str(contract["account_id"])
    owner = str(contract["owner"])
    state_handle = str(contract["state_handle"])
    slug = (
        f"phase2-finalist-rsna-{model_id.lower()}-r{resolution}-v1-{args.run_id}"
    )
    kernel_ref = f"{owner}/{slug}"
    runner_temp = pathlib.Path(os.environ["RUNNER_TEMP"])
    launch_path = runner_temp / f"{model_id}_R{resolution}_RSNA_LAUNCH.json"

    existing = legacy.post_json(
        legacy.READ_ENDPOINT,
        read_token,
        {
            "action": "resolved_kernel_status",
            "account_id": account_id,
            "kernel_ref": kernel_ref,
        },
        timeout=120,
    )
    existing_status = legacy.recursive_status(existing)
    transient_probe_status = int(existing.get("http_status") or 0)
    if transient_probe_status in {408, 425, 429, 500, 502, 503, 504}:
        legacy.write_json(
            launch_path,
            {
                "ok": False,
                "reused_existing": False,
                "existing_probe": existing,
                "kernel_ref": kernel_ref,
            },
        )
        raise SystemExit(
            "PHASE2_FINALIST_RSNA_EXISTING_PROBE_TRANSIENT_HTTP_"
            + str(transient_probe_status)
        )
    if existing_status in {"ERROR", "CANCELLED"}:
        raise SystemExit(
            "PHASE2_FINALIST_RSNA_EXISTING_KERNEL_TERMINAL_" + existing_status
        )

    if existing_status in {"QUEUED", "RUNNING", "COMPLETE"}:
        provider_ref = kernel_ref
        compute_mode = "REUSED_EXISTING"
        launch = {
            "ok": True,
            "reused_existing": True,
            "existing_status": existing_status,
            "provider_ref": provider_ref,
            "account_id": account_id,
            "model_id": model_id,
            "resolution": resolution,
            "cgp_compute_mode": compute_mode,
        }
        legacy.write_json(launch_path, launch)
    else:
        launch_payload = {
            "request_id": (
                f"phase2-finalist-rsna-{model_id.lower()}-r{resolution}-v1-"
                f"{args.run_id}"
            ),
            "provider": "kaggle",
            "operation_class": "compute",
            "account_id": account_id,
            "purpose": (
                f"Post-freeze report-only RSNA pediatric external validation for "
                f"{model_id} R{resolution}. Inference only; frozen five-fold weights "
                "and original validation thresholds; no training, HPO, adaptation, "
                "calibration fitting, threshold tuning, selection, or reselection."
            ),
            "service": "kernels.KernelsApiService",
            "method": "SaveKernel",
            "body": {
                "slug": kernel_ref,
                "newTitle": (
                    f"Phase2 Finalist RSNA {model_id} R{resolution} V1 {args.run_id}"
                ),
                "text": code,
                "language": "python",
                "kernelType": "script",
                "kernelExecutionType": "SAVE_AND_RUN_ALL",
                "isPrivate": True,
                "enableGpu": True,
                "enableTpu": False,
                "enableInternet": False,
                "kernelDataSources": [],
                "datasetDataSources": [state_handle, DATASET_REF],
                "competitionDataSources": [],
                "modelDataSources": [],
            },
        }
        compute_mode = "GPU"
        launch = legacy.post_json(
            legacy.ACTION_ENDPOINT, action_token, launch_payload, timeout=240
        )
        provider_result = (
            launch.get("result")
            if isinstance(launch.get("result"), dict)
            else {}
        )
        provider_error = str(
            provider_result.get("error") or launch.get("error") or ""
        )
        if "maximum weekly gpu quota" in provider_error.casefold():
            launch_payload["request_id"] += "-cpu-fallback"
            launch_payload["body"]["enableGpu"] = False
            compute_mode = "CPU_FALLBACK_GPU_QUOTA"
            launch = legacy.post_json(
                legacy.ACTION_ENDPOINT, action_token, launch_payload, timeout=240
            )
            provider_result = (
                launch.get("result")
                if isinstance(launch.get("result"), dict)
                else {}
            )
            provider_error = str(
                provider_result.get("error") or launch.get("error") or ""
            )
        launch["cgp_compute_mode"] = compute_mode
        legacy.write_json(launch_path, launch)
        if not launch.get("ok") or provider_error:
            raise SystemExit("PHASE2_FINALIST_RSNA_SUBMISSION_REJECTED")
        provider_ref = legacy.normalize_kernel_ref(
            launch.get("provider_ref")
            or provider_result.get("ref")
            or kernel_ref,
            kernel_ref,
            owner,
        )

    history = []
    terminal = ""
    last_refresh = time.monotonic()
    for attempt in range(1, 541):
        if time.monotonic() - last_refresh > 300:
            read_token = legacy.issue_oidc("cgp-control-plane-v3")
            last_refresh = time.monotonic()
        row = legacy.post_json(
            legacy.READ_ENDPOINT,
            read_token,
            {
                "action": "resolved_kernel_status",
                "account_id": account_id,
                "kernel_ref": provider_ref,
            },
            timeout=120,
        )
        if (
            row.get("http_status") == 403
            and "jwt expired" in str(row.get("error") or "").casefold()
        ):
            read_token = legacy.issue_oidc("cgp-control-plane-v3")
            last_refresh = time.monotonic()
            row = legacy.post_json(
                legacy.READ_ENDPOINT,
                read_token,
                {
                    "action": "resolved_kernel_status",
                    "account_id": account_id,
                    "kernel_ref": provider_ref,
                },
                timeout=120,
            )
        history.append(row)
        terminal = legacy.recursive_status(row)
        print(
            "PHASE2_FINALIST_RSNA_STATUS",
            json.dumps(
                {
                    "model_id": model_id,
                    "resolution": resolution,
                    "attempt": attempt,
                    "status": terminal or "UNKNOWN",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if terminal in {"COMPLETE", "ERROR", "CANCELLED"}:
            break
        time.sleep(20)

    status_path = runner_temp / f"{model_id}_R{resolution}_RSNA_STATUS.json"
    legacy.write_json(status_path, history)
    if terminal != "COMPLETE":
        raise SystemExit(
            "PHASE2_FINALIST_RSNA_TERMINAL_" + (terminal or "TIMEOUT")
        )

    if time.monotonic() - last_refresh > 240:
        read_token = legacy.issue_oidc("cgp-control-plane-v3")

    items = []
    kernel_owner, kernel_slug = provider_ref.split("/", 1)
    for page in range(1, 31):
        listing = legacy.post_json(
            legacy.READ_ENDPOINT,
            read_token,
            {
                "action": "raw_read",
                "account_id": account_id,
                "service": "kernels.KernelsApiService",
                "method": "ListKernelSessionOutput",
                "body": {
                    "userName": kernel_owner,
                    "kernelSlug": kernel_slug,
                    "page": page,
                    "pageSize": 100,
                },
            },
            timeout=120,
        )
        if not listing.get("ok"):
            if (
                listing.get("http_status") == 403
                and "jwt expired" in str(listing.get("error") or "").casefold()
            ):
                read_token = legacy.issue_oidc("cgp-control-plane-v3")
                listing = legacy.post_json(
                    legacy.READ_ENDPOINT,
                    read_token,
                    {
                        "action": "raw_read",
                        "account_id": account_id,
                        "service": "kernels.KernelsApiService",
                        "method": "ListKernelSessionOutput",
                        "body": {
                            "userName": kernel_owner,
                            "kernelSlug": kernel_slug,
                            "page": page,
                            "pageSize": 100,
                        },
                    },
                    timeout=120,
                )
            if not listing.get("ok"):
                raise SystemExit("PHASE2_FINALIST_RSNA_OUTPUT_LIST_FAILED")
        result = legacy._read_payload(listing) if hasattr(legacy, "_read_payload") else None
        if isinstance(result, dict):
            files = result.get("files") or []
        else:
            current = listing
            for _ in range(4):
                nested = current.get("result") if isinstance(current, dict) else None
                if not isinstance(nested, dict):
                    break
                current = nested
            files = current.get("files") or []
        items.extend(files)
        if len(files) < 100:
            break

    receipt_suffix = f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_TERMINAL_RECEIPT.json"
    zip_suffix = (
        f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_R{resolution}_V1_COMPLETE.zip"
    )
    receipt_item = next(
        (x for x in items if str(x.get("fileName", "")).endswith(receipt_suffix)),
        None,
    )
    zip_item = next(
        (x for x in items if str(x.get("fileName", "")).endswith(zip_suffix)),
        None,
    )
    if not receipt_item or not receipt_item.get("url"):
        raise SystemExit("PHASE2_FINALIST_RSNA_TERMINAL_RECEIPT_NOT_FOUND")
    if not zip_item or not zip_item.get("url"):
        raise SystemExit("PHASE2_FINALIST_RSNA_COMPLETE_ZIP_NOT_FOUND")

    receipt_path = runner_temp / receipt_suffix
    zip_path = runner_temp / zip_suffix
    legacy.safe_download(str(receipt_item["url"]), receipt_path)
    legacy.safe_download(str(zip_item["url"]), zip_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_receipt(
        receipt,
        contract,
        selection_freeze_sha256=selection_freeze_sha256,
    )

    handoff = {
        "schema": "pneumonia.phase2.finalist.external.github_handoff.v1",
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "model_id": model_id,
        "resolution": resolution,
        "account_id": account_id,
        "owner": owner,
        "kernel_ref": provider_ref,
        "state_handle": state_handle,
        "compute_mode": compute_mode,
        "external_dataset": DATASET_REF,
        "external_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "selection_freeze_sha256": selection_freeze_sha256,
        "source_evaluator_sha256": source_sha256,
        "terminal_receipt_sha256": receipt["receipt_sha256"],
        "complete_zip_sha256": sha256_bytes(zip_path.read_bytes()),
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
        "external_calibration_fitting": False,
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "external_may_reselect": False,
    }
    legacy.write_json(
        runner_temp / f"{model_id}_R{resolution}_RSNA_HANDOFF.json",
        handoff,
    )
    print(
        "PHASE2_FINALIST_RSNA_EXTERNAL_PASS",
        json.dumps(handoff, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
