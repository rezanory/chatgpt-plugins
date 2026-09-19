#!/usr/bin/env python3
"""Deterministic M07 runtime identity and evidence guards.

This module is intentionally dependency-free so the workflow can validate a generated
runtime notebook before any Kaggle compute is submitted and can validate the runtime
receipt after the exact kernel reaches a terminal state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "d260914d"
PERSISTENCE_HANDLE = "rezanory/m07-final-5fold-fix2-d260914d"
R320_BRIDGE_HANDLE = "trickermark/m07-r320-producer-v4-bridge-e3884dd1/versions/1"
R320_BRIDGE_DATASET = "trickermark/m07-r320-producer-v4-bridge-e3884dd1"
R384_STATE_HANDLE = "trickermark/m07-gate-r384-state-v1-7"
RAW_DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"
SPLIT_FINGERPRINT = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
RECIPE_FINGERPRINT = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"
TOKEN_RE = re.compile(r"M07_RUNTIME_M07_R(320|384)_A([0-9]{2})")
RUNTIME_KERNEL_RE = re.compile(
    r"(?P<owner>[A-Za-z0-9._-]+)/m07-runtime-r(?P<resolution>320|384)-"
    r"a(?P<attempt>[0-9]{2})-(?P<date>[0-9]{8})-(?P<run_id>[0-9]+)"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCEL", "CANCELLED", "CANCELED"}
ACTIVE_STATES = {"RUNNING", "QUEUED", "PENDING"}


class ContractError(RuntimeError):
    pass


def parse_runtime_token(token: str) -> tuple[int, int]:
    match = TOKEN_RE.fullmatch(str(token or "").strip())
    if match is None:
        raise ContractError(f"invalid M07 runtime token: {token!r}")
    resolution = int(match.group(1))
    attempt = int(match.group(2))
    if not 1 <= attempt <= 99:
        raise ContractError("runtime attempt must be between 1 and 99")
    if resolution == 320 and attempt < 7:
        raise ContractError("historical R320 attempts below A07 are immutable and cannot be relaunched")
    return resolution, attempt


def expected_runtime_contract(token: str) -> dict[str, Any]:
    resolution, attempt = parse_runtime_token(token)
    return {
        "resolution": resolution,
        "attempt": attempt,
        "account_id": "kg-05",
        "owner": "trickermark",
        "source_phase2_state_handle": R320_BRIDGE_HANDLE if resolution == 320 else None,
        "source_persistence_handle": PERSISTENCE_HANDLE,
        "max_new_folds": 1,
        "machine_shape": "NvidiaTeslaT4",
        "execution_scope": "SETUP_DEFINITIONS_EXACT_RUNTIME_STAGE_ONLY",
        "pre_runtime_analysis_cells_removed": True,
        "legacy_r224_preflight_removed": True,
        "bridge_restore_mode": (
            "PREATTACHED_READONLY_MOUNT" if resolution == 320 else None
        ),
    }


def runtime_dataset_sources(token: str) -> list[str]:
    """Return exact pre-attached datasets required by one immutable runtime attempt."""
    resolution, attempt = parse_runtime_token(token)
    sources = [RAW_DATASET_HANDLE, PERSISTENCE_HANDLE]
    if resolution == 320:
        sources.append(R320_BRIDGE_DATASET)
    elif resolution == 384 and attempt > 1:
        sources.append(R384_STATE_HANDLE)
    return sources


def validate_runtime_state_dataset(payload: dict[str, Any], *, token: str) -> dict[str, Any]:
    """Fail closed unless the required R384 continuation state is an exact Ready version."""
    resolution, attempt = parse_runtime_token(token)
    required = resolution == 384 and attempt > 1
    if not required:
        return {"status": "NOT_REQUIRED", "required": False}
    _require(isinstance(payload, dict), "runtime state dataset payload is not an object")
    _require(payload.get("ref") == R384_STATE_HANDLE, "R384 runtime state dataset ref mismatch")
    _require(
        str(payload.get("ownerRef") or "").lower() == "trickermark",
        "R384 runtime state owner mismatch",
    )
    version = int(
        payload.get("currentVersionNumber")
        or payload.get("current_version_number")
        or 0
    )
    _require(version >= 1, "R384 runtime state dataset has no published version")
    versions = payload.get("versions") or []
    ready = any(
        isinstance(item, dict)
        and int(item.get("versionNumber") or item.get("version_number") or 0) == version
        and str(item.get("status") or "").strip().upper() == "READY"
        for item in versions
    )
    _require(ready, "R384 runtime state current version is not Ready")
    return {
        "status": "PASS",
        "required": True,
        "dataset_ref": R384_STATE_HANDLE,
        "current_version_number": version,
    }


def parse_runtime_kernel_ref(kernel_ref: str) -> dict[str, Any] | None:
    match = RUNTIME_KERNEL_RE.fullmatch(str(kernel_ref or "").strip())
    if match is None:
        return None
    return {
        "owner": match.group("owner"),
        "resolution": int(match.group("resolution")),
        "attempt": int(match.group("attempt")),
        "date": match.group("date"),
        "run_id": match.group("run_id"),
        "kernel_ref": str(kernel_ref).strip(),
    }


def assess_runtime_inventory(
    *,
    token: str,
    kernels: list[dict[str, Any]],
    statuses: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on attempt reuse, stale dispatch or overlapping GPU runtime."""
    expected = expected_runtime_contract(token)
    resolution = int(expected["resolution"])
    requested_attempt = int(expected["attempt"])
    owner = str(expected["owner"])
    candidates = []
    blockers = []
    for item in kernels:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        parsed = parse_runtime_kernel_ref(ref)
        if parsed is None:
            continue
        if parsed["owner"].lower() != owner.lower() or parsed["resolution"] != resolution:
            continue
        state = extract_session_state(statuses.get(ref))
        row = {
            "kernel_ref": ref,
            "attempt": parsed["attempt"],
            "state": state,
            "lastRunTime": str(item.get("lastRunTime") or ""),
        }
        candidates.append(row)
        if parsed["attempt"] == requested_attempt:
            blockers.append({"reason": "ATTEMPT_ALREADY_USED", **row})
        elif parsed["attempt"] > requested_attempt:
            blockers.append({"reason": "NEWER_ATTEMPT_ALREADY_EXISTS", **row})
        if state in ACTIVE_STATES:
            blockers.append({"reason": "ACTIVE_RUNTIME_EXISTS", **row})

    if blockers:
        raise ContractError(
            "runtime admission blocked: "
            + json.dumps(
                {
                    "token": token,
                    "requested_attempt": requested_attempt,
                    "resolution": resolution,
                    "blockers": blockers,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return {
        "status": "PASS",
        "token": token,
        "resolution": resolution,
        "requested_attempt": requested_attempt,
        "candidate_count": len(candidates),
        "candidates": sorted(candidates, key=lambda row: (row["attempt"], row["kernel_ref"])),
        "duplicate_gpu_blocked": False,
        "attempt_reuse_blocked": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_runtime_artifact(
    *,
    token: str,
    run_id: str,
    notebook_path: Path,
    sha_path: Path,
    slug_path: Path,
    title_path: Path,
) -> dict[str, Any]:
    expected = expected_runtime_contract(token)
    run_id = str(run_id or "").strip()
    _require(run_id.isdigit(), f"invalid GitHub run id: {run_id!r}")
    _require(notebook_path.is_file(), f"runtime notebook missing: {notebook_path}")
    _require(sha_path.is_file(), f"runtime SHA sidecar missing: {sha_path}")
    _require(slug_path.is_file(), f"runtime slug sidecar missing: {slug_path}")
    _require(title_path.is_file(), f"runtime title sidecar missing: {title_path}")

    actual_sha = sha256_file(notebook_path)
    recorded_sha = sha_path.read_text(encoding="utf-8-sig").strip().lower()
    _require(bool(SHA256_RE.fullmatch(recorded_sha)), "runtime SHA sidecar is not a SHA-256")
    _require(actual_sha == recorded_sha, "runtime notebook SHA does not match sealed sidecar")

    slug = slug_path.read_text(encoding="utf-8-sig").strip()
    title = title_path.read_text(encoding="utf-8-sig").strip()
    resolution = expected["resolution"]
    attempt = expected["attempt"]
    slug_match = re.fullmatch(
        rf"trickermark/m07-runtime-r{resolution}-a{attempt:02d}-(\d{{8}})-{re.escape(run_id)}",
        slug,
    )
    _require(slug_match is not None, f"runtime kernel ref drift: {slug}")
    runtime_date = slug_match.group(1)
    expected_title = f"M07 Runtime R{resolution} A{attempt:02d} {runtime_date} {run_id}"
    _require(title == expected_title, f"runtime title drift: {title}")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    meta = (notebook.get("metadata") or {}).get("cgp_continuation") or {}
    runtime = meta.get("runtime_partition") or {}
    _require(meta.get("campaign_id") == CAMPAIGN_ID, "runtime campaign id drift")
    _require(meta.get("persistence_handle") == PERSISTENCE_HANDLE, "runtime persistence handle drift")
    _require(meta.get("locked_test_used") is False, "locked test must not be used by runtime training")
    _require(meta.get("external_used") is False, "external data must not be used by runtime training")
    _require(meta.get("phase2_unlocked") is True, "runtime notebook must be in explicit approved runtime mode")
    _require(runtime.get("enabled") is True, "runtime partition is not enabled")
    _require(
        runtime.get("execution_scope") == expected["execution_scope"],
        f"runtime execution scope drift: {runtime.get('execution_scope')!r}",
    )
    _require(
        runtime.get("pre_runtime_analysis_cells_removed") is True,
        "pre-runtime analysis cells were not removed",
    )
    _require(
        runtime.get("legacy_r224_preflight_removed") is True,
        "legacy R224 runtime preflight was not removed",
    )
    _require(
        runtime.get("bridge_restore_mode") == expected["bridge_restore_mode"],
        f"runtime bridge restore mode drift: {runtime.get('bridge_restore_mode')!r}",
    )

    for key in (
        "resolution",
        "attempt",
        "account_id",
        "owner",
        "source_phase2_state_handle",
        "source_persistence_handle",
        "max_new_folds",
    ):
        _require(runtime.get(key) == expected[key], f"runtime.{key} drift: {runtime.get(key)!r}")

    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    _require("CGP_MAX_NEW_FOLDS_PER_RUN = 1" in source, "bounded one-fold runtime guard missing")
    _require("CGP_RUNTIME_READONLY_PREDECESSOR = True" in source, "predecessor read-only guard missing")
    _require("Runtime predecessor persistence is read-only" in source, "generic predecessor upload guard missing")
    forbidden_pre_fold5_markers = (
        "# 8B) M07 RUNTIME PRE-FLIGHT",
        "# 10) FINAL M07",
        "# 11) M07 OOF",
        "# 12) ONE-TIME LOCKED TEST",
        "# 12B) STATISTICAL / CALIBRATION / COVERAGE DIAGNOSTICS",
        "# 13) COMPLETE FIGURE SUITE",
        "# 14) VALIDATION XAI",
        "# 14B) M07 MODEL-INTERNAL VISUALS",
        "# 14C) LOCKED-TEST MISCLASSIFICATION CASEBOOK",
        "# 15) EXTERNAL VALIDATION",
        "# 16) FINAL M07 REPLICATION GATE REPORT + EVIDENCE PACKAGE",
    )
    for marker in forbidden_pre_fold5_markers:
        _require(marker not in source, f"forbidden pre-Fold5 execution cell retained: {marker}")
    if resolution == 320:
        _require(R320_BRIDGE_HANDLE in source, "exact R320 bridge handle missing")
        _require("def _m07_resolve_exact_bridge_mount" in source, "bridge mount resolver missing")
        _require(
            "restore_root = _m07_resolve_exact_bridge_mount(model_id, resolution)" in source,
            "exact R320 restore is not routed through the pre-attached bridge mount",
        )
        _require(
            "M07_R320_BRIDGE_MOUNT_VERIFIED" in source,
            "bridge mount evidence marker missing",
        )
        _require(
            "work_root=temp" in source,
            "bridge restore does not isolate runtime writes from the read-only mount",
        )
        _require("M07_R320_BRIDGE_EXTRACTED_RESTORE_VERIFIED" in source, "bridge restore evidence marker missing")
        _require(
            "def _resolve_runtime_predecessor_mount" in source,
            "runtime predecessor mount resolver missing",
        )
        _require(
            "downloaded_root = _resolve_runtime_predecessor_mount()" in source,
            "runtime predecessor restore is not routed through the pre-attached mount",
        )
        _require(
            "M07_RUNTIME_PREDECESSOR_MOUNT_VERIFIED" in source,
            "runtime predecessor mount evidence marker missing",
        )
    if resolution == 384 and attempt > 1:
        _require(R384_STATE_HANDLE in source, "exact R384 state dataset handle missing")
        _require(
            "def _m07_resolve_r384_state_mount" in source,
            "R384 pre-attached state mount resolver missing",
        )
        _require(
            "def _m07_restore_r384_expanded_state" in source,
            "R384 expanded-state restore helper missing",
        )
        _require(
            "r384_restored = _m07_restore_r384_expanded_state(model_id, resolution)" in source,
            "R384 phase2_restore is not routed through expanded-state compatibility",
        )
        _require(
            "M07_R384_EXPANDED_STATE_RESTORE_VERIFIED" in source,
            "R384 expanded-state evidence marker missing",
        )

    return {
        "status": "PASS",
        "token": token,
        "resolution": resolution,
        "attempt": attempt,
        "kernel_ref": slug,
        "title": title,
        "notebook_sha256": actual_sha,
        "runtime_date": runtime_date,
        "source_phase2_state_handle": expected["source_phase2_state_handle"],
        "max_new_folds": 1,
        "locked_test_used": False,
        "external_used": False,
    }


def extract_session_state(payload: Any) -> str | None:
    """Return a canonical Kaggle session state without trusting arbitrary prose."""
    candidates: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and "status" in key.lower():
            candidates.append(value.strip().upper())

    visit(payload)
    for state in candidates:
        if state in TERMINAL_STATES | ACTIVE_STATES:
            return "CANCEL" if state in {"CANCELLED", "CANCELED"} else state
    return None


def validate_runtime_receipt(payload: dict[str, Any], *, token: str) -> dict[str, Any]:
    expected = expected_runtime_contract(token)
    _require(isinstance(payload, dict), "runtime receipt is not an object")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    receipt_sha = str(payload.get("receipt_sha256") or "").lower()
    _require(bool(SHA256_RE.fullmatch(receipt_sha)), "runtime receipt SHA missing/invalid")
    _require(receipt_sha == canonical_fingerprint(body), "runtime receipt seal mismatch")
    _require(payload.get("status") in {"PARTIAL_FOLD_UNIT_COMPLETE", "COMPLETE"}, "runtime stage did not succeed")
    _require(payload.get("model_id") == "M07", "runtime receipt model mismatch")
    _require(int(payload.get("resolution", -1)) == expected["resolution"], "runtime receipt resolution mismatch")
    _require(int(payload.get("attempt", -1)) == expected["attempt"], "runtime receipt attempt mismatch")
    _require(int(payload.get("max_new_folds", -1)) == 1, "runtime receipt one-fold bound missing")
    _require(payload.get("persistence_handle") == PERSISTENCE_HANDLE, "runtime receipt persistence handle mismatch")
    if expected["resolution"] == 320:
        _require(
            int(payload.get("restored_folds", -1)) == 4,
            "R320 runtime must restore exactly four sealed folds",
        )
        _require(
            payload.get("status") == "PARTIAL_FOLD_UNIT_COMPLETE",
            "R320 Fold5 recovery must stop before final/Locked-Test reporting",
        )
        _require(int(payload.get("completed_fold", -1)) == 5, "R320 recovery must complete only Fold5")
        _require(int(payload.get("new_folds_completed", -1)) == 1, "R320 recovery must train exactly one new fold")
        _require(payload.get("five_fold_ready") is True, "R320 recovery did not seal a five-fold-ready state")
        _require(
            payload.get("next_action") == "VALIDATE_FIVE_FOLD_STATE_BEFORE_LOCKED_TEST",
            "R320 recovery next-action drift",
        )
        _require(payload.get("locked_test_started") is False, "R320 Fold5 recovery crossed the Locked-Test boundary")
    contract = payload.get("run_contract") or {}
    _require(contract.get("stage") == "phase2_campaign", "runtime receipt stage mismatch")
    _require(contract.get("model_id") == "M07", "runtime run contract model mismatch")
    _require(int(contract.get("resolution", -1)) == expected["resolution"], "runtime run contract resolution mismatch")
    _require(contract.get("split_fingerprint") == SPLIT_FINGERPRINT, "runtime split fingerprint mismatch")
    extra = contract.get("extra") or {}
    _require(extra.get("recipe_fingerprint") == RECIPE_FINGERPRINT, "runtime recipe fingerprint mismatch")
    artifacts = payload.get("artifact_sha256")
    _require(isinstance(artifacts, dict), "runtime artifact manifest missing")
    if expected["resolution"] == 320:
        _require("campaign_state" in artifacts, "R320 recovery campaign-state hash missing")
        _require("final_report" not in artifacts, "R320 Fold5 recovery unexpectedly contains final-report evidence")
    return {
        "status": "PASS",
        "resolution": expected["resolution"],
        "attempt": expected["attempt"],
        "stage_status": payload.get("status"),
        "restored_folds": payload.get("restored_folds"),
        "completed_fold": payload.get("completed_fold"),
        "new_folds_completed": payload.get("new_folds_completed"),
        "five_fold_ready": payload.get("five_fold_ready"),
        "next_action": payload.get("next_action"),
        "locked_test_started": payload.get("locked_test_started"),
        "max_new_folds": payload.get("max_new_folds"),
        "receipt_sha256": receipt_sha,
    }


def _cmd_validate_artifact(args: argparse.Namespace) -> int:
    report = validate_runtime_artifact(
        token=args.token,
        run_id=args.run_id,
        notebook_path=Path(args.notebook),
        sha_path=Path(args.sha_file),
        slug_path=Path(args.slug_file),
        title_path=Path(args.title_file),
    )
    print("M07_RUNTIME_ARTIFACT_CONTRACT", json.dumps(report, sort_keys=True))
    return 0


def _cmd_validate_receipt(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.receipt).read_text(encoding="utf-8-sig"))
    report = validate_runtime_receipt(payload, token=args.token)
    print("M07_RUNTIME_SCIENTIFIC_RECEIPT", json.dumps(report, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    artifact = sub.add_parser("validate-artifact")
    artifact.add_argument("--token", required=True)
    artifact.add_argument("--run-id", required=True)
    artifact.add_argument("--notebook", required=True)
    artifact.add_argument("--sha-file", required=True)
    artifact.add_argument("--slug-file", required=True)
    artifact.add_argument("--title-file", required=True)
    artifact.set_defaults(func=_cmd_validate_artifact)
    receipt = sub.add_parser("validate-receipt")
    receipt.add_argument("--token", required=True)
    receipt.add_argument("--receipt", required=True)
    receipt.set_defaults(func=_cmd_validate_receipt)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
