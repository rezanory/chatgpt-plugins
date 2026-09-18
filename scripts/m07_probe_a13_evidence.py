#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
OIDC_AUDIENCE = "cgp-control-plane-v3"
ACCOUNT_ID = "kg-05"
KERNEL_REF = "trickermark/m07-runtime-r320-a13-20260917-35265801216"
OWNER, KERNEL_SLUG = KERNEL_REF.split("/", 1)
EXPECTED_SPLIT = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
EXPECTED_RECIPE = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"
RUNTIME_NAME = "M07_RUNTIME_R320_A13_RECEIPT.json"
CAMPAIGN_PATH = "M07_GATE_MULTIRES_V17/STATE/R320/CAMPAIGN_STATE.json"
PARTIAL_PATH = "M07_GATE_MULTIRES_V17/RUNS/R320/PARTIAL_RUN_RECEIPT.json"
FOLD5_PATH = "M07_GATE_MULTIRES_V17/RUNS/R320/FOLDS/fold_5/COMPLETED.json"
REQUESTED = [RUNTIME_NAME, CAMPAIGN_PATH, PARTIAL_PATH, FOLD5_PATH]


def canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_oidc_token() -> str:
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "").strip()
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "").strip()
    if not request_url or len(request_token) < 20:
        raise RuntimeError("GitHub OIDC request context unavailable")
    sep = "&" if "?" in request_url else "?"
    target = request_url + sep + "audience=" + urllib.parse.quote(OIDC_AUDIENCE, safe="")
    last_error: Exception | None = None
    for attempt in range(1, 5):
        request = urllib.request.Request(
            target,
            headers={"Authorization": "Bearer " + request_token, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8", "replace") or "{}")
            token = str(payload.get("value") or "").strip()
            if len(token) < 100:
                raise RuntimeError("GitHub OIDC token unavailable")
            return token
        except urllib.error.HTTPError as exc:
            transient = exc.code in {408, 429} or 500 <= exc.code <= 599
            if not transient or attempt >= 4:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt >= 4:
                raise
            last_error = exc
        time.sleep(5 * attempt)
    raise RuntimeError("GitHub OIDC refresh exhausted") from last_error


class ReadBroker:
    def __init__(self) -> None:
        self.token = request_oidc_token()

    def read(self, payload: dict, *, timeout: int = 120) -> dict:
        def request_once(token: str) -> dict:
            request = urllib.request.Request(
                READ_ENDPOINT,
                data=json.dumps(payload, separators=(",", ":")).encode(),
                method="POST",
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "m07-a13-readonly-evidence-probe/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8", "replace") or "{}")
            if value.get("ok") is not True or value.get("read_only") is not True:
                raise RuntimeError("Control-plane read broker did not return ok/read_only")
            result = value.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Control-plane read broker result missing")
            return result

        try:
            return request_once(self.token)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4000).decode("utf-8", "replace")
            if exc.code == 403 and "GitHub OIDC JWT expired" in detail:
                self.token = request_oidc_token()
                return request_once(self.token)
            raise RuntimeError(f"Control-plane read HTTP {exc.code}: {detail[:1600]}") from exc


def seal_diagnostic(payload: dict) -> dict:
    receipt_sha = str(payload.get("receipt_sha256") or "").lower()
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    expected_receipt_sha = canonical_fingerprint(body)
    contract = payload.get("run_contract")
    run_fp = str(payload.get("run_fingerprint") or "").lower()
    expected_run_fp = canonical_fingerprint(contract) if isinstance(contract, dict) else ""
    return {
        "observed_receipt_sha256": receipt_sha,
        "expected_receipt_sha256": expected_receipt_sha,
        "receipt_seal_matches": bool(receipt_sha) and receipt_sha == expected_receipt_sha,
        "observed_run_fingerprint": run_fp,
        "expected_run_fingerprint": expected_run_fp,
        "run_fingerprint_matches": bool(expected_run_fp) and run_fp == expected_run_fp,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    out = pathlib.Path(os.environ.get("M07_EVIDENCE_DIR", "m07-a13-readonly-evidence"))
    out.mkdir(parents=True, exist_ok=True)
    broker = ReadBroker()

    session = broker.read(
        {
            "action": "raw_read",
            "account_id": ACCOUNT_ID,
            "service": "kernels.KernelsApiService",
            "method": "GetKernelSessionStatus",
            "body": {"userName": OWNER, "kernelSlug": KERNEL_SLUG},
        },
        timeout=60,
    )
    session_text = json.dumps(session, ensure_ascii=False, sort_keys=True)
    require("COMPLETE" in session_text.upper(), "A13 is not terminal COMPLETE")

    output = broker.read(
        {
            "action": "output_json_files",
            "account_id": ACCOUNT_ID,
            "kernel_ref": KERNEL_REF,
            "file_names": REQUESTED,
            "max_bytes_per_file": 262144,
        },
        timeout=180,
    )
    require(output.get("account_id") == ACCOUNT_ID, "output broker account drift")
    require(output.get("kernel_ref") == KERNEL_REF, "output broker kernel drift")
    require(output.get("signed_urls_returned") is False, "signed output URLs escaped Worker")

    files = output.get("files")
    require(isinstance(files, list) and len(files) == len(REQUESTED), "exact evidence file count mismatch")
    by_requested: dict[str, dict] = {}
    for item in files:
        require(isinstance(item, dict), "invalid output item")
        requested = str(item.get("file_name") or "")
        require(requested in REQUESTED and requested not in by_requested, "unexpected/duplicate output item")
        source = str(item.get("source_file_name") or "")
        value = item.get("json")
        require(isinstance(value, dict), f"JSON object missing for {requested}")
        if "/" in requested:
            require(source == requested, f"exact source path drift for {requested}: {source}")
        else:
            require(pathlib.PurePosixPath(source).name == requested, f"source basename drift for {requested}")
        by_requested[requested] = {"source_file_name": source, "json": value}
    require(set(by_requested) == set(REQUESTED), "requested evidence set mismatch")

    safe_names = {
        RUNTIME_NAME: "runtime-receipt.json",
        CAMPAIGN_PATH: "campaign-state.json",
        PARTIAL_PATH: "partial-run-receipt.json",
        FOLD5_PATH: "fold5-completed.json",
    }
    diagnostics = {}
    for requested, record in by_requested.items():
        value = record["json"]
        (out / safe_names[requested]).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        diagnostics[requested] = seal_diagnostic(value)

    runtime = by_requested[RUNTIME_NAME]["json"]
    campaign = by_requested[CAMPAIGN_PATH]["json"]
    partial = by_requested[PARTIAL_PATH]["json"]
    fold5 = by_requested[FOLD5_PATH]["json"]

    require(runtime.get("status") == "PARTIAL_FOLD_UNIT_COMPLETE", "runtime status drift")
    require(runtime.get("model_id") == "M07" and runtime.get("resolution") == 320, "runtime identity drift")
    require(runtime.get("attempt") == 13, "runtime attempt drift")
    require(runtime.get("restored_folds") == 4 and runtime.get("max_new_folds") == 1, "runtime restore/new-fold bound drift")
    require(runtime.get("completed_fold") == 5 and runtime.get("new_folds_completed") == 1, "runtime Fold5 boundary drift")
    require(runtime.get("five_fold_ready") is True, "runtime five-fold boundary not ready")
    require(runtime.get("next_action") == "VALIDATE_FIVE_FOLD_STATE_BEFORE_LOCKED_TEST", "runtime next-action drift")
    require(runtime.get("locked_test_started") is False, "runtime crossed Locked-Test boundary")

    require(campaign.get("schema") == "phase2.state.v2", "campaign schema drift")
    require(campaign.get("status") == "COMPLETE", "campaign state not COMPLETE")
    require(campaign.get("model_id") == "M07" and campaign.get("resolution") == 320, "campaign identity drift")
    campaign_artifacts = campaign.get("artifact_sha256")
    require(isinstance(campaign_artifacts, dict), "campaign artifact map missing")
    expected_archives = {f"FOLD_{fold}_RECOVERY.zip" for fold in range(1, 6)}
    require(set(campaign_artifacts) == expected_archives, "campaign recovery archive inventory is not exactly folds 1-5")

    require(partial.get("status") == "PARTIAL_FOLD_UNIT_COMPLETE", "partial receipt status drift")
    require(partial.get("model_id") == "M07" and partial.get("resolution") == 320, "partial identity drift")
    require(partial.get("completed_fold") == 5 and partial.get("new_folds_completed") == 1, "partial Fold5 boundary drift")
    require(partial.get("five_fold_ready") is True, "partial five-fold boundary not ready")
    require(partial.get("next_action") == "VALIDATE_FIVE_FOLD_STATE_BEFORE_LOCKED_TEST", "partial next-action drift")

    require(fold5.get("status") == "COMPLETED", "Fold5 receipt status is not COMPLETED")
    require(fold5.get("model_id") == "M07" and fold5.get("resolution") == 320, "Fold5 identity drift")
    require(fold5.get("fold_id") == 5, "Fold5 receipt fold identity drift")
    require(fold5.get("locked_test_used_for_training") is False, "Fold5 used Locked Test for training")
    require(fold5.get("external_used_for_training") is False, "Fold5 used external data for training")

    for label, payload in {
        "runtime": runtime,
        "campaign": campaign,
        "partial": partial,
        "fold5": fold5,
    }.items():
        contract = payload.get("run_contract")
        require(isinstance(contract, dict), f"{label} run contract missing")
        require(contract.get("split_fingerprint") == EXPECTED_SPLIT, f"{label} split fingerprint drift")
        extra = contract.get("extra")
        require(isinstance(extra, dict), f"{label} run-contract extra missing")
        require(extra.get("recipe_fingerprint") == EXPECTED_RECIPE, f"{label} recipe fingerprint drift")

    require(runtime["run_contract"] == partial["run_contract"] == campaign["run_contract"],
            "phase2_campaign run contract differs across A13 receipts")
    require(runtime.get("run_fingerprint") == partial.get("run_fingerprint") == campaign.get("run_fingerprint"),
            "phase2_campaign observed run fingerprint differs across A13 receipts")
    require(
        runtime.get("artifact_sha256", {}).get("campaign_state")
        == partial.get("artifact_sha256", {}).get("campaign_state"),
        "runtime/partial campaign-state artifact hash drift",
    )

    summary = {
        "schema": "m07.d260914d.a13.readonly-evidence-probe.v1",
        "status": "EVIDENCE_RECOVERED_FAIL_CLOSED",
        "kernel_ref": KERNEL_REF,
        "account_id": ACCOUNT_ID,
        "requested_files": REQUESTED,
        "source_files": {name: by_requested[name]["source_file_name"] for name in REQUESTED},
        "session_complete": True,
        "scientific_boundary": {
            "restored_folds": runtime["restored_folds"],
            "max_new_folds": runtime["max_new_folds"],
            "completed_fold": runtime["completed_fold"],
            "new_folds_completed": runtime["new_folds_completed"],
            "five_fold_ready": runtime["five_fold_ready"],
            "locked_test_started": runtime["locked_test_started"],
            "campaign_recovery_archives": sorted(campaign_artifacts),
            "fold5_locked_test_used_for_training": fold5["locked_test_used_for_training"],
            "fold5_external_used_for_training": fold5["external_used_for_training"],
        },
        "seal_diagnostics": diagnostics,
        "phase2_campaign_contract_equal": True,
        "phase2_campaign_observed_run_fingerprint_equal": True,
        "runtime_partial_campaign_state_hash_equal": True,
        "source_sha": os.environ.get("GITHUB_SHA", ""),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "acceptance": "BLOCKED_UNTIL_SEAL_MISMATCH_IS_EXPLAINED_OR_RECONCILED",
    }
    (out / "evidence-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print("M07_A13_READONLY_EVIDENCE_RECOVERED", json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
