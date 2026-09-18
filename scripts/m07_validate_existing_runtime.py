#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

from m07_runtime_contract import (
    canonical_fingerprint,
    extract_session_state,
    expected_runtime_contract,
    validate_runtime_receipt,
)

READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
OIDC_AUDIENCE = "cgp-control-plane-v3"

# A13 was generated from this exact immutable notebook contract. The Cloudflare
# read broker parses nested JSON before wrapping it in another JSON response.
# JavaScript JSON.parse/JSON.stringify erases the lexical distinction between
# integral floats such as 2.0 and integers such as 2. The runtime canonical
# seal, however, was computed by Python before that transport conversion.
#
# These are the ONLY integral-float fields in the frozen V1.7 phase2_campaign
# contract that the A13 source defines as floats. Repair is allowed only for the
# exact A13 object and only if restoring these three types exactly reproduces
# BOTH the embedded run_fingerprint and the embedded receipt_sha256.
A13_TOKEN = "M07_RUNTIME_M07_R320_A13"
A13_KERNEL_REF = "trickermark/m07-runtime-r320-a13-20260917-35265801216"
A13_CODE_SHA256 = "d099b6e51c243b3c489f0a280ca247971e84b1cf1c504cba2a51028c93fba27f"
A13_SPLIT_FINGERPRINT = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
A13_RECIPE_FINGERPRINT = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"


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

        wait_seconds = 5 * attempt
        print(
            "M07_EXISTING_RUNTIME_OIDC_RETRY",
            json.dumps(
                {
                    "attempt": attempt,
                    "wait_seconds": wait_seconds,
                    "error": type(last_error).__name__ if last_error else "unknown",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(wait_seconds)

    raise RuntimeError("GitHub OIDC refresh exhausted") from last_error


class ReadBroker:
    def __init__(self) -> None:
        self.token = request_oidc_token()

    def read(self, payload: dict, *, timeout: int = 90) -> dict:
        def request_once(token: str) -> dict:
            request = urllib.request.Request(
                READ_ENDPOINT,
                data=json.dumps(payload, separators=(",", ":")).encode(),
                method="POST",
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "m07-existing-runtime-validator/1.1",
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


def _seal_diagnostic(scientific: dict) -> dict:
    receipt_body = {key: value for key, value in scientific.items() if key != "receipt_sha256"}
    observed_receipt_sha = str(scientific.get("receipt_sha256") or "").lower()
    expected_receipt_sha = canonical_fingerprint(receipt_body)
    run_contract = scientific.get("run_contract") if isinstance(scientific.get("run_contract"), dict) else {}
    observed_run_fp = str(scientific.get("run_fingerprint") or "").lower()
    expected_run_fp = canonical_fingerprint(run_contract) if run_contract else ""
    return {
        "observed_receipt_sha256": observed_receipt_sha,
        "expected_receipt_sha256": expected_receipt_sha,
        "receipt_seal_matches": bool(observed_receipt_sha) and observed_receipt_sha == expected_receipt_sha,
        "observed_run_fingerprint": observed_run_fp,
        "expected_run_fingerprint": expected_run_fp,
        "run_fingerprint_matches": bool(expected_run_fp) and observed_run_fp == expected_run_fp,
    }


def _restore_a13_integral_float_types(
    scientific: dict,
    *,
    token: str,
    kernel_ref: str,
) -> tuple[dict, dict]:
    if token != A13_TOKEN or kernel_ref != A13_KERNEL_REF:
        raise RuntimeError("Transport number-type repair is restricted to the exact immutable A13 object")

    contract = scientific.get("run_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("A13 run contract missing")
    extra = contract.get("extra")
    if not isinstance(extra, dict):
        raise RuntimeError("A13 run-contract extra missing")
    if contract.get("code_sha256") != A13_CODE_SHA256:
        raise RuntimeError("A13 code SHA drift; transport repair forbidden")
    if contract.get("split_fingerprint") != A13_SPLIT_FINGERPRINT:
        raise RuntimeError("A13 split fingerprint drift; transport repair forbidden")
    if extra.get("recipe_fingerprint") != A13_RECIPE_FINGERPRINT:
        raise RuntimeError("A13 recipe fingerprint drift; transport repair forbidden")

    before = _seal_diagnostic(scientific)
    if before["receipt_seal_matches"] or before["run_fingerprint_matches"]:
        raise RuntimeError("A13 transport repair requires the exact dual seal/fingerprint mismatch pattern")

    repaired = copy.deepcopy(scientific)
    repaired_extra = repaired["run_contract"]["extra"]

    # Fail closed if any field is already a float, a bool, missing, or has a
    # value other than the exact integer produced by JS number normalization.
    static_value = repaired_extra.get("static_focal_gamma")
    if isinstance(static_value, bool) or type(static_value) is not int or static_value != 2:
        raise RuntimeError("A13 static_focal_gamma transport pattern mismatch")
    repaired_extra["static_focal_gamma"] = 2.0

    flsd = repaired_extra.get("flsd")
    if not isinstance(flsd, list) or len(flsd) != 3:
        raise RuntimeError("A13 FLSD transport pattern mismatch")
    if isinstance(flsd[1], bool) or type(flsd[1]) is not int or flsd[1] != 5:
        raise RuntimeError("A13 hard-gamma transport pattern mismatch")
    if isinstance(flsd[2], bool) or type(flsd[2]) is not int or flsd[2] != 3:
        raise RuntimeError("A13 easy-gamma transport pattern mismatch")
    flsd[1] = 5.0
    flsd[2] = 3.0

    after = _seal_diagnostic(repaired)
    if not after["run_fingerprint_matches"] or not after["receipt_seal_matches"]:
        raise RuntimeError("A13 transport type restoration did not reproduce both original seals")

    reconciliation = {
        "schema": "m07.d260914d.runtime.transport-number-type-reconciliation.v1",
        "status": "PASS",
        "kernel_ref": kernel_ref,
        "runtime_token": token,
        "cause": "JS_JSON_NUMBER_NORMALIZATION_OF_INTEGRAL_FLOATS",
        "restored_fields": [
            {"path": "/run_contract/extra/static_focal_gamma", "transport_value": 2, "sealed_python_value": 2.0},
            {"path": "/run_contract/extra/flsd/1", "transport_value": 5, "sealed_python_value": 5.0},
            {"path": "/run_contract/extra/flsd/2", "transport_value": 3, "sealed_python_value": 3.0},
        ],
        "before": before,
        "after": after,
        "observed_run_fingerprint_preserved": scientific.get("run_fingerprint"),
        "observed_receipt_sha256_preserved": scientific.get("receipt_sha256"),
        "acceptance_rule": "repair is exact-object-bound and accepted only because both embedded hashes reproduce exactly",
    }
    return repaired, reconciliation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--expected-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    expected = expected_runtime_contract(args.token)
    owner, kernel_slug = args.kernel_ref.split("/", 1)
    if owner.lower() != str(expected["owner"]).lower():
        raise SystemExit("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: runtime owner drift")
    if args.account_id != expected["account_id"]:
        raise SystemExit("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: runtime account drift")

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    broker = ReadBroker()

    session = broker.read(
        {
            "action": "raw_read",
            "account_id": args.account_id,
            "service": "kernels.KernelsApiService",
            "method": "GetKernelSessionStatus",
            "body": {"userName": owner, "kernelSlug": kernel_slug},
        },
        timeout=60,
    )
    state = extract_session_state(session)
    if state != "COMPLETE":
        (out / "existing-runtime-status.json").write_text(
            json.dumps(
                {"kernel_ref": args.kernel_ref, "terminal_state": state, "session": session},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise SystemExit(f"M07_EXISTING_RUNTIME_NOT_COMPLETE:{state}")

    output = broker.read(
        {
            "action": "output_json_files",
            "account_id": args.account_id,
            "kernel_ref": args.kernel_ref,
            "file_names": [args.expected_receipt],
            "max_bytes_per_file": 262144,
        },
        timeout=120,
    )
    if output.get("account_id") != args.account_id or output.get("kernel_ref") != args.kernel_ref:
        raise SystemExit("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: output broker identity drift")
    files = output.get("files")
    if not isinstance(files, list) or len(files) != 1:
        count = len(files) if isinstance(files, list) else -1
        raise SystemExit(
            f"BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: runtime_receipt_matches={count} "
            f"expected={args.expected_receipt}"
        )
    item = files[0]
    if not isinstance(item, dict) or item.get("file_name") != args.expected_receipt:
        raise SystemExit("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: exact runtime receipt name mismatch")
    source_file_name = str(item.get("source_file_name") or "")
    if pathlib.PurePosixPath(source_file_name).name != args.expected_receipt:
        raise SystemExit("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: runtime receipt source basename mismatch")
    scientific = item.get("json")
    if not isinstance(scientific, dict):
        raise SystemExit("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: runtime receipt JSON object missing")

    (out / "output-inventory.json").write_text(
        json.dumps(
            {
                "kernel_ref": args.kernel_ref,
                "file_names": [args.expected_receipt],
                "source_file_name": source_file_name,
                "signed_urls_returned": output.get("signed_urls_returned"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / args.expected_receipt).write_text(
        json.dumps(scientific, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    before = _seal_diagnostic(scientific)
    (out / "receipt-seal-diagnostic.json").write_text(
        json.dumps(
            {
                "schema": "m07.d260914d.runtime.receipt-seal-diagnostic.v2",
                "status": "PASS" if before["receipt_seal_matches"] else "SEAL_MISMATCH",
                "kernel_ref": args.kernel_ref,
                "runtime_token": args.token,
                "source_file_name": source_file_name,
                **before,
                "top_level_keys": sorted(scientific),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scientific_for_validation = scientific
    reconciliation = None
    if not before["receipt_seal_matches"] or not before["run_fingerprint_matches"]:
        scientific_for_validation, reconciliation = _restore_a13_integral_float_types(
            scientific,
            token=args.token,
            kernel_ref=args.kernel_ref,
        )
        (out / "transport-number-type-reconciliation.json").write_text(
            json.dumps(reconciliation, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (out / "reconciled-runtime-receipt.json").write_text(
            json.dumps(scientific_for_validation, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    report = validate_runtime_receipt(scientific_for_validation, token=args.token)
    evidence = {
        "schema": "m07.d260914d.runtime.existing-output-validation.v2",
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "kernel_ref": args.kernel_ref,
        "terminal_state": state,
        "runtime_token": args.token,
        "scientific_report": report,
        "transport_reconciliation": reconciliation,
        "read_transport": "control-plane-v3-oidc",
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "source_sha": os.environ.get("GITHUB_SHA", ""),
    }
    (out / "existing-runtime-validation.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("M07_EXISTING_RUNTIME_SCIENTIFIC_PASS", json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
