#!/usr/bin/env python3
"""OIDC-backed admission and terminal verification for one Phase-2 unit."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pneumonia_phase2_unit import (
    next_token,
    unit_contract,
    validate_terminal_receipt,
)


READ_ENDPOINT = (
    "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/"
    "control-plane/v3/read/kaggle"
)
TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCEL"}
ACTIVE_STATES = {"RUNNING", "QUEUED", "PENDING", "INITIALIZING"}


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def session_state(payload: dict[str, Any]) -> str:
    text = " ".join(_walk_strings(payload)).upper()
    if "CANCEL_ACKNOWLEDGED" in text or "CANCELED" in text or "CANCELLED" in text:
        return "CANCEL"
    for state in ("COMPLETE", "ERROR", "RUNNING", "QUEUED", "PENDING", "INITIALIZING"):
        if state in text:
            return state
    return "UNKNOWN"


class OidcReadBroker:
    def __init__(self) -> None:
        self.token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
        self.issued = time.monotonic()
        if len(self.token) < 100:
            raise RuntimeError("BLOCKED_ADMISSION_AUTHORITY_MISSING: read OIDC unavailable")

    def refresh(self) -> str:
        request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "").strip()
        request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "").strip()
        if not request_url or len(request_token) < 20:
            raise RuntimeError("Phase2 OIDC refresh context unavailable")
        separator = "&" if "?" in request_url else "?"
        target = request_url + separator + "audience=" + urllib.parse.quote(
            "cgp-control-plane-v3", safe=""
        )
        request = urllib.request.Request(
            target,
            headers={"Authorization": "Bearer " + request_token, "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8", "replace") or "{}")
        token = str(payload.get("value") or "").strip()
        if len(token) < 100:
            raise RuntimeError("Phase2 OIDC refresh returned no token")
        self.token = token
        self.issued = time.monotonic()
        return token

    def read(self, payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
        if time.monotonic() - self.issued >= 120:
            self.refresh()

        def request_once(token: str) -> dict[str, Any]:
            request = urllib.request.Request(
                READ_ENDPOINT,
                data=json.dumps(payload, separators=(",", ":")).encode(),
                method="POST",
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "pneumonia-phase2-runtime/1.0",
                },
            )
            last_error: Exception | None = None
            for attempt in range(1, 5):
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        return json.loads(
                            response.read().decode("utf-8", "replace") or "{}"
                        )
                except urllib.error.HTTPError as error:
                    if error.code not in {408, 429} and not 500 <= error.code <= 599:
                        raise
                    last_error = error
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    last_error = error
                if attempt == 4:
                    break
                time.sleep(min(2 ** (attempt - 1), 4))
            raise RuntimeError("Phase2 read broker transient retry exhausted") from last_error

        try:
            envelope = request_once(self.token)
        except urllib.error.HTTPError as error:
            detail = error.read(4000).decode("utf-8", "replace")
            if error.code == 403 and "GitHub OIDC JWT expired" in detail:
                envelope = request_once(self.refresh())
            else:
                raise RuntimeError(
                    f"Phase2 read broker HTTP {error.code}: {detail[:1600]}"
                ) from error
        if envelope.get("ok") is not True or envelope.get("read_only") is not True:
            raise RuntimeError("Phase2 read broker did not return ok/read_only")
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Phase2 read broker result missing")
        return result


def _status(broker: OidcReadBroker, contract: dict[str, Any], kernel_ref: str) -> dict[str, Any]:
    owner, slug = kernel_ref.split("/", 1)
    if owner.lower() != contract["owner"].lower():
        raise RuntimeError("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: Phase2 owner drift")
    return broker.read(
        {
            "action": "raw_read",
            "account_id": contract["account_id"],
            "service": "kernels.KernelsApiService",
            "method": "GetKernelSessionStatus",
            "body": {"userName": owner, "kernelSlug": slug},
        },
        timeout=60,
    )


def admission_preflight(token: str, run_id: str) -> dict[str, Any]:
    contract = unit_contract(token, run_id)
    broker = OidcReadBroker()
    listed = broker.read(
        {
            "action": "raw_read",
            "account_id": contract["account_id"],
            "service": "kernels.KernelsApiService",
            "method": "ListKernels",
            "body": {
                "group": "PROFILE",
                "user": contract["owner"],
                "sortBy": "DATE_RUN",
                "page": 1,
                "pageSize": 100,
            },
        }
    )
    kernels = [item for item in listed.get("kernels", []) if isinstance(item, dict)]
    exact = [item for item in kernels if str(item.get("ref", "")).lower() == contract["kernel_ref"].lower()]
    if exact:
        raise RuntimeError(
            "BLOCKED_IMMUTABLE_SOURCE_HYGIENE: exact Phase2 kernel candidate already exists"
        )
    prefix = (
        f"{contract['owner']}/p17-p2-{contract['model_id'].lower()}-"
        f"r{contract['resolution']}-"
    ).lower()
    related = sorted(
        {
            str(item.get("ref") or "")
            for item in kernels
            if str(item.get("ref") or "").lower().startswith(prefix)
        }
    )
    active: list[dict[str, Any]] = []
    for kernel_ref in related:
        payload = _status(broker, contract, kernel_ref)
        state = session_state(payload)
        if state in ACTIVE_STATES:
            active.append({"kernel_ref": kernel_ref, "state": state})
    if active:
        raise RuntimeError(
            "BLOCKED_ADMISSION_AUTHORITY_MISSING: active duplicate Phase2 unit "
            + json.dumps(active, sort_keys=True)
        )
    return {
        "schema": "pneumonia.phase2.unit.admission.v1",
        "status": "PASS",
        "token": token,
        "kernel_ref": contract["kernel_ref"],
        "account_id": contract["account_id"],
        "owner": contract["owner"],
        "related_terminal_candidates": related,
        "active_duplicates": [],
    }


def verify_terminal(
    token: str,
    run_id: str,
    kernel_ref: str,
    evidence_dir: pathlib.Path,
    max_polls: int,
    interval_seconds: int,
) -> dict[str, Any]:
    date_match = re.search(r"-a[0-9]{2}-(20[0-9]{6})-" + re.escape(str(run_id)) + r"$", kernel_ref)
    if date_match is None:
        raise RuntimeError("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: Phase2 kernel date/run drift")
    contract = unit_contract(token, run_id, date_match.group(1))
    if kernel_ref != contract["kernel_ref"]:
        raise RuntimeError("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: Phase2 kernel ref drift")
    broker = OidcReadBroker()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    terminal_payload: dict[str, Any] | None = None
    terminal_state = "UNKNOWN"
    started = time.time()
    for poll in range(max_polls):
        payload = _status(broker, contract, kernel_ref)
        state = session_state(payload)
        observation = {
            "poll": poll,
            "elapsed_seconds": round(time.time() - started, 1),
            "state": state,
        }
        history.append(observation)
        print("PHASE2_UNIT_STATUS", json.dumps(observation, sort_keys=True), flush=True)
        if state in TERMINAL_STATES:
            terminal_payload = payload
            terminal_state = state
            break
        if poll + 1 < max_polls:
            time.sleep(interval_seconds)
    terminal_envelope = {
        "schema": "pneumonia.phase2.unit.verification.v1",
        "token": token,
        "kernel_ref": kernel_ref,
        "account_id": contract["account_id"],
        "terminal_state": terminal_state,
        "session_status": terminal_payload,
        "poll_count": len(history),
        "elapsed_seconds": round(time.time() - started, 1),
        "scientific_receipt_validated": False,
        "github_run_id": run_id,
        "source_sha": os.environ.get("GITHUB_SHA", ""),
    }
    result_path = evidence_dir / "phase2-unit-verification.json"
    result_path.write_text(json.dumps(terminal_envelope, indent=2), encoding="utf-8")
    if terminal_state == "UNKNOWN":
        raise RuntimeError(
            "BLOCKED_VALIDATION_INFRASTRUCTURE: Phase2 unit did not reach terminal state"
        )
    if terminal_state != "COMPLETE":
        log_error = None
        try:
            live = broker.read(
                {
                    "action": "live_log",
                    "account_id": contract["account_id"],
                    "kernel_ref": kernel_ref,
                    "max_chars": 40000,
                },
                timeout=120,
            )
            log_tail = str(live.get("log_tail") or "")[-40000:]
            if log_tail:
                (evidence_dir / "phase2-error-log-tail.txt").write_text(
                    log_tail, encoding="utf-8"
                )
        except Exception as error:  # evidence capture must not hide the terminal state
            log_error = f"{type(error).__name__}: {error}"
        terminal_envelope["log_capture_error"] = log_error
        result_path.write_text(json.dumps(terminal_envelope, indent=2), encoding="utf-8")
        raise RuntimeError(f"PHASE2_UNIT_SCIENTIFIC_TERMINAL_{terminal_state}")

    file_name = "PHASE2_UNIT_TERMINAL_RECEIPT.json"
    exact_output = broker.read(
        {
            "action": "output_json_files",
            "account_id": contract["account_id"],
            "kernel_ref": kernel_ref,
            "file_names": [file_name],
            "max_bytes_per_file": 262144,
        },
        timeout=180,
    )
    if exact_output.get("account_id") != contract["account_id"] or exact_output.get("kernel_ref") != kernel_ref:
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 output identity drift")
    if exact_output.get("signed_urls_returned") is not False:
        raise RuntimeError("BLOCKED_VALIDATION_INFRASTRUCTURE: signed URLs escaped read broker")
    files = exact_output.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 receipt missing")
    item = files[0]
    if not isinstance(item, dict) or pathlib.PurePosixPath(
        str(item.get("source_file_name") or "")
    ).name != file_name:
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 receipt basename drift")
    receipt = item.get("json")
    if not isinstance(receipt, dict):
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 receipt JSON missing")
    validate_terminal_receipt(receipt, contract)
    successor = next_token(token, str(receipt["status"]))
    (evidence_dir / file_name).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    terminal_envelope.update(
        {
            "scientific_receipt_validated": True,
            "scientific_status": receipt["status"],
            "scientific_receipt": receipt,
            "next_token": successor,
            "status": "SCIENTIFIC_RECEIPT_PASS",
        }
    )
    result_path.write_text(json.dumps(terminal_envelope, indent=2), encoding="utf-8")
    print("PHASE2_UNIT_TERMINAL_SCIENTIFIC_PASS", json.dumps(terminal_envelope, sort_keys=True))
    return terminal_envelope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--token", required=True)
    preflight_parser.add_argument("--run-id", required=True)
    preflight_parser.add_argument("--output", type=pathlib.Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--token", required=True)
    verify_parser.add_argument("--run-id", required=True)
    verify_parser.add_argument("--kernel-ref", required=True)
    verify_parser.add_argument("--evidence-dir", required=True, type=pathlib.Path)
    verify_parser.add_argument("--max-polls", type=int, default=631)
    verify_parser.add_argument("--interval-seconds", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = admission_preflight(args.token, args.run_id)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = verify_terminal(
            args.token,
            args.run_id,
            args.kernel_ref,
            args.evidence_dir,
            args.max_polls,
            args.interval_seconds,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
