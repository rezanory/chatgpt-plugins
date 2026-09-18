#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

from m07_runtime_contract import (
    extract_session_state,
    expected_runtime_contract,
    validate_runtime_receipt,
)

READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
OIDC_AUDIENCE = "cgp-control-plane-v3"


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
                    "User-Agent": "m07-existing-runtime-validator/1.0",
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


def download_json(url: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.kaggleusercontent.com":
        raise RuntimeError("Unsafe Kaggle output URL")
    request = urllib.request.Request(url, headers={"User-Agent": "m07-existing-runtime-validator/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read(262145)
    if len(raw) > 262144:
        raise RuntimeError("Runtime receipt exceeds bounded size")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError("Runtime receipt JSON object missing")
    return value


def find_output_listing(value: object) -> tuple[list[object], str] | None:
    if isinstance(value, dict):
        files = value.get("files")
        if isinstance(files, list):
            if not files or any(isinstance(item, dict) and item.get("fileName") for item in files):
                return files, str(value.get("nextPageToken") or "").strip()
        for child in value.values():
            found = find_output_listing(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_output_listing(child)
            if found is not None:
                return found
    return None


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
    report = validate_runtime_receipt(scientific, token=args.token)
    (out / args.expected_receipt).write_text(
        json.dumps(scientific, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evidence = {
        "schema": "m07.d260914d.runtime.existing-output-validation.v1",
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "kernel_ref": args.kernel_ref,
        "terminal_state": state,
        "runtime_token": args.token,
        "scientific_report": report,
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
