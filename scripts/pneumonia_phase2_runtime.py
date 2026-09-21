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
    FROZEN_M07_RECIPE_FINGERPRINT,
    SPLIT_FINGERPRINT,
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


class BrokerTransientError(RuntimeError):
    """A bounded transport failure whose operation outcome remains unknown."""


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
        target = (
            request_url
            + separator
            + "audience="
            + urllib.parse.quote("cgp-control-plane-v3", safe="")
        )
        last_error: Exception | None = None
        payload: dict[str, Any] | None = None
        for attempt in range(1, 5):
            request = urllib.request.Request(
                target,
                headers={
                    "Authorization": "Bearer " + request_token,
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8", "replace") or "{}")
                break
            except urllib.error.HTTPError as error:
                if error.code not in {408, 429} and not 500 <= error.code <= 599:
                    raise
                last_error = error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
            if attempt < 4:
                time.sleep(min(2 ** (attempt - 1), 4))
        if payload is None:
            raise BrokerTransientError(
                "Phase2 OIDC refresh transient retry exhausted"
            ) from last_error
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
                        return json.loads(response.read().decode("utf-8", "replace") or "{}")
                except urllib.error.HTTPError as error:
                    if error.code not in {408, 429} and not 500 <= error.code <= 599:
                        raise
                    last_error = error
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    last_error = error
                if attempt == 4:
                    break
                time.sleep(min(2 ** (attempt - 1), 4))
            raise BrokerTransientError(
                "Phase2 read broker transient retry exhausted"
            ) from last_error

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
    try:
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
    except RuntimeError as error:
        # Kaggle currently returns HTTP 403 for GetKernelSessionStatus on some
        # freshly submitted private kernels even though the same account can
        # authenticate, list the exact kernel, and later read its outputs.  Do
        # not turn that provider-specific status failure into a compute replay.
        # Reconcile the immutable ref through ListKernels, then use a bounded
        # exact receipt probe as the terminal signal.
        if "Kaggle API HTTP 403" not in str(error):
            raise
        return _status_from_exact_listing(broker, contract, kernel_ref, owner, slug)


def _listed_kernel_matches(item: dict[str, Any], owner: str, slug: str) -> bool:
    expected = f"{owner}/{slug}".lower()
    if str(item.get("ref") or "").lower() == expected:
        return True
    item_owner = str(
        item.get("ownerRef")
        or item.get("ownerUser")
        or item.get("owner")
        or item.get("userName")
        or ""
    ).lower()
    item_slug = str(item.get("kernelSlug") or item.get("slug") or "").lower()
    return item_owner == owner.lower() and item_slug == slug.lower()


def _listed_kernel_state(item: dict[str, Any]) -> str:
    for key in (
        "status",
        "state",
        "runStatus",
        "run_status",
        "currentState",
        "current_state",
        "kernelSessionStatus",
        "kernel_session_status",
    ):
        if key in item:
            state = session_state({key: item[key]})
            if state != "UNKNOWN":
                return state
    if item.get("isRunning") is True or item.get("is_running") is True:
        return "RUNNING"
    return "UNKNOWN"


def expected_provider_kernel_ref(contract: dict[str, Any]) -> str:
    """Return the exact Kaggle ref produced from the submitted display title.

    Kaggle's SaveKernel response can canonicalize the requested ``slug`` from
    ``newTitle``.  Keep the scientific contract ref immutable, but bind the
    transport to this one deterministic provider alias when it is returned by
    the accepted launch receipt.
    """
    return (
        f"{contract['owner']}/pneumonia-v1-7-phase2-"
        f"{str(contract['model_id']).lower()}-r{int(contract['resolution'])}-"
        f"a{int(contract['attempt']):02d}-{contract['run_id']}"
    )


def _state_marker_folds(marker: dict[str, Any], contract: dict[str, Any]) -> list[int]:
    if marker.get("schema") != "phase2.state.v2" or marker.get("status") != "COMPLETE":
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state marker status drift"
        )
    if marker.get("model_id") != contract["model_id"] or int(marker.get("resolution", -1)) != int(
        contract["resolution"]
    ):
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state marker identity drift"
        )
    run_contract = marker.get("run_contract")
    if not isinstance(run_contract, dict):
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state contract missing")
    if (
        run_contract.get("schema") != "pneumonia.experiment.v1.7"
        or run_contract.get("stage") != "phase2_campaign"
        or run_contract.get("model_id") != contract["model_id"]
        or int(run_contract.get("resolution", -1)) != int(contract["resolution"])
        or run_contract.get("split_fingerprint") != SPLIT_FINGERPRINT
        or not isinstance(run_contract.get("extra"), dict)
        or run_contract["extra"].get("recipe_fingerprint") != FROZEN_M07_RECIPE_FINGERPRINT
    ):
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state contract drift")
    # The read gateway returns parsed JSON, and its JSON round-trip can change
    # the byte representation of floating-point values.  Therefore the raw
    # notebook seal cannot be recomputed from this transport object.  Require
    # both embedded seals to remain exact SHA-256 values here; the generated
    # notebook revalidates them from the downloaded raw marker before restore.
    if re.fullmatch(r"[0-9a-f]{64}", str(marker.get("run_fingerprint") or "")) is None:
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state contract seal invalid"
        )
    if re.fullmatch(r"[0-9a-f]{64}", str(marker.get("receipt_sha256") or "")) is None:
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state marker seal invalid"
        )
    artifacts = marker.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state artifacts missing"
        )
    folds: list[int] = []
    for name, digest in artifacts.items():
        match = re.fullmatch(r"FOLD_([1-5])_RECOVERY\.(?:zip|cgpzip)", str(name))
        if match is None:
            if str(name) != "FINAL_EVIDENCE.cgpzip":
                raise RuntimeError(
                    "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state artifact name invalid"
                )
        else:
            folds.append(int(match.group(1)))
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise RuntimeError(
                "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state artifact hash invalid"
            )
    folds = sorted(set(folds))
    if folds != list(range(1, len(folds) + 1)):
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state fold prefix invalid"
        )
    return folds


def _state_snapshot(broker: OidcReadBroker, contract: dict[str, Any]) -> dict[str, Any]:
    listed = broker.read(
        {
            "action": "raw_read",
            "account_id": contract["account_id"],
            "service": "datasets.DatasetApiService",
            "method": "ListDatasets",
            "body": {
                "group": "MY",
                "search": contract["state_dataset_handle"].split("/", 1)[1],
                "page": 1,
                "pageSize": 100,
            },
        },
        timeout=60,
    )
    datasets = [item for item in listed.get("datasets", []) if isinstance(item, dict)]
    exact = [
        item
        for item in datasets
        if str(item.get("ref") or "").lower() == contract["state_dataset_handle"].lower()
    ]
    if not exact:
        return {
            "dataset_ref": contract["state_dataset_handle"],
            "dataset_version_number": None,
            "completed_folds": [],
            "topology": "ABSENT",
        }
    if len(exact) != 1:
        raise RuntimeError(
            "BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: exact Phase2 state dataset count invalid"
        )
    version = exact[0].get("currentVersionNumber", exact[0].get("current_version_number"))
    if type(version) is not int or version < 1:
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state version invalid")
    marker_result = broker.read(
        {
            "action": "dataset_json_files",
            "account_id": contract["account_id"],
            "dataset_ref": contract["state_dataset_handle"],
            "dataset_version_number": version,
            "file_names": ["CAMPAIGN_STATE.json"],
            "max_bytes_per_file": 262144,
        },
        timeout=120,
    )
    if (
        marker_result.get("dataset_ref") != contract["state_dataset_handle"]
        or marker_result.get("dataset_version_number") != version
        or marker_result.get("signed_urls_returned") is not False
    ):
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state marker transport drift"
        )
    files = marker_result.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state marker missing")
    marker = files[0].get("json")
    if not isinstance(marker, dict):
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state marker JSON missing"
        )
    marker_file_sha256 = str(files[0].get("sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", marker_file_sha256) is None:
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state marker file seal missing"
        )
    folds = _state_marker_folds(marker, contract)
    listing = broker.read(
        {
            "action": "raw_read",
            "account_id": contract["account_id"],
            "service": "datasets.DatasetApiService",
            "method": "ListDatasetFiles",
            "body": {
                "ownerSlug": contract["owner"],
                "datasetSlug": contract["state_dataset_handle"].split("/", 1)[1],
                "datasetVersionNumber": version,
                "pageSize": 100,
            },
        },
        timeout=60,
    )
    raw_files = listing.get("datasetFiles", listing.get("files"))
    if not isinstance(raw_files, list):
        raise RuntimeError("BLOCKED_VALIDATION_INFRASTRUCTURE: Phase2 state file inventory missing")
    names = {
        str(item.get("name") or item.get("ref") or item.get("fileName") or "")
        for item in raw_files
        if isinstance(item, dict)
    }
    artifacts = set(marker["artifact_sha256"])
    if artifacts.issubset(names):
        topology = (
            "SEALED_CGPZIP"
            if all(name.endswith(".cgpzip") for name in artifacts)
            else "SEALED_LEGACY_ZIP"
        )
    elif all(
        name.endswith(".zip")
        and any(listed_name.startswith(name[:-4] + "/") for listed_name in names)
        for name in artifacts
    ):
        topology = "LEGACY_KAGGLE_EXPANDED_ZIP"
    else:
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 state file topology invalid"
        )
    return {
        "dataset_ref": contract["state_dataset_handle"],
        "dataset_version_number": version,
        "completed_folds": folds,
        "topology": topology,
        "marker_file_sha256": marker_file_sha256,
    }


def _wait_for_state_ready(
    broker: OidcReadBroker,
    contract: dict[str, Any],
    completed_folds: list[int],
    max_polls: int = 60,
    interval_seconds: int = 15,
) -> dict[str, Any]:
    last_error = ""
    last_snapshot: dict[str, Any] | None = None
    for poll in range(max_polls):
        try:
            last_snapshot = _state_snapshot(broker, contract)
            if (
                last_snapshot.get("completed_folds") == completed_folds
                and last_snapshot.get("topology") == "SEALED_CGPZIP"
            ):
                return {**last_snapshot, "readiness_poll_count": poll + 1}
            last_error = "state version has not exposed the exact sealed fold lineage"
        except (BrokerTransientError, RuntimeError) as error:
            last_error = str(error)
        if poll + 1 < max_polls:
            time.sleep(interval_seconds)
    raise RuntimeError(
        "BLOCKED_VALIDATION_INFRASTRUCTURE: Phase2 persisted state not ready for successor; "
        + last_error
        + ("; last_snapshot=" + json.dumps(last_snapshot, sort_keys=True) if last_snapshot else "")
    )


def _status_from_exact_listing(
    broker: OidcReadBroker,
    contract: dict[str, Any],
    kernel_ref: str,
    owner: str,
    slug: str,
) -> dict[str, Any]:
    listed = broker.read(
        {
            "action": "raw_read",
            "account_id": contract["account_id"],
            "service": "kernels.KernelsApiService",
            "method": "ListKernels",
            "body": {
                "group": "PROFILE",
                "user": owner,
                "search": slug,
                "sortBy": "DATE_RUN",
                "page": 1,
                "pageSize": 100,
            },
        },
        timeout=60,
    )
    kernels = [item for item in listed.get("kernels", []) if isinstance(item, dict)]
    exact = [item for item in kernels if _listed_kernel_matches(item, owner, slug)]
    if len(exact) != 1:
        raise RuntimeError(
            "BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: Phase2 status fallback "
            f"expected one exact listed kernel, observed {len(exact)}"
        )
    listed_state = _listed_kernel_state(exact[0])
    if listed_state != "UNKNOWN":
        return {
            "state": listed_state,
            "kernel_ref": kernel_ref,
            "status_transport": "LIST_KERNELS_EXACT_FALLBACK",
        }

    file_name = "PHASE2_UNIT_TERMINAL_RECEIPT.json"
    try:
        output = broker.read(
            {
                "action": "output_json_files",
                "account_id": contract["account_id"],
                "kernel_ref": kernel_ref,
                "file_names": [file_name],
                "max_bytes_per_file": 262144,
            },
            timeout=180,
        )
    except RuntimeError as output_error:
        if "Kaggle API HTTP 403" not in str(output_error):
            raise
        return {
            "state": "RUNNING",
            "kernel_ref": kernel_ref,
            "status_transport": "LIST_KERNELS_EXACT_OUTPUT_PENDING",
        }
    files = output.get("files")
    receipt_present = isinstance(files, list) and any(
        isinstance(item, dict)
        and pathlib.PurePosixPath(str(item.get("source_file_name") or "")).name == file_name
        and isinstance(item.get("json"), dict)
        for item in files
    )
    return {
        "state": "COMPLETE" if receipt_present else "RUNNING",
        "kernel_ref": kernel_ref,
        "status_transport": (
            "EXACT_TERMINAL_RECEIPT_FALLBACK"
            if receipt_present
            else "LIST_KERNELS_EXACT_OUTPUT_NOT_TERMINAL"
        ),
    }


def admission_preflight(token: str, run_id: str) -> dict[str, Any]:
    contract = unit_contract(token, run_id)
    broker = OidcReadBroker()
    state_before = _state_snapshot(broker, contract)
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
    immutable_refs = {
        contract["kernel_ref"].lower(),
        expected_provider_kernel_ref(contract).lower(),
    }
    exact = [item for item in kernels if str(item.get("ref", "")).lower() in immutable_refs]
    if exact:
        raise RuntimeError(
            "BLOCKED_IMMUTABLE_SOURCE_HYGIENE: exact Phase2 kernel candidate already exists"
        )
    prefixes = (
        (
            f"{contract['owner']}/p17-p2-{contract['model_id'].lower()}-r{contract['resolution']}-"
        ).lower(),
        (
            f"{contract['owner']}/pneumonia-v1-7-phase2-"
            f"{contract['model_id'].lower()}-r{contract['resolution']}-"
        ).lower(),
    )
    related = sorted(
        {
            str(item.get("ref") or "")
            for item in kernels
            if str(item.get("ref") or "").lower().startswith(prefixes)
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
        "state_before": state_before,
        "expected_restored_folds": state_before["completed_folds"],
    }


def verify_terminal(
    token: str,
    run_id: str,
    kernel_ref: str,
    provider_kernel_ref: str | None,
    evidence_dir: pathlib.Path,
    max_polls: int,
    interval_seconds: int,
    expected_restored_folds: list[int] | None = None,
) -> dict[str, Any]:
    date_match = re.search(r"-a[0-9]{2}-(20[0-9]{6})-" + re.escape(str(run_id)) + r"$", kernel_ref)
    if date_match is None:
        raise RuntimeError("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: Phase2 kernel date/run drift")
    contract = unit_contract(
        token,
        run_id,
        date_match.group(1),
        expected_restored_folds,
    )
    if kernel_ref != contract["kernel_ref"]:
        raise RuntimeError("BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: Phase2 kernel ref drift")
    provider_kernel_ref = provider_kernel_ref or kernel_ref
    allowed_provider_refs = {kernel_ref, expected_provider_kernel_ref(contract)}
    if provider_kernel_ref not in allowed_provider_refs:
        raise RuntimeError(
            "BLOCKED_EXACT_OBJECT_IDENTITY_MISMATCH: Phase2 provider kernel ref drift"
        )
    broker = OidcReadBroker()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    terminal_payload: dict[str, Any] | None = None
    terminal_state = "UNKNOWN"
    started = time.time()
    for poll in range(max_polls):
        transport_error = None
        try:
            payload = _status(broker, contract, provider_kernel_ref)
            state = session_state(payload)
        except BrokerTransientError as error:
            payload = None
            state = "UNKNOWN"
            transport_error = str(error)
        observation = {
            "poll": poll,
            "elapsed_seconds": round(time.time() - started, 1),
            "state": state,
        }
        if transport_error is not None:
            observation["transport_error"] = transport_error
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
        "provider_kernel_ref": provider_kernel_ref,
        "account_id": contract["account_id"],
        "terminal_state": terminal_state,
        "session_status": terminal_payload,
        "poll_count": len(history),
        "history": history,
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
                    "kernel_ref": provider_kernel_ref,
                    "max_chars": 40000,
                },
                timeout=120,
            )
            log_tail = str(live.get("log_tail") or "")[-40000:]
            if log_tail:
                (evidence_dir / "phase2-error-log-tail.txt").write_text(log_tail, encoding="utf-8")
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
            "kernel_ref": provider_kernel_ref,
            "file_names": [file_name],
            "max_bytes_per_file": 262144,
        },
        timeout=180,
    )
    if (
        exact_output.get("account_id") != contract["account_id"]
        or exact_output.get("kernel_ref") != provider_kernel_ref
    ):
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 output identity drift")
    if exact_output.get("signed_urls_returned") is not False:
        raise RuntimeError("BLOCKED_VALIDATION_INFRASTRUCTURE: signed URLs escaped read broker")
    files = exact_output.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 receipt missing")
    item = files[0]
    if (
        not isinstance(item, dict)
        or pathlib.PurePosixPath(str(item.get("source_file_name") or "")).name != file_name
    ):
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 receipt basename drift")
    receipt = item.get("json")
    if not isinstance(receipt, dict):
        raise RuntimeError("BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: Phase2 receipt JSON missing")
    validate_terminal_receipt(receipt, contract)
    state_ready = _wait_for_state_ready(
        broker,
        contract,
        list(receipt["completed_folds"]),
    )
    successor = next_token(token, str(receipt["status"]))
    (evidence_dir / file_name).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    terminal_envelope.update(
        {
            "scientific_receipt_validated": True,
            "scientific_status": receipt["status"],
            "scientific_receipt": receipt,
            "persisted_state_ready": state_ready,
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
    verify_parser.add_argument("--provider-kernel-ref")
    verify_parser.add_argument("--evidence-dir", required=True, type=pathlib.Path)
    verify_parser.add_argument("--max-polls", type=int, default=631)
    verify_parser.add_argument("--interval-seconds", type=int, default=20)
    verify_parser.add_argument("--expected-restored-folds-json", default="[]")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = admission_preflight(args.token, args.run_id)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        try:
            expected_restored_folds = json.loads(args.expected_restored_folds_json)
        except json.JSONDecodeError as error:
            raise RuntimeError("PHASE2_EXPECTED_RESTORED_FOLDS_JSON_INVALID") from error
        if not isinstance(expected_restored_folds, list) or any(
            type(item) is not int for item in expected_restored_folds
        ):
            raise RuntimeError("PHASE2_EXPECTED_RESTORED_FOLDS_JSON_INVALID")
        result = verify_terminal(
            args.token,
            args.run_id,
            args.kernel_ref,
            args.provider_kernel_ref,
            args.evidence_dir,
            args.max_polls,
            args.interval_seconds,
            expected_restored_folds,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
