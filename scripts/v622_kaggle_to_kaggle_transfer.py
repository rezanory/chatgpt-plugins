from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/finalization-artifacts"
TOKEN = os.environ["V622_TRANSFER_TOKEN"]
ROOT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "v622-kaggle-to-kaggle-transfer"
ROOT.mkdir(parents=True, exist_ok=True)
RECEIPT = ROOT / "transfer-receipt.json"
FAILURE = ROOT / "failure.txt"
TARGETS = [
    "M06__convnext_tiny__config.json",
    "M06__convnext_tiny__final_selected.keras",
    "M06__densenet121__config.json",
    "M06__densenet121__final_selected.keras",
    "M06__resnet50v2__config.json",
    "M06__resnet50v2__final_selected.keras",
    "FROZEN_BACKBONE_ENSEMBLE_POLICY.json",
]


def post(endpoint: str, payload: dict | None = None, timeout: int = 1800) -> dict:
    raw = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/{endpoint}",
        data=raw,
        method="POST",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "v622-kaggle-direct-stream-controller/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1600]
        raise RuntimeError(f"bridge {endpoint} HTTP {exc.code}: {detail}") from exc
    if value.get("ok") is False:
        raise RuntimeError(f"bridge {endpoint}: {value.get('error')}")
    return value


def main() -> None:
    initial = post("plan", timeout=180)
    if initial.get("project") != "PNEUMONIA V6.2.2":
        raise RuntimeError("transfer plan identity mismatch")
    if initial.get("destination_dataset") != "azadka/pneumonia-v6-2-2-finalization-artifacts":
        raise RuntimeError("unexpected transfer destination")
    if initial.get("target_count") != 7:
        raise RuntimeError("unexpected transfer cardinality")

    tokens: list[dict] = []
    artifacts: list[dict] = []
    for index, dest_name in enumerate(TARGETS, start=1):
        print(f"DIRECT_STREAM {index}/7 start {dest_name}", flush=True)
        row = post("stream-one", {"dest_name": dest_name}, timeout=1800)
        if row.get("dest_name") != dest_name or row.get("transport_mode") != "DIRECT_STREAM":
            raise RuntimeError(f"{dest_name}: direct-stream response mismatch")
        token = str(row.get("token") or "")
        if len(token) < 10:
            raise RuntimeError(f"{dest_name}: upload token missing")
        tokens.append({"dest_name": dest_name, "token": token})
        artifacts.append({
            "dest_name": dest_name,
            "bytes": row.get("bytes"),
            "source_file_name": row.get("source_file_name"),
            "expected_sha256": row.get("expected_sha256"),
            "transport_mode": "DIRECT_STREAM",
            "upload_http_status": row.get("upload_http_status"),
        })
        print(f"DIRECT_STREAM {index}/7 complete {dest_name} bytes={row.get('bytes')}", flush=True)

    finalized = post("finalize", {"tokens": tokens}, timeout=300)
    print(f"DATASET finalize action={finalized.get('action')}", flush=True)

    status = None
    for attempt in range(1, 61):
        try:
            status = post("status", timeout=120)
            version = int(status.get("current_version_number") or 0)
            print(f"DATASET readiness {attempt}/60 version={version} ready={status.get('ready')}", flush=True)
            if status.get("ready") is True and version > 0:
                break
        except Exception as exc:
            print(f"DATASET readiness {attempt}/60 transient: {exc}", flush=True)
        time.sleep(10)
    else:
        raise RuntimeError("azadka finalization artifact dataset did not become readable in time")

    receipt = {
        "project": "PNEUMONIA V6.2.2",
        "stage": "KAGGLE_TO_KAGGLE_ARTIFACT_TRANSFER",
        "status": "PASS",
        "source_kernel": initial.get("source_kernel"),
        "destination_dataset": initial.get("destination_dataset"),
        "dataset_version": status.get("current_version_number") if status else None,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "transport_mode": "KAGGLE_TO_KAGGLE_DIRECT_STREAM",
        "github_artifact_storage_used": False,
        "github_runner_checkpoint_storage_used": False,
        "model_compute_performed_by_bridge": False,
        "training_hpo_confirmation": False,
        "locked_test_used": False,
        "external_validation_used": False,
        "next_stage": "FINAL_FREEZE_MANIFEST",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "dataset": receipt["destination_dataset"], "dataset_version": receipt["dataset_version"], "artifact_count": 7, "transport_mode": receipt["transport_mode"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        FAILURE.write_text(str(exc)[:2400] + "\n", encoding="utf-8")
        raise
