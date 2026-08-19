from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/finalization-artifacts"
TOKEN = os.environ["V622_TRANSFER_TOKEN"]
RUN_ROOT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "v622-kaggle-to-kaggle-transfer"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = pathlib.Path(os.environ.get("RUNNER_WORKSPACE", os.environ.get("RUNNER_TEMP", "."))) / ".v622-kaggle-transfer-cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
RECEIPT = RUN_ROOT / "transfer-receipt.json"
FAILURE = RUN_ROOT / "failure.txt"


def post(endpoint: str, payload: dict | None = None, timeout: int = 180) -> dict:
    raw = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/{endpoint}",
        data=raw,
        method="POST",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "v622-kaggle-to-kaggle-transport/1.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"bridge {endpoint} HTTP {exc.code}: {detail}") from exc
    if value.get("ok") is False:
        raise RuntimeError(f"bridge {endpoint}: {value.get('error')}")
    return value


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fresh_target(dest_name: str) -> dict:
    value = post("plan")
    if value.get("project") != "PNEUMONIA V6.2.2" or value.get("target_count") != 7:
        raise RuntimeError("transfer plan identity/cardinality mismatch")
    rows = [row for row in value.get("targets", []) if row.get("dest_name") == dest_name]
    if len(rows) != 1:
        raise RuntimeError(f"{dest_name}: expected one transfer-plan row, found {len(rows)}")
    return rows[0]


def download(dest_name: str, path: pathlib.Path) -> dict:
    last = ""
    for attempt in range(1, 6):
        row = fresh_target(dest_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "curl.exe",
            "--http1.1",
            "--fail",
            "--location",
            "--retry", "6",
            "--retry-all-errors",
            "--retry-delay", "3",
            "--connect-timeout", "30",
            "--max-time", "2400",
            "--continue-at", "-",
            "--output", str(path),
            str(row["source_url"]),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and path.is_file() and path.stat().st_size > 0:
            digest = sha256(path)
            expected = row.get("expected_sha256")
            if expected and digest != expected:
                last = f"SHA mismatch expected={expected} actual={digest}"
                path.unlink(missing_ok=True)
            else:
                return {
                    "dest_name": dest_name,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "source_file_name": row.get("source_file_name"),
                    "expected_sha256": expected,
                    "download_attempt": attempt,
                    "resumable_cache": str(path),
                }
        else:
            last = (proc.stderr or f"curl exit {proc.returncode}")[-1200:]
            # If the origin refuses a range request, restart this file once from zero.
            if "range" in last.lower() and path.exists():
                path.unlink(missing_ok=True)
        time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"{dest_name}: source download failed after retries: {last}")


def upload(dest_name: str, path: pathlib.Path) -> dict:
    size = path.stat().st_size
    last = ""
    for attempt in range(1, 6):
        start = post("start-upload", {"dest_name": dest_name, "bytes": size})
        session_url = str(start["create_url"])
        upload_token = str(start["token"])
        cmd = [
            "curl.exe",
            "--http1.1",
            "--show-error",
            "--connect-timeout", "30",
            "--max-time", "3600",
            "--request", "PUT",
            "--header", "Content-Type: application/octet-stream",
            "--header", f"Content-Range: bytes 0-{size - 1}/{size}",
            "--upload-file", str(path),
            "--output", "NUL",
            "--write-out", "%{http_code}",
            session_url,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        code = (proc.stdout or "").strip()[-3:]
        if proc.returncode == 0 and code in {"200", "201"}:
            return {"dest_name": dest_name, "token": upload_token, "upload_attempt": attempt}
        last = f"curl={proc.returncode} http={code} stderr={(proc.stderr or '')[-1000:]}"
        time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"{dest_name}: Kaggle blob upload failed after retries: {last}")


def main() -> None:
    initial = post("plan")
    targets = initial.get("targets", [])
    if initial.get("destination_dataset") != "azadka/pneumonia-v6-2-2-finalization-artifacts" or len(targets) != 7:
        raise RuntimeError("unexpected transfer destination/cardinality")

    tokens: list[dict] = []
    artifacts: list[dict] = []
    cached_paths: list[pathlib.Path] = []
    for index, row in enumerate(targets, start=1):
        dest_name = str(row["dest_name"])
        local = CACHE_ROOT / dest_name
        cached_paths.append(local)
        existing = local.stat().st_size if local.exists() else 0
        print(f"TRANSFER {index}/7 {dest_name}: source download/resume existing_bytes={existing}", flush=True)
        evidence = download(dest_name, local)
        print(f"TRANSFER {index}/7 {dest_name}: sha256={evidence['sha256']} bytes={evidence['bytes']}", flush=True)
        uploaded = upload(dest_name, local)
        tokens.append({"dest_name": dest_name, "token": uploaded["token"]})
        evidence["upload_attempt"] = uploaded["upload_attempt"]
        artifacts.append(evidence)
        print(f"TRANSFER {index}/7 {dest_name}: uploaded to Kaggle azadka blob; cache retained until Dataset PASS", flush=True)

    finalized = post("finalize", {"tokens": tokens})
    print(f"DATASET finalize action={finalized.get('action')}", flush=True)

    status = None
    for attempt in range(1, 61):
        try:
            status = post("status")
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
        "github_artifact_storage_used": False,
        "model_compute_performed_by_bridge": False,
        "training_hpo_confirmation": False,
        "locked_test_used": False,
        "external_validation_used": False,
        "next_stage": "FINAL_FREEZE_MANIFEST",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "dataset": receipt["destination_dataset"], "dataset_version": receipt["dataset_version"], "artifact_count": 7}), flush=True)

    # Only clear persistent local cache after Kaggle Dataset readiness is proven.
    for path in cached_paths:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        FAILURE.write_text(str(exc)[:2000] + "\n", encoding="utf-8")
        raise
