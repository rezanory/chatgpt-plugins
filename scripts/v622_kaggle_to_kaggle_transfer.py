from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/finalization-artifacts"
TOKEN = os.environ["V622_TRANSFER_TOKEN"]
RUN_ROOT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "v622-kaggle-to-kaggle-transfer"
CACHE_ROOT = pathlib.Path(os.environ.get("RUNNER_WORKSPACE", os.environ.get("RUNNER_TEMP", "."))) / ".v622-kaggle-transfer-cache"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
RECEIPT = RUN_ROOT / "transfer-receipt.json"
FAILURE = RUN_ROOT / "failure.txt"
CHUNK = 8 * 1024 * 1024
TARGETS = [
    "M06__convnext_tiny__config.json",
    "M06__convnext_tiny__final_selected.keras",
    "M06__densenet121__config.json",
    "M06__densenet121__final_selected.keras",
    "M06__resnet50v2__config.json",
    "M06__resnet50v2__final_selected.keras",
    "FROZEN_BACKBONE_ENSEMBLE_POLICY.json",
]


def post(endpoint: str, payload: dict | None = None, timeout: int = 180) -> dict:
    raw = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/{endpoint}", data=raw, method="POST", headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "User-Agent": "v622-kaggle-chunked-controller/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1600]
        raise RuntimeError(f"bridge {endpoint} HTTP {exc.code}: {detail}") from exc
    if value.get("ok") is False:
        raise RuntimeError(f"bridge {endpoint}: {value.get('error')}")
    return value


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def target_row(dest_name: str) -> dict:
    plan = post("plan")
    rows = [r for r in plan.get("targets", []) if r.get("dest_name") == dest_name]
    if len(rows) != 1:
        raise RuntimeError(f"{dest_name}: expected one plan row")
    return rows[0]


def download(dest_name: str, path: pathlib.Path) -> dict:
    last = ""
    for attempt in range(1, 6):
        row = target_row(dest_name)
        existing = path.stat().st_size if path.exists() else 0
        print(f"DOWNLOAD {dest_name} attempt={attempt} existing={existing}", flush=True)
        cmd = ["curl.exe", "--http1.1", "--fail", "--location", "--show-error", "--retry", "3", "--retry-all-errors", "--retry-delay", "2", "--connect-timeout", "20", "--speed-limit", "2048", "--speed-time", "120", "--max-time", "900", "--continue-at", "-", "--output", str(path), str(row["source_url"])]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode == 0 and path.exists() and path.stat().st_size > 0:
            digest = sha256(path)
            expected = row.get("expected_sha256")
            if expected and digest != expected:
                path.unlink(missing_ok=True)
                last = f"SHA mismatch expected={expected} actual={digest}"
            else:
                return {"dest_name": dest_name, "bytes": path.stat().st_size, "sha256": digest, "source_file_name": row.get("source_file_name"), "expected_sha256": expected}
        else:
            last = (p.stderr or f"curl exit {p.returncode}")[-1200:]
            if p.returncode == 33:
                path.unlink(missing_ok=True)
        time.sleep(min(attempt * 3, 12))
    raise RuntimeError(f"{dest_name}: download failed: {last}")


def put_chunk(url: str, data: bytes, start: int, end: int, total: int) -> tuple[int, int]:
    req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": "application/octet-stream", "Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(len(data))})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, end + 1
    except urllib.error.HTTPError as exc:
        if exc.code == 308:
            rng = exc.headers.get("Range", "")
            m = re.search(r"(\d+)-(\d+)$", rng)
            committed = int(m.group(2)) + 1 if m else start
            return 308, committed
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"upload HTTP {exc.code}: {detail}") from exc


def upload(dest_name: str, path: pathlib.Path) -> dict:
    total = path.stat().st_size
    started = post("start-upload", {"dest_name": dest_name, "bytes": total})
    url = str(started["create_url"])
    token = str(started["token"])
    pos = 0
    with path.open("rb") as f:
        while pos < total:
            f.seek(pos)
            data = f.read(min(CHUNK, total - pos))
            end = pos + len(data) - 1
            last = ""
            for attempt in range(1, 6):
                try:
                    status, next_pos = put_chunk(url, data, pos, end, total)
                    print(f"UPLOAD {dest_name} {pos}-{end}/{total} status={status} next={next_pos}", flush=True)
                    pos = next_pos
                    break
                except Exception as exc:
                    last = str(exc)
                    time.sleep(min(attempt * 2, 10))
            else:
                raise RuntimeError(f"{dest_name}: chunk upload failed at {pos}: {last}")
    return {"dest_name": dest_name, "token": token}


def main() -> None:
    initial = post("plan")
    if initial.get("destination_dataset") != "azadka/pneumonia-v6-2-2-finalization-artifacts" or initial.get("target_count") != 7:
        raise RuntimeError("transfer plan mismatch")
    tokens, artifacts, cache_paths = [], [], []
    for i, dest_name in enumerate(TARGETS, 1):
        path = CACHE_ROOT / dest_name
        cache_paths.append(path)
        print(f"TRANSFER {i}/7 {dest_name}", flush=True)
        evidence = download(dest_name, path)
        uploaded = upload(dest_name, path)
        tokens.append(uploaded)
        artifacts.append(evidence)
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
            print(f"DATASET readiness transient: {exc}", flush=True)
        time.sleep(10)
    else:
        raise RuntimeError("azadka dataset did not become readable")
    receipt = {"project":"PNEUMONIA V6.2.2","stage":"KAGGLE_TO_KAGGLE_ARTIFACT_TRANSFER","status":"PASS","source_kernel":initial.get("source_kernel"),"destination_dataset":initial.get("destination_dataset"),"dataset_version":status.get("current_version_number"),"artifact_count":7,"artifacts":artifacts,"transport_mode":"RESUMABLE_DOWNLOAD_PLUS_CHUNKED_KAGGLE_UPLOAD","github_artifact_storage_used":False,"model_compute_performed_by_bridge":False,"training_hpo_confirmation":False,"locked_test_used":False,"external_validation_used":False,"next_stage":"FINAL_FREEZE_MANIFEST"}
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    for p in cache_paths:
        p.unlink(missing_ok=True)
    print(json.dumps({"status":"PASS","dataset_version":receipt["dataset_version"],"artifact_count":7}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        FAILURE.write_text(str(exc)[:2400]+"\n", encoding="utf-8")
        raise
