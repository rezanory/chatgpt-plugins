from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/selected-run-config"
ROOT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "v622-selected-run-config"
ROOT.mkdir(parents=True, exist_ok=True)


def fetch() -> tuple[dict, bytes]:
    token = os.environ["V622_SELECTED_RUN_TOKEN"]
    req = urllib.request.Request(
        BASE,
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "v622-selected-run-config-collector/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        value = json.loads(response.read().decode("utf-8"))
    if value.get("project") != "PNEUMONIA V6.2.2" or value.get("purpose") != "selected_run_recipe_source_recovery":
        raise RuntimeError(f"unexpected response: {value}")
    if value.get("worker_id") != "W16" or value.get("model_id") != "M06" or value.get("resolution") != 224:
        raise RuntimeError("selected-run identity mismatch")
    if value.get("exact_training_run_path_pinned") is not True:
        raise RuntimeError("selected-run path was not exact-pinned")
    content = value.get("content")
    if not isinstance(content, str):
        raise RuntimeError("selected-run config content missing")
    raw = content.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != value.get("sha256"):
        raise RuntimeError("selected-run config transport hash mismatch")
    return value, raw


def main() -> None:
    last = None
    for attempt in range(1, 21):
        try:
            meta, raw = fetch()
            (ROOT / "config.json").write_bytes(raw)
            receipt = {
                "project": "PNEUMONIA V6.2.2",
                "stage": "selected_run_recipe_source_recovery",
                "worker_id": "W16",
                "model_id": "M06",
                "resolution": 224,
                "kernel_ref": meta.get("kernel_ref"),
                "file_name": meta.get("file_name"),
                "config_sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "attempt": attempt,
                "locked_test_used": False,
                "external_cohort_used": False,
                "kaggle_compute_launched": False,
            }
            (ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(json.dumps(receipt))
            return
        except Exception as exc:
            last = exc
            print(f"selected-run config attempt {attempt}/20: {exc}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"selected-run config recovery failed: {last}")


if __name__ == "__main__":
    main()
