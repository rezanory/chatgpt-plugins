from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/selected-kernel-source"
EXPECTED_FP = "fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838"
ROOT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "v622-selected-kernel-source"
ROOT.mkdir(parents=True, exist_ok=True)


def fetch_value(token: str) -> dict:
    last: Exception | None = None
    for attempt in range(1, 31):
        req = urllib.request.Request(
            BASE,
            data=b"{}",
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "v622-selected-kernel-source-collector/1.1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                value = json.loads(response.read().decode("utf-8"))
            if value.get("project") != "PNEUMONIA V6.2.2" or value.get("purpose") != "selected_kernel_source_recovery":
                raise RuntimeError("stale or unexpected selected-kernel-source response")
            if attempt > 1:
                print(f"selected-kernel source bridge converged on attempt {attempt}/30", flush=True)
            return value
        except Exception as exc:
            last = exc
            print(f"selected-kernel source request attempt {attempt}/30 not ready: {exc}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"selected-kernel source bridge did not converge: {last}")


def main() -> None:
    token = os.environ["V622_SELECTED_KERNEL_SOURCE_TOKEN"]
    value = fetch_value(token)
    if value.get("kernel_ref") != "trickermark/pneumonia-v6-2-2-train-w16-m06-r224":
        raise RuntimeError("unexpected W16 kernel ref")
    blob = value.get("blob") or {}
    source = blob.get("source")
    if not isinstance(source, str) or EXPECTED_FP not in source:
        raise RuntimeError("canonical source fingerprint missing from W16 source")
    notebook = json.loads(source)
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list) or len(cells) != 1:
        raise RuntimeError(f"unexpected W16 notebook cell count: {0 if cells is None else len(cells)}")
    raw = source.encode("utf-8")
    (ROOT / "w16-kernel.ipynb").write_bytes(raw)
    safe = {
        "project": value.get("project"),
        "purpose": value.get("purpose"),
        "account_id": value.get("account_id"),
        "kernel_ref": value.get("kernel_ref"),
        "expected_source_fingerprint": value.get("expected_source_fingerprint"),
        "metadata": value.get("metadata"),
        "blob": {"language": blob.get("language"), "kernelType": blob.get("kernelType")},
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "cell_count": len(cells),
        "kaggle_compute_launched": False,
        "locked_test_used": False,
        "external_cohort_used": False,
    }
    (ROOT / "source_metadata.json").write_text(json.dumps(safe, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"source_ready": True, "source_sha256": safe["source_sha256"], "bytes": len(raw)}))


if __name__ == "__main__":
    main()
