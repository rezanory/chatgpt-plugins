from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import time
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/selection-artifact"
EXPECTED_SELECT_SHA = "f6be9ddf5060dbea5fab233ee0d349820ef679c64662eec23dd01c2929aec9b2"
ROOT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "v622-validation-bundle"
EXP = ROOT / "experiments"
SRC = ROOT / "governance"
ROOT.mkdir(parents=True, exist_ok=True)
EXP.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

GOVERNANCE = [
    "select_champion.py",
    "auto_ensemble_selection.py",
    "backbone_ensemble_selection.py",
    "balanced_metrics.py",
    "backbone_comparison_contracts.py",
    "source_package_integrity.py",
    "v6_fingerprint_contracts.py",
]
VALIDATION = ["final_report.json", "validation_metrics.json", "val_predictions.csv"]


def request(worker_id: str, artifact_name: str, timeout: int = 120) -> tuple[dict, bytes]:
    token = os.environ["V622_SELECTION_TOKEN"]
    payload = json.dumps({"worker_id": worker_id, "artifact_name": artifact_name}).encode("utf-8")
    req = urllib.request.Request(
        BASE,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "v622-validation-bundle-collector/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if value.get("project") != "PNEUMONIA V6.2.2" or value.get("purpose") != "validation_only_selection_artifact":
        raise RuntimeError(f"unexpected response for {worker_id}/{artifact_name}")
    content = value.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"missing content for {worker_id}/{artifact_name}")
    raw = content.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != value.get("sha256"):
        raise RuntimeError(f"transport hash mismatch for {worker_id}/{artifact_name}")
    return value, raw


def request_governance(name: str) -> tuple[dict, bytes]:
    last = None
    for attempt in range(1, 31):
        try:
            value, raw = request("W01", name, timeout=60)
            if value.get("governance_source_package_pinned") is not True:
                raise RuntimeError("stale bridge edge")
            if name == "select_champion.py" and value.get("sha256") != EXPECTED_SELECT_SHA:
                raise RuntimeError(f"noncanonical select_champion sha={value.get('sha256')}")
            return value, raw
        except Exception as exc:
            last = exc
            print(f"governance {name} attempt {attempt}/30: {exc}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"could not recover canonical governance {name}: {last}")


def main() -> None:
    manifest: dict = {
        "project": "PNEUMONIA V6.2.2",
        "stage": "validation_bundle_collection",
        "locked_test_used": False,
        "external_cohort_used": False,
        "hpo_repeated": False,
        "confirmation_repeated": False,
        "workers": [],
        "governance": [],
    }

    for name in GOVERNANCE:
        meta, raw = request_governance(name)
        path = SRC / name
        path.write_bytes(raw)
        manifest["governance"].append(
            {
                "name": name,
                "file_name": meta.get("file_name"),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )

    schema = None
    rows_per_run = None
    for number in range(1, 37):
        worker_id = f"W{number:02d}"
        run_dir = EXP / worker_id
        run_dir.mkdir(parents=True, exist_ok=True)
        row = {"worker_id": worker_id, "artifacts": {}}
        for name in VALIDATION:
            meta, raw = request(worker_id, name)
            (run_dir / name).write_bytes(raw)
            row["artifacts"][name] = {
                "file_name": meta.get("file_name"),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
            if name == "val_predictions.csv":
                text = raw.decode("utf-8")
                reader = csv.reader(text.splitlines())
                header = next(reader)
                count = sum(1 for _ in reader)
                if schema is None:
                    schema = header
                    rows_per_run = count
                elif header != schema or count != rows_per_run:
                    raise RuntimeError(f"validation prediction shape mismatch at {worker_id}")
        manifest["workers"].append(row)

    manifest["prediction_schema"] = schema
    manifest["prediction_rows_per_run"] = rows_per_run
    manifest["canonical_runs"] = len(manifest["workers"])
    if manifest["canonical_runs"] != 36:
        raise RuntimeError("bundle must contain exactly 36 canonical runs")

    manifest_path = ROOT / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps({"bundle_ready": True, "runs": 36, "path": str(ROOT)}))


if __name__ == "__main__":
    main()
