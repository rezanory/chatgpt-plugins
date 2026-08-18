from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/v6-2-2/governance-source"
ROOT = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "v622-governance-source-bundle"
ROOT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "select_champion.py",
    "freeze_selected_recipe.py",
    "run_backbone_comparison.py",
    "backbone_comparison_contracts.py",
    "governance_contracts.py",
]
OPTIONAL = [
    "final_experiment_contracts.py",
    "inference_preprocessing_contract.py",
    "source_package_integrity.py",
    "v6_fingerprint_contracts.py",
    "run_final_canonical_workflow.py",
    "run_qualification_model_selection.py",
]
EXPECTED_SELECT_SHA = "f6be9ddf5060dbea5fab233ee0d349820ef679c64662eec23dd01c2929aec9b2"


def request(name: str) -> tuple[dict, bytes]:
    token = os.environ["V622_GOVERNANCE_TOKEN"]
    payload = json.dumps({"source_name": name}).encode("utf-8")
    req = urllib.request.Request(
        BASE,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "v622-governance-source-collector/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        value = json.loads(response.read().decode("utf-8"))
    if value.get("project") != "PNEUMONIA V6.2.2" or value.get("purpose") != "canonical_governance_source_recovery":
        raise RuntimeError(f"unexpected response for {name}: {value}")
    if value.get("canonical_source_directory_pinned") is not True:
        raise RuntimeError(f"source directory not pinned for {name}")
    content = value.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"missing content for {name}")
    raw = content.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != value.get("sha256"):
        raise RuntimeError(f"transport hash mismatch for {name}")
    if name == "select_champion.py" and digest != EXPECTED_SELECT_SHA:
        raise RuntimeError(f"canonical select_champion hash mismatch: {digest}")
    return value, raw


def recover(name: str, required: bool) -> dict:
    last = None
    for attempt in range(1, 21):
        try:
            meta, raw = request(name)
            path = ROOT / name
            path.write_bytes(raw)
            return {
                "name": name,
                "status": "RECOVERED",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "file_name": meta.get("file_name"),
                "attempt": attempt,
            }
        except Exception as exc:
            last = exc
            print(f"{name} attempt {attempt}/20: {exc}", flush=True)
            time.sleep(3)
    if required:
        raise RuntimeError(f"required governance source recovery failed for {name}: {last}")
    return {"name": name, "status": "NOT_RECOVERED_OPTIONAL", "error": str(last)[:500]}


def main() -> None:
    rows = [recover(name, True) for name in REQUIRED]
    rows.extend(recover(name, False) for name in OPTIONAL)
    manifest = {
        "project": "PNEUMONIA V6.2.2",
        "stage": "canonical_governance_source_recovery",
        "locked_test_used": False,
        "external_cohort_used": False,
        "kaggle_compute_launched": False,
        "sources": rows,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"governance_source_bundle_ready": True, "recovered": sum(row["status"] == "RECOVERED" for row in rows)}))


if __name__ == "__main__":
    main()
