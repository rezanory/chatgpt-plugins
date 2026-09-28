from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import pathlib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
RUNNER = ROOT / "soilbin_v6_1_runner.py"
MODEL_B64 = ROOT / "all_features_51.csv.gz.b64"
ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
KERNEL_REF = "mylovevpn1/soilbin-q1-q2-v6-1-virtual-synthetic-20260928"
TITLE = "SoilBin Q1 Q2 V6.1 Virtual Synthetic 20260928"
EXPECTED_MODEL_SHA = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"


def verify_model(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    raw = gzip.decompress(base64.b64decode(text, validate=True))
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_MODEL_SHA:
        raise SystemExit(f"model fingerprint mismatch: {actual}")
    return text


runner = RUNNER.read_text(encoding="utf-8")
compile(runner, str(RUNNER), "exec")
for required in [
    "BASELINE_REAL_ONLY",
    "VS_FULL_LOCAL_MIXUP_100",
    "V61_NESTED_SELECTED",
    "SYNTHETIC_SOURCE_LEAKAGE",
    "RESULTS_V6_1.json",
    "RUN_MANIFEST_V6_1.json",
    "LEAKAGE_AUDIT_V6_1.json",
    "V5_BASELINE_REPRODUCTION_DRIFT_GT_2PCT",
    "does_not_supersede_v5",
]:
    if required not in runner:
        raise SystemExit("required V6.1 guard missing: " + required)

model_b64 = verify_model(MODEL_B64)
future_line = "from __future__ import annotations"
if runner.count(future_line) != 1:
    raise SystemExit("runner future-import contract changed")
source = runner.replace(
    future_line,
    future_line + "\nMODEL_CSV_GZ_B64 = " + repr(model_b64),
    1,
)
compile(source, "soilbin_v6_1_notebook_source.py", "exec")
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SoilBin V6.1 â€” leakage-safe virtual sensing and synthetic augmentation\n",
                "Post-lock exploratory challenge. V5 remains frozen and must reproduce before interpretation.\n",
                "Virtual channels are deterministic transforms of previous LC1/LC5; they are not measured sensors.\n",
                "Synthetic rows are generated inside training folds only; all validation and outer-test rows are real.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "soilbin": {
            "schema": "soilbin.q1.v6.1.virtual-synthetic",
            "cpu_only": True,
            "post_lock_exploratory": True,
            "does_not_supersede_v5": True,
            "candidate_family": ["RandomForest", "ExtraTrees"],
            "random_split_used": False,
            "synthetic_rows_train_only": True,
            "evaluation_real_only": True,
            "virtual_channels_are_physical_sensors": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
notebook_text = json.dumps(notebook, ensure_ascii=False, indent=1)
notebook_sha = hashlib.sha256(notebook_text.encode("utf-8")).hexdigest()
print("SOILBIN_V6_1_NOTEBOOK_VALIDATED", notebook_sha, flush=True)

token = os.environ.get("CGP_ACTION_OIDC_TOKEN", "")
if len(token) < 100:
    raise SystemExit("CGP_ACTION_OIDC_TOKEN unavailable")

payload = {
    "request_id": "soilbin-q1-v6-1-virtual-synthetic-kg09-20260928-r1",
    "provider": "kaggle",
    "operation_class": "compute",
    "account_id": "kg-09",
    "purpose": (
        "SoilBin V6.1 post-lock exploratory challenge: exact V5-family baseline reproduction, "
        "deterministic previous-pass virtual channels, and training-fold-only jitter/local mixup "
        "under nested leave-one-SpeedxLoad-group-out validation with explicit leakage auditing."
    ),
    "service": "kernels.KernelsApiService",
    "method": "SaveKernel",
    "body": {
        "slug": KERNEL_REF,
        "newTitle": TITLE,
        "text": notebook_text,
        "language": "PYTHON",
        "kernelType": "NOTEBOOK",
        "kernelExecutionType": "SAVE_AND_RUN_ALL",
        "isPrivate": True,
        "enableGpu": False,
        "enableInternet": True,
    },
}
request = urllib.request.Request(
    ENDPOINT,
    data=json.dumps(payload, separators=(",", ":")).encode(),
    method="POST",
    headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "soilbin-q1-v6-1/1.0",
    },
)
try:
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read().decode("utf-8", "replace") or "{}")
except urllib.error.HTTPError as exc:
    detail = exc.read(5000).decode("utf-8", "replace")
    raise SystemExit(f"Cloudflare action HTTP {exc.code}: {detail}") from exc

if not result.get("ok"):
    raise SystemExit("Cloudflare action returned ok=false: " + json.dumps(result)[:2500])

provider_ref = result.get("provider_ref") or (result.get("result") or {}).get("ref") or KERNEL_REF
receipt = {
    "accepted": True,
    "account_id": "kg-09",
    "owner": "mylovevpn1",
    "kernel_ref": provider_ref,
    "notebook_sha256": notebook_sha,
    "runner_sha256": hashlib.sha256(runner.encode("utf-8")).hexdigest(),
    "model_csv_sha256": EXPECTED_MODEL_SHA,
    "cpu_only": True,
    "gpu": False,
    "post_lock_exploratory": True,
    "does_not_supersede_v5": True,
    "synthetic_rows_train_only": True,
    "evaluation_real_only": True,
    "broker_run_id": result.get("broker_run_id"),
    "broker_sha": result.get("broker_sha"),
}
print("SOILBIN_V6_1_LAUNCH_ACCEPTED", json.dumps(receipt, ensure_ascii=False), flush=True)
receipt_path = pathlib.Path(os.environ.get("RUNNER_TEMP", ".")) / "soilbin_v6_1_launch_receipt.json"
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as handle:
    handle.write(f"SOILBIN_V61_RECEIPT_PATH={receipt_path}\n")
    handle.write(f"SOILBIN_V61_KERNEL_REF={provider_ref}\n")
