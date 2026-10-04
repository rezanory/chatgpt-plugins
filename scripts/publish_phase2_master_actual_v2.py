from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

import phase2_final_evidence_extract_v1 as broker

KERNEL_REF = "azadka/pneumonia-phase2-master-actual-kernels-v2"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    if len(read_token) < 100 or len(action_token) < 100:
        raise SystemExit("OIDC_TOKEN_INVALID")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.notebook.read_bytes()
    source = raw.decode("utf-8")
    expected_sha = sha256_bytes(raw)

    payload = {
        "request_id": "phase2-master-actual-kernels-v2-" + os.environ["GITHUB_RUN_ID"],
        "provider": "kaggle",
        "operation_class": "compute",
        "account_id": "master",
        "purpose": (
            "Quick-save exact consolidated notebook harvested from the 36 completed "
            "Phase-2 Kaggle kernels. No training or inference."
        ),
        "service": "kernels.KernelsApiService",
        "method": "SaveKernel",
        "body": {
            "slug": KERNEL_REF,
            "newTitle": "Pneumonia Phase2 Master - Actual 36 Kaggle Kernels V2",
            "text": source,
            "language": "PYTHON",
            "kernelType": "NOTEBOOK",
            "kernelExecutionType": "QUICK_SAVE",
            "isPrivate": True,
            "enableGpu": False,
            "enableTpu": False,
            "enableInternet": False,
            "datasetDataSources": [],
            "competitionDataSources": [],
            "modelDataSources": [],
        },
    }
    launch = broker.post_json(
        broker.ACTION_ENDPOINT, action_token, payload, timeout=300
    )
    (args.output_dir / "KAGGLE_MASTER_ACTUAL_SAVE.json").write_text(
        json.dumps(launch, indent=2), encoding="utf-8"
    )
    result = launch.get("result") if isinstance(launch.get("result"), dict) else {}
    provider_error = str(result.get("error") or launch.get("error") or "")
    if not launch.get("ok") or provider_error:
        raise RuntimeError(
            "KAGGLE_MASTER_ACTUAL_QUICK_SAVE_REJECTED=" + json.dumps(launch)
        )

    readback = broker.post_json(
        broker.READ_ENDPOINT,
        read_token,
        {
            "action": "raw_read",
            "account_id": "master",
            "service": "kernels.KernelsApiService",
            "method": "GetKernel",
            "body": {
                "userName": "azadka",
                "kernelSlug": "pneumonia-phase2-master-actual-kernels-v2",
            },
        },
        timeout=240,
    )
    (args.output_dir / "KAGGLE_MASTER_ACTUAL_GET.json").write_text(
        json.dumps(readback, indent=2), encoding="utf-8"
    )
    if not readback.get("ok"):
        raise RuntimeError("KAGGLE_MASTER_ACTUAL_READBACK_FAILED")

    value = broker._read_payload(readback)
    metadata = value.get("metadata") or {}
    blob = value.get("blob") or {}
    actual_source = blob.get("source")
    if not isinstance(actual_source, str):
        raise RuntimeError("KAGGLE_MASTER_ACTUAL_READBACK_SOURCE_MISSING")
    actual_sha = sha256_bytes(actual_source.encode("utf-8"))
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"KAGGLE_MASTER_ACTUAL_SHA_MISMATCH:{expected_sha}:{actual_sha}"
        )

    receipt = {
        "schema": "pneumonia.phase2.master.actual-kaggle-sources.publish.v2",
        "status": "PASS",
        "kernel_ref": metadata.get("ref") or KERNEL_REF,
        "kernel_id": metadata.get("id"),
        "version": metadata.get("currentVersionNumber"),
        "source_sha256": actual_sha,
        "quick_save": True,
        "training_performed": False,
        "inference_performed": False,
    }
    (args.output_dir / "KAGGLE_MASTER_ACTUAL_VERIFY.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("KAGGLE_MASTER_ACTUAL_READBACK_PASS " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
