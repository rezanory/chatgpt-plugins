from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ACTION_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/action/kaggle"
SOURCE_ACCOUNT = "kg-03"
SOURCE_OWNER = "rezanory"
SOURCE_SLUG = "m07-phase2-unlock-20260915"
SOURCE_REF = f"{SOURCE_OWNER}/{SOURCE_SLUG}"
TARGET_DATASET = "trickermark/m07-r320-producer-v4-bridge-e3884dd1"
EXPECTED_SPLIT = "896491de87f9dc2a1d7d63548b7c5c22206da11f27a37efece8efc8e1557c8a9"
EXPECTED_RECIPE = "02c77dd257612e866756b16270f71816e61fb9f2403e14126aaacb9f01a8da0b"
EXPECTED_CAMPAIGN_RECEIPT = "e3884dd10e323d2f0925660599a420964781111d850664e8c88a830324460c77"
EXPECTED_ARCHIVES = {
    "FOLD_1_RECOVERY.zip": "cd18468bf9305bb801f809098f8917462b16fb69fe9313b8bd17f2a95a7d1531",
    "FOLD_2_RECOVERY.zip": "a75967888cec5cea6133a2ed457bb998edb9ada01442be88b2cc35e970e6f72d",
    "FOLD_3_RECOVERY.zip": "4b04f657ddfcab61dc720f17ba4a8a5a8f234cd2bb7ed10b64e7357e049a9ad6",
    "FOLD_4_RECOVERY.zip": "74a7c69c572a66a139beb99dfa375185ce2f25e96689642eac2a1e56928deb93",
}
STATE_PREFIX = "M07_GATE_MULTIRES_V17/STATE/R320/"
CAMPAIGN_PATH = STATE_PREFIX + "CAMPAIGN_STATE.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def broker(payload: dict) -> dict:
    token = os.environ.get("CGP_ACTION_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise RuntimeError("CGP_ACTION_OIDC_TOKEN unavailable")
    request = urllib.request.Request(
        ACTION_ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "m07-r320-producer-bridge/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8", "replace") or "{}")
    if result.get("ok") is not True:
        raise RuntimeError("trusted action broker rejected source inventory")
    inner = result.get("result")
    if not isinstance(inner, dict):
        raise RuntimeError("trusted action broker returned no result object")
    return inner


def list_source_outputs() -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for page in range(1, 30):
        outer = broker(
            {
                "request_id": f"m07-r320-producer-bridge-list-{os.environ.get('GITHUB_RUN_ID', 'local')}-{page}",
                "provider": "kaggle",
                "operation_class": "privileged",
                "account_id": SOURCE_ACCOUNT,
                "purpose": (
                    "Read exact output inventory from the verified M07 R320 producer only; "
                    "no SaveKernel, no upload, no training, no Locked Test."
                ),
                "service": "kernels.KernelsApiService",
                "method": "ListKernelSessionOutput",
                "body": {
                    "userName": SOURCE_OWNER,
                    "kernelSlug": SOURCE_SLUG,
                    "page": page,
                    "pageSize": 100,
                },
            }
        )
        files = outer.get("files") or []
        for item in files:
            if not isinstance(item, dict):
                continue
            name = str(item.get("fileName") or "").strip().replace("\\", "/")
            if name:
                entries[name] = item
        if len(files) < 100:
            break
    if not entries:
        raise RuntimeError("producer output inventory is empty")
    return entries


def download_exact(item: dict, destination: Path) -> None:
    url = str(item.get("url") or "").strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.kaggleusercontent.com":
        raise RuntimeError("unexpected producer output host")
    req = urllib.request.Request(url, headers={"User-Agent": "m07-r320-producer-bridge/1.0"})
    with urllib.request.urlopen(req, timeout=300) as response:
        destination.write_bytes(response.read())


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            pp = PurePosixPath(info.filename)
            if pp.is_absolute() or ".." in pp.parts:
                raise RuntimeError(f"unsafe archive member: {info.filename}")
            target = destination.joinpath(*pp.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def verify_fold_archive(archive: Path, fold_id: int) -> dict:
    extract_root = archive.parent / f"_verify_fold_{fold_id}"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    safe_extract(archive, extract_root)
    receipts = [
        p
        for p in extract_root.rglob("COMPLETED.json")
        if f"fold_{fold_id}" in [part.lower() for part in p.parts]
    ]
    if len(receipts) != 1:
        raise RuntimeError(f"fold {fold_id} recovery receipt inventory mismatch: {len(receipts)}")
    receipt_path = receipts[0]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "pneumonia.phase2.fold.v1.7" or receipt.get("status") != "COMPLETED":
        raise RuntimeError(f"fold {fold_id} receipt status/schema mismatch")
    if int(receipt.get("fold_id", -1)) != fold_id or int(receipt.get("resolution", -1)) != 320:
        raise RuntimeError(f"fold {fold_id} receipt identity mismatch")
    if receipt.get("model_id") != "M07":
        raise RuntimeError(f"fold {fold_id} model identity mismatch")
    if receipt.get("split_fingerprint") != EXPECTED_SPLIT:
        raise RuntimeError(f"fold {fold_id} split fingerprint mismatch")
    if receipt.get("confirmed_m07_recipe_fingerprint") != EXPECTED_RECIPE:
        raise RuntimeError(f"fold {fold_id} recipe fingerprint mismatch")
    if receipt.get("locked_test_used_for_training") is not False:
        raise RuntimeError(f"fold {fold_id} locked-test training flag is not false")
    if receipt.get("external_used_for_training") is not False:
        raise RuntimeError(f"fold {fold_id} external-training flag is not false")

    artifact_hashes = receipt.get("artifact_sha256") or {}
    verified_artifacts: dict[str, str] = {}
    for file_name, expected_hash in sorted(artifact_hashes.items()):
        matches = [p for p in extract_root.rglob(file_name) if p.is_file()]
        if len(matches) != 1:
            raise RuntimeError(f"fold {fold_id} artifact inventory mismatch for {file_name}: {len(matches)}")
        observed = sha256_file(matches[0])
        if observed != expected_hash:
            raise RuntimeError(f"fold {fold_id} artifact sha mismatch for {file_name}")
        verified_artifacts[file_name] = observed

    return {
        "fold_id": fold_id,
        "receipt_sha256": receipt.get("receipt_sha256"),
        "run_fingerprint": receipt.get("run_fingerprint"),
        "artifact_sha256": verified_artifacts,
    }


def main() -> int:
    temp_root = Path(os.environ.get("RUNNER_TEMP") or Path.cwd() / ".tmp") / "m07-r320-producer-bridge"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    bridge = temp_root / "dataset"
    bridge.mkdir(parents=True)

    outputs = list_source_outputs()
    required_paths = [CAMPAIGN_PATH] + [STATE_PREFIX + name for name in EXPECTED_ARCHIVES]
    missing = [name for name in required_paths if name not in outputs]
    if missing:
        raise RuntimeError("producer required output missing: " + json.dumps(missing))

    campaign_out = bridge / "CAMPAIGN_STATE.json"
    download_exact(outputs[CAMPAIGN_PATH], campaign_out)
    campaign = json.loads(campaign_out.read_text(encoding="utf-8"))
    if campaign.get("schema") != "phase2.state.v2" or campaign.get("status") != "COMPLETE":
        raise RuntimeError("producer campaign state schema/status mismatch")
    if campaign.get("model_id") != "M07" or int(campaign.get("resolution", -1)) != 320:
        raise RuntimeError("producer campaign identity mismatch")
    if campaign.get("receipt_sha256") != EXPECTED_CAMPAIGN_RECEIPT:
        raise RuntimeError("producer campaign receipt mismatch")
    contract = campaign.get("run_contract") or {}
    if contract.get("split_fingerprint") != EXPECTED_SPLIT:
        raise RuntimeError("producer campaign split fingerprint mismatch")
    recipe = ((contract.get("extra") or {}).get("recipe_fingerprint"))
    if recipe != EXPECTED_RECIPE:
        raise RuntimeError("producer campaign recipe fingerprint mismatch")
    if (campaign.get("artifact_sha256") or {}) != EXPECTED_ARCHIVES:
        raise RuntimeError("producer campaign archive manifest mismatch")

    folds: list[dict] = []
    archive_manifest: dict[str, dict] = {}
    for fold_id in range(1, 5):
        name = f"FOLD_{fold_id}_RECOVERY.zip"
        destination = bridge / name
        download_exact(outputs[STATE_PREFIX + name], destination)
        observed = sha256_file(destination)
        if observed != EXPECTED_ARCHIVES[name]:
            raise RuntimeError(f"producer archive sha mismatch for {name}")
        archive_manifest[name] = {"sha256": observed, "bytes": destination.stat().st_size}
        folds.append(verify_fold_archive(destination, fold_id))

    receipt = {
        "schema": "m07.r320.producer.bridge.v1",
        "status": "VERIFIED_SOURCE_READY_FOR_PRIVATE_BRIDGE_UPLOAD",
        "source_kernel_ref": SOURCE_REF,
        "source_account_id": SOURCE_ACCOUNT,
        "target_dataset": TARGET_DATASET,
        "resolution": 320,
        "model_id": "M07",
        "split_fingerprint": EXPECTED_SPLIT,
        "recipe_fingerprint": EXPECTED_RECIPE,
        "campaign_receipt_sha256": EXPECTED_CAMPAIGN_RECEIPT,
        "campaign_state_sha256": sha256_file(campaign_out),
        "archive_manifest": archive_manifest,
        "folds": folds,
        "sealed_fold_ids": [1, 2, 3, 4],
        "missing_fold_ids": [5],
        "locked_test_used_for_training": False,
        "external_used_for_training": False,
        "mutable_dataset_versions_5_6_used": False,
        "training_performed": False,
        "upload_permitted_after_this_receipt": True,
    }
    receipt_path = bridge / "BRIDGE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["bridge_receipt_sha256"] = sha256_file(receipt_path)

    metadata = {
        "title": "M07 R320 Producer V4 Recovery Bridge e3884dd1",
        "id": TARGET_DATASET,
        "licenses": [{"name": "other"}],
        "description": (
            "Private fail-closed recovery bridge copied only from the verified output of "
            "rezanory/m07-phase2-unlock-20260915. Contains exact R320 sealed folds 1-4; "
            "Fold 5 is intentionally absent. Mutable R320 dataset versions 5/6 are not used."
        ),
    }
    (bridge / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as handle:
            handle.write(f"M07_BRIDGE_DIR={bridge}\n")
            handle.write(f"M07_BRIDGE_DATASET={TARGET_DATASET}\n")
            handle.write(f"M07_BRIDGE_RECEIPT_SHA256={receipt['bridge_receipt_sha256']}\n")

    print(json.dumps({"status": "VERIFIED", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
