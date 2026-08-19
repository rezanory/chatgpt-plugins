from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT = "PNEUMONIA V6.2.2"
KERNEL = "trickermark/pneumonia-v6-2-2-backbone-m06-r224"
RECIPE_SHA = "27cae8c59e06b609f9dfd02b526d807dc359b2bb14ae7bc5ee77c68c7553b08f"
POLICY_FILE_SHA = "7e23bbed9246d5588c67a46380b570e1c1a3e869062613b9dcdb298e2e822861"
POLICY_INTERNAL_SHA = "7f5817f3429bb86668718cac9158a7061c6e730cd14bc6137402be82baf576d7"
IDS = ["M06__convnext_tiny", "M06__densenet121", "M06__resnet50v2"]
EXPECTED_CONFIG_SHA = {
    "M06__convnext_tiny": "167cb7b1d17b3978f2b4f47f55e315a1d4fae293006d897d25d13b5074dadcea",
    "M06__densenet121": "c9b3107715849a633d90689902298c110bbb7c9c1bc1c8a6ac9ff1693f55386e",
    "M06__resnet50v2": "dce7845aa7c07ccb913cded8e9ff368e7061f3a43723ded21f8a94dba9fab697",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stage1-run-id", required=True)
    ap.add_argument("--stage1-run-url", required=True)
    args = ap.parse_args()

    root = args.root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest" / "SELECTED_ARTIFACT_MANIFEST.json"
    manifest_sha_path = root / "manifest" / "SELECTED_ARTIFACT_MANIFEST.sha256"
    if not manifest_path.is_file() or not manifest_sha_path.is_file():
        raise SystemExit("Stage 1 manifest payload incomplete")

    stage1_manifest_sha = sha256(manifest_path)
    declared_stage1_sha = manifest_sha_path.read_text(encoding="utf-8").strip().lower()
    if stage1_manifest_sha != declared_stage1_sha:
        raise SystemExit("Stage 1 manifest SHA mismatch")

    stage1 = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stage1.get("project") != PROJECT:
        raise SystemExit("Wrong Stage 1 project")
    if stage1.get("status") != "RECOVERED_SELECTED_ONLY":
        raise SystemExit("Stage 1 not recovery-complete")
    if stage1.get("kernel_ref") != KERNEL:
        raise SystemExit("Wrong source kernel")
    if stage1.get("recipe_sha256") != RECIPE_SHA:
        raise SystemExit("Recipe SHA mismatch")
    if stage1.get("frozen_policy_sha256") != POLICY_FILE_SHA:
        raise SystemExit("Frozen policy file SHA mismatch")
    if list(stage1.get("selected_candidate_ids") or []) != IDS:
        raise SystemExit("Selected candidate identity/order mismatch")
    if int(stage1.get("selected_artifact_count", -1)) != 6:
        raise SystemExit("Selected artifact count mismatch")
    if stage1.get("locked_test_used") is not False or stage1.get("external_data_used") is not False:
        raise SystemExit("Forbidden evidence appears in Stage 1")

    rows = list(stage1.get("artifacts") or [])
    by_key = {(r.get("candidate_id"), r.get("role")): r for r in rows}
    if len(rows) != 6 or len(by_key) != 6:
        raise SystemExit("Stage 1 artifact rows are not exactly six unique rows")

    dirs = {
        "M06__convnext_tiny": root / "convnext",
        "M06__densenet121": root / "densenet",
        "M06__resnet50v2": root / "resnet",
    }
    reconciled = []
    for cid in IDS:
        for role, basename in (("config", "config.json"), ("checkpoint", "final_selected.keras")):
            path = dirs[cid] / basename
            if not path.is_file() or path.stat().st_size <= 0:
                raise SystemExit(f"Missing/empty selected artifact: {cid} {role}")
            digest = sha256(path)
            src = by_key.get((cid, role))
            if src is None:
                raise SystemExit(f"Missing Stage 1 row: {cid} {role}")
            if digest != str(src.get("sha256", "")).lower():
                raise SystemExit(f"Stage1/download SHA mismatch: {cid} {role}")
            if path.stat().st_size != int(src.get("bytes", -1)):
                raise SystemExit(f"Stage1/download size mismatch: {cid} {role}")
            if role == "config" and digest != EXPECTED_CONFIG_SHA[cid]:
                raise SystemExit(f"Canonical config SHA mismatch: {cid}")
            reconciled.append(
                {
                    "candidate_id": cid,
                    "role": role,
                    "basename": basename,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "source_file_name": src.get("source_file_name"),
                }
            )

    qualification = {
        "schema_version": 1,
        "project": PROJECT,
        "stage": "QUALIFICATION_NON_CONSUMPTION",
        "status": "RECORDED",
        "selection_basis": "validation-only recovered canonical backbone governance",
        "selected_candidate_ids": IDS,
        "recipe_sha256": RECIPE_SHA,
        "frozen_policy_sha256": POLICY_FILE_SHA,
        "frozen_policy_file_sha256": POLICY_FILE_SHA,
        "frozen_policy_internal_sha256": POLICY_INTERNAL_SHA,
        "stage1_selected_artifact_manifest_sha256": stage1_manifest_sha,
        "used_for_selection": False,
        "qualification_performed": False,
        "qualification_evidence_consumed": False,
        "locked_test_used": False,
        "external_data_used": False,
        "note": "No qualification result exists or was consumed before champion/policy freeze; qualification remains intentionally unperformed at this gate.",
    }
    qualification_path = out_dir / "QUALIFICATION_NON_CONSUMPTION.json"
    qualification_sha = write_json(qualification_path, qualification)
    (out_dir / "QUALIFICATION_NON_CONSUMPTION.sha256").write_text(qualification_sha + "\n", encoding="utf-8")

    final_manifest = {
        "schema_version": 1,
        "project": PROJECT,
        "stage": "FINAL_FREEZE_MANIFEST",
        "status": "FROZEN",
        "source_kernel": KERNEL,
        "stage1_recovery_run_id": int(args.stage1_run_id),
        "stage1_recovery_run_url": args.stage1_run_url,
        "stage1_selected_artifact_manifest_sha256": stage1_manifest_sha,
        "recipe_sha256": RECIPE_SHA,
        "frozen_policy_sha256": POLICY_FILE_SHA,
        "frozen_policy_file_sha256": POLICY_FILE_SHA,
        "frozen_policy_internal_sha256": POLICY_INTERNAL_SHA,
        "selection_basis": "validation-only recovered canonical backbone governance",
        "selected_candidate_ids": IDS,
        "ensemble_weights": {cid: {"numerator": 1, "denominator": 3} for cid in IDS},
        "selection_evidence": {
            "oof_balanced_accuracy": 0.9977578475336323,
            "oof_mcc": 0.9913388287238792,
            "false_positives": 0,
            "false_negatives": 2,
        },
        "artifacts": reconciled,
        "selected_artifact_count": 6,
        "qualification_non_consumption_sha256": qualification_sha,
        "qualification_used_for_selection": False,
        "qualification_performed": False,
        "selection_locked": True,
        "policy_frozen": True,
        "locked_test_used": False,
        "external_data_used": False,
        "new_kaggle_compute_launched": False,
        "canonical_m01_m12_rerun": False,
        "hpo_or_confirmation_repeated": False,
        "independent_reconciliation": "PENDING",
        "next_permitted_stage": "INDEPENDENT_FINAL_FREEZE_RECONCILIATION",
    }
    final_path = out_dir / "FINAL_FREEZE_MANIFEST.json"
    final_sha = write_json(final_path, final_manifest)
    (out_dir / "FINAL_FREEZE_MANIFEST.sha256").write_text(final_sha + "\n", encoding="utf-8")

    print(f"STAGE1_MANIFEST_SHA256={stage1_manifest_sha}")
    print(f"QUALIFICATION_NON_CONSUMPTION_SHA256={qualification_sha}")
    print(f"FINAL_FREEZE_MANIFEST_SHA256={final_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
