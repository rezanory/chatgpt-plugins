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
    ap.add_argument("--stage1-root", type=Path, required=True)
    ap.add_argument("--freeze-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    stage1_root = args.stage1_root
    freeze_dir = args.freeze_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_manifest_path = stage1_root / "manifest" / "SELECTED_ARTIFACT_MANIFEST.json"
    stage1_manifest_sha_path = stage1_root / "manifest" / "SELECTED_ARTIFACT_MANIFEST.sha256"
    final_path = freeze_dir / "FINAL_FREEZE_MANIFEST.json"
    final_sha_path = freeze_dir / "FINAL_FREEZE_MANIFEST.sha256"
    qual_path = freeze_dir / "QUALIFICATION_NON_CONSUMPTION.json"
    qual_sha_path = freeze_dir / "QUALIFICATION_NON_CONSUMPTION.sha256"

    for p in (stage1_manifest_path, stage1_manifest_sha_path, final_path, final_sha_path, qual_path, qual_sha_path):
        if not p.is_file():
            raise SystemExit(f"Missing reconciliation input: {p}")

    stage1_manifest_sha = sha256(stage1_manifest_path)
    if stage1_manifest_sha != stage1_manifest_sha_path.read_text(encoding="utf-8").strip().lower():
        raise SystemExit("Stage 1 manifest SHA mismatch during independent reconciliation")
    final_sha = sha256(final_path)
    if final_sha != final_sha_path.read_text(encoding="utf-8").strip().lower():
        raise SystemExit("FINAL_FREEZE_MANIFEST SHA mismatch")
    qual_sha = sha256(qual_path)
    if qual_sha != qual_sha_path.read_text(encoding="utf-8").strip().lower():
        raise SystemExit("Qualification non-consumption SHA mismatch")

    stage1 = json.loads(stage1_manifest_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    qual = json.loads(qual_path.read_text(encoding="utf-8"))

    if final.get("project") != PROJECT or final.get("stage") != "FINAL_FREEZE_MANIFEST" or final.get("status") != "FROZEN":
        raise SystemExit("Final freeze identity/status mismatch")
    if final.get("source_kernel") != KERNEL:
        raise SystemExit("Final freeze kernel mismatch")
    if final.get("recipe_sha256") != RECIPE_SHA:
        raise SystemExit("Final freeze recipe mismatch")
    if final.get("frozen_policy_sha256") != POLICY_FILE_SHA or final.get("frozen_policy_file_sha256") != POLICY_FILE_SHA:
        raise SystemExit("Final freeze policy file SHA mismatch")
    if final.get("frozen_policy_internal_sha256") != POLICY_INTERNAL_SHA:
        raise SystemExit("Final freeze policy internal SHA mismatch")
    if final.get("stage1_selected_artifact_manifest_sha256") != stage1_manifest_sha:
        raise SystemExit("Final freeze does not bind exact Stage 1 manifest")
    if list(final.get("selected_candidate_ids") or []) != IDS:
        raise SystemExit("Final freeze selected candidate mismatch")
    if int(final.get("selected_artifact_count", -1)) != 6:
        raise SystemExit("Final freeze selected artifact count mismatch")
    if final.get("qualification_non_consumption_sha256") != qual_sha:
        raise SystemExit("Final freeze qualification binding mismatch")
    if final.get("qualification_used_for_selection") is not False or final.get("qualification_performed") is not False:
        raise SystemExit("Final freeze qualification flags are not honest non-consumption")
    if final.get("selection_locked") is not True or final.get("policy_frozen") is not True:
        raise SystemExit("Champion/policy are not frozen")
    if final.get("locked_test_used") is not False or final.get("external_data_used") is not False:
        raise SystemExit("Forbidden evidence appears in final freeze")
    if final.get("independent_reconciliation") != "PENDING":
        raise SystemExit("Final freeze was expected to await independent reconciliation")

    if qual.get("project") != PROJECT or qual.get("stage") != "QUALIFICATION_NON_CONSUMPTION":
        raise SystemExit("Qualification record identity mismatch")
    for key in ("used_for_selection", "qualification_performed", "qualification_evidence_consumed", "locked_test_used", "external_data_used"):
        if qual.get(key) is not False:
            raise SystemExit(f"Qualification non-consumption flag mismatch: {key}")
    if qual.get("recipe_sha256") != RECIPE_SHA:
        raise SystemExit("Qualification record recipe mismatch")
    if qual.get("frozen_policy_sha256") != POLICY_FILE_SHA or qual.get("frozen_policy_file_sha256") != POLICY_FILE_SHA:
        raise SystemExit("Qualification record policy file SHA mismatch")
    if qual.get("frozen_policy_internal_sha256") != POLICY_INTERNAL_SHA:
        raise SystemExit("Qualification record policy internal SHA mismatch")
    if qual.get("stage1_selected_artifact_manifest_sha256") != stage1_manifest_sha:
        raise SystemExit("Qualification record Stage 1 binding mismatch")

    weights = final.get("ensemble_weights") or {}
    if set(weights) != set(IDS):
        raise SystemExit("Ensemble member set mismatch")
    for cid in IDS:
        if weights[cid] != {"numerator": 1, "denominator": 3}:
            raise SystemExit(f"Ensemble weight mismatch: {cid}")

    stage1_rows = {(r.get("candidate_id"), r.get("role")): r for r in (stage1.get("artifacts") or [])}
    final_rows = {(r.get("candidate_id"), r.get("role")): r for r in (final.get("artifacts") or [])}
    if len(stage1_rows) != 6 or len(final_rows) != 6:
        raise SystemExit("Artifact row cardinality mismatch")

    dirs = {
        "M06__convnext_tiny": stage1_root / "convnext",
        "M06__densenet121": stage1_root / "densenet",
        "M06__resnet50v2": stage1_root / "resnet",
    }
    verified = []
    for cid in IDS:
        for role, basename in (("config", "config.json"), ("checkpoint", "final_selected.keras")):
            path = dirs[cid] / basename
            if not path.is_file() or path.stat().st_size <= 0:
                raise SystemExit(f"Missing/empty independent artifact: {cid} {role}")
            digest = sha256(path)
            s = stage1_rows[(cid, role)]
            f = final_rows[(cid, role)]
            if digest != str(s.get("sha256", "")).lower() or digest != str(f.get("sha256", "")).lower():
                raise SystemExit(f"Independent SHA mismatch: {cid} {role}")
            if path.stat().st_size != int(s.get("bytes", -1)) or path.stat().st_size != int(f.get("bytes", -1)):
                raise SystemExit(f"Independent size mismatch: {cid} {role}")
            if role == "config" and digest != EXPECTED_CONFIG_SHA[cid]:
                raise SystemExit(f"Independent canonical config mismatch: {cid}")
            verified.append({"candidate_id": cid, "role": role, "bytes": path.stat().st_size, "sha256": digest})

    evidence = final.get("selection_evidence") or {}
    expected_evidence = {
        "oof_balanced_accuracy": 0.9977578475336323,
        "oof_mcc": 0.9913388287238792,
        "false_positives": 0,
        "false_negatives": 2,
    }
    if evidence != expected_evidence:
        raise SystemExit("Validation-only selection evidence mismatch")

    record = {
        "schema_version": 1,
        "project": PROJECT,
        "stage": "FINAL_FREEZE_RECONCILIATION",
        "status": "PASS",
        "independent_reconciliation": True,
        "final_freeze_manifest_sha256": final_sha,
        "qualification_non_consumption_sha256": qual_sha,
        "stage1_selected_artifact_manifest_sha256": stage1_manifest_sha,
        "recipe_sha256": RECIPE_SHA,
        "frozen_policy_sha256": POLICY_FILE_SHA,
        "frozen_policy_file_sha256": POLICY_FILE_SHA,
        "frozen_policy_internal_sha256": POLICY_INTERNAL_SHA,
        "selected_candidate_ids": IDS,
        "verified_artifacts": verified,
        "selection_locked": True,
        "policy_frozen": True,
        "used_for_selection": False,
        "qualification_performed": False,
        "locked_test_used": False,
        "external_data_used": False,
        "lockbox_access_authorized": False,
        "next_required_gate": "HARDEN_DISABLE_OBSOLETE_CLOUDFLARE_MUTATING_WORKFLOWS",
    }
    out = out_dir / "FINAL_FREEZE_RECONCILIATION.json"
    out_sha = write_json(out, record)
    (out_dir / "FINAL_FREEZE_RECONCILIATION.sha256").write_text(out_sha + "\n", encoding="utf-8")
    print(f"FINAL_FREEZE_RECONCILIATION_SHA256={out_sha}")
    print(f"FINAL_FREEZE_MANIFEST_SHA256={final_sha}")
    print(f"QUALIFICATION_NON_CONSUMPTION_SHA256={qual_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
