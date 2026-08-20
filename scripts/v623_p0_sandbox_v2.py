from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = "PNEUMONIA V6.2.3-P0"
SOURCE_ZIP_B64 = "__SOURCE_ZIP_B64__"
RAW_DATASET = "paultimothymooney/chest-xray-pneumonia"
MODEL_ID = "M06"
BACKBONE = "convnext_tiny"
IMAGE_SIZE = 224
BATCH_SIZE = 8
SEEDS = [42, 2026]
HEAD_EPOCHS = 2
FINETUNE_EPOCHS = 12
DROPOUT = 0.15
HEAD_LR = 5.8025153801026264e-05
FINETUNE_LR = 4.723037146949559e-05
WEIGHT_DECAY = 1.637057637057399e-05
POOL_TARGETS = {"train": 480, "cal": 120, "shadow": 120}
CLASS_TARGETS = {
    "NORMAL": {"train": 240, "cal": 60, "shadow": 60},
    "PNEUMONIA": {"train": 240, "cal": 60, "shadow": 60},
}
SPLIT_SEED_BASE = 62300
NEAR_DUP_HAMMING = 2
MAX_SPLIT_ATTEMPTS = 40
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/V623_P0_WORK")
OUT = Path("/kaggle/working/V623_P0")
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
START = time.perf_counter()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return sha256_file(path)


source_zip = WORK / "source_python.zip"
source_zip.write_bytes(base64.b64decode(SOURCE_ZIP_B64))
SOURCE_ZIP_SHA256 = sha256_file(source_zip)
source_dir = WORK / "source"
source_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(source_zip) as zf:
    zf.extractall(source_dir)
PROJECT_CODE = source_dir / "SOURCE_PYTHON"
if not PROJECT_CODE.is_dir():
    raise RuntimeError("SOURCE_PYTHON missing from injected canonical source package")
sys.path.insert(0, str(PROJECT_CODE))

from build_effective_split import exact_group_subset, extract_patient_id
from auto_ensemble_selection import apply_platt_calibrator, expected_calibration_error, fit_platt_calibrator, select_threshold
import pneumonia_runner_base as runner
import tensorflow as tf

EXPECTED_SOURCE_ZIP_SHA256 = "ce3623b87ca221a52bcf886836070ebaebc490cf7574f6f3ce4db78095a32cde"
if SOURCE_ZIP_SHA256 != EXPECTED_SOURCE_ZIP_SHA256:
    raise RuntimeError(f"canonical W16 source ZIP SHA drift: {SOURCE_ZIP_SHA256}")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str
    source_partition: str
    patient_id: str
    patient_source: str


def looks_development_root(root: Path) -> bool:
    try:
        return all((root / part / cls).is_dir() for part in ("train", "val") for cls in ("NORMAL", "PNEUMONIA"))
    except Exception:
        return False


walk_dirs = [INPUT]
for current, dirs, _files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d.lower() != "test" and d != "__MACOSX"]
    for d in dirs:
        walk_dirs.append(Path(current) / d)
candidates = [p.resolve() for p in walk_dirs if looks_development_root(p)]
roots = [p for p in sorted(set(candidates), key=lambda x: (len(x.parts), str(x))) if not any(p != q and q in p.parents for q in candidates)]
if len(roots) != 1:
    raise RuntimeError(f"Expected exactly one TRAIN/VAL development root; found {roots}")
DATA_ROOT = roots[0]


def scan_dev_only() -> list[Sample]:
    rows: list[Sample] = []
    for partition in ("train", "val"):
        for class_name, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
            class_dir = DATA_ROOT / partition / class_name
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                    pid, source = extract_patient_id(path.name)
                    rows.append(Sample(path.resolve(), label, class_name, partition, pid, source))
    return rows


all_samples = scan_dev_only()
if not all_samples:
    raise RuntimeError("Development-only sample inventory is empty")
source_partition_counts = {part: sum(1 for s in all_samples if s.source_partition == part) for part in ("train", "val")}
class_counts = {cls: sum(1 for s in all_samples if s.class_name == cls) for cls in ("NORMAL", "PNEUMONIA")}
if min(class_counts.values()) < 360:
    raise RuntimeError(f"Insufficient development samples for balanced P0 pool: {class_counts}")

patient_labels: dict[str, set[str]] = {}
for sample in all_samples:
    patient_labels.setdefault(sample.patient_id, set()).add(sample.class_name)
mixed_patients = sorted(pid for pid, labels in patient_labels.items() if len(labels) > 1)
if mixed_patients:
    raise RuntimeError(f"Mixed-label patient groups detected; refusing P0 split: {mixed_patients[:10]}")


def class_groups(samples: list[Sample], class_name: str) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for item in samples:
        if item.class_name == class_name:
            groups.setdefault(item.patient_id, []).append(item)
    return groups


def select_group_subset(groups: dict[str, list[Sample]], target: int, seed: int) -> tuple[list[str], int]:
    selected, count = exact_group_subset(groups, target=target, seed=seed)
    return list(selected), int(count)


_hash_cache: dict[str, str] = {}
_dhash_cache: dict[str, int] = {}


def file_hash(path: Path) -> str:
    key = str(path)
    if key not in _hash_cache:
        _hash_cache[key] = sha256_file(path)
    return _hash_cache[key]


def dhash(path: Path) -> int:
    key = str(path)
    if key not in _dhash_cache:
        with Image.open(path) as image:
            gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = np.asarray(gray, dtype=np.uint8)
        bits = pixels[:, 1:] > pixels[:, :-1]
        value = 0
        for bit in bits.reshape(-1):
            value = (value << 1) | int(bit)
        _dhash_cache[key] = value
    return _dhash_cache[key]


def build_assignments(seed: int) -> dict[str, list[Sample]] | None:
    assignments: dict[str, list[Sample]] = {"train": [], "cal": [], "shadow": []}
    for class_index, class_name in enumerate(("NORMAL", "PNEUMONIA")):
        remaining = class_groups(all_samples, class_name)
        for offset, split_name in enumerate(("shadow", "cal", "train")):
            target = CLASS_TARGETS[class_name][split_name]
            chosen_ids, count = select_group_subset(remaining, target, seed + class_index * 1000 + offset * 101)
            if count != target:
                return None
            chosen = set(chosen_ids)
            for pid in chosen_ids:
                assignments[split_name].extend(remaining[pid])
            remaining = {pid: rows for pid, rows in remaining.items() if pid not in chosen}
    return assignments


def audit_assignments(assignments: dict[str, list[Sample]]) -> dict:
    patient_sets = {name: {s.patient_id for s in rows} for name, rows in assignments.items()}
    names = list(assignments)
    patient_overlap = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sorted(patient_sets[a] & patient_sets[b])
            if overlap:
                patient_overlap.append({"a": a, "b": b, "count": len(overlap), "examples": overlap[:10]})

    rows = []
    for split_name, samples in assignments.items():
        for sample in samples:
            rows.append({"split": split_name, "sample": sample, "sha256": file_hash(sample.path), "dhash": dhash(sample.path)})

    exact_cross = []
    by_sha: dict[str, list[dict]] = {}
    for row in rows:
        by_sha.setdefault(row["sha256"], []).append(row)
    for digest, matches in by_sha.items():
        splits = sorted({m["split"] for m in matches})
        if len(splits) > 1:
            exact_cross.append({"sha256": digest, "splits": splits, "count": len(matches)})

    near_cross = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left["split"] == right["split"] or left["sha256"] == right["sha256"]:
                continue
            dist = (int(left["dhash"]) ^ int(right["dhash"])).bit_count()
            if dist <= NEAR_DUP_HAMMING:
                near_cross.append({"a": left["split"], "b": right["split"], "distance": int(dist), "a_name": left["sample"].path.name, "b_name": right["sample"].path.name})
                if len(near_cross) >= 50:
                    break
        if len(near_cross) >= 50:
            break

    return {"patient_overlap": patient_overlap, "exact_duplicate_cross_split": exact_cross, "near_duplicate_cross_split": near_cross, "near_duplicate_hamming_threshold": NEAR_DUP_HAMMING, "pass": not patient_overlap and not exact_cross and not near_cross}


assignments = None
split_audit = None
split_seed = None
for attempt in range(MAX_SPLIT_ATTEMPTS):
    candidate_seed = SPLIT_SEED_BASE + attempt * 97
    candidate = build_assignments(candidate_seed)
    if candidate is None:
        continue
    if {k: len(v) for k, v in candidate.items()} != POOL_TARGETS:
        continue
    audit = audit_assignments(candidate)
    if audit["pass"]:
        assignments, split_audit, split_seed = candidate, audit, candidate_seed
        break
if assignments is None or split_audit is None or split_seed is None:
    raise RuntimeError("Could not construct a patient-disjoint, duplicate-clean 480/120/120 development-only P0 split")


def frame_for(rows: list[Sample], split_name: str) -> pd.DataFrame:
    values = []
    for sample in sorted(rows, key=lambda s: (s.class_name, s.patient_id, s.path.name)):
        digest = file_hash(sample.path)
        values.append({"sample_id": hashlib.sha256(f"{sample.source_partition}/{sample.class_name}/{sample.path.name}/{digest}".encode()).hexdigest()[:24], "filepath": str(sample.path), "label": int(sample.label), "class_name": sample.class_name, "patient_id": sample.patient_id, "patient_id_source": sample.patient_source, "source_partition": sample.source_partition, "p0_split": split_name, "sha256": digest})
    return pd.DataFrame(values)


train_df = frame_for(assignments["train"], "train")
cal_df = frame_for(assignments["cal"], "cal")
shadow_df = frame_for(assignments["shadow"], "shadow")

split_manifest = pd.concat([train_df, cal_df, shadow_df], ignore_index=True)
split_manifest_path = OUT / "P0_SPLIT_MANIFEST.csv"
split_manifest.to_csv(split_manifest_path, index=False)
split_receipt = {"schema_version": 1, "project": PROJECT, "stage": "P0_DEVELOPMENT_SPLIT", "status": "PASS", "development_source": RAW_DATASET, "source_partitions_enumerated": ["train", "val"], "test_partition_enumerated": False, "locked_test_used": False, "external_data_used": False, "split_seed": int(split_seed), "sample_counts": {name: int(len(rows)) for name, rows in assignments.items()}, "class_counts": {name: {cls: int(sum(s.class_name == cls for s in rows)) for cls in ("NORMAL", "PNEUMONIA")} for name, rows in assignments.items()}, "source_partition_counts": source_partition_counts, "development_inventory_class_counts": class_counts, "audit": split_audit, "split_manifest_sha256": sha256_file(split_manifest_path)}
split_receipt["receipt_sha256"] = sha256_json(split_receipt)
write_json(OUT / "P0_SPLIT_RECEIPT.json", split_receipt)

spec = runner.EXPERIMENTS[MODEL_ID]
if spec.model_id != MODEL_ID or spec.backbone_name != BACKBONE or spec.loss_name != "dynamic_focal":
    raise RuntimeError(f"Unexpected canonical M06 spec: {spec}")


def safe_metric(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else float("nan")


seed_results = []
for seed in SEEDS:
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_ds = runner.build_dataset(train_df, IMAGE_SIZE, BATCH_SIZE, True, seed, spec.batch_policy, 0.2, 2, 1, balanced_batches=False)
    cal_ds = runner.build_dataset(cal_df, IMAGE_SIZE, BATCH_SIZE, False, seed, "none", 0.2, 2, 1)

    model, backbone = runner.build_model(spec, IMAGE_SIZE, DROPOUT, "imagenet", seed, edge_filters=16, edge_max_gate=0.25)
    if backbone is None:
        raise RuntimeError("M06 canonical backbone unexpectedly missing")

    balance = runner.class_balance(train_df)
    if not np.isclose(float(balance["positive_focal_alpha"]), 0.5, atol=1e-12):
        raise RuntimeError(f"P0 TRAIN is not balanced: {balance}")
    loss = runner.FLSD53BinaryFocalLoss(alpha=0.5, threshold=0.20, hard_gamma=5.0, easy_gamma=3.0)

    backbone.trainable = False
    runner.compile_model(model, loss, HEAD_LR, WEIGHT_DECAY, use_ema=False)
    head_history = model.fit(train_ds, epochs=HEAD_EPOCHS, verbose=2)

    backbone.trainable = True
    runner.compile_model(model, loss, FINETUNE_LR, WEIGHT_DECAY, use_ema=False)
    finetune_history = model.fit(train_ds, epochs=FINETUNE_EPOCHS, verbose=2)

    raw_cal = runner.predict_dataset(model, cal_ds, verbose=0)
    cal_labels = cal_df["label"].to_numpy(dtype=int)
    calibrator = fit_platt_calibrator(raw_cal, cal_labels, seed=seed)
    calibrated_cal = apply_platt_calibrator(raw_cal, calibrator)
    threshold_summary = select_threshold(cal_labels, calibrated_cal)
    frozen_threshold = float(threshold_summary["threshold"])
    policy = {"seed": int(seed), "model_id": MODEL_ID, "backbone": BACKBONE, "calibrator": calibrator, "threshold": frozen_threshold, "threshold_source": "P0_CAL_ONLY", "weights_or_ensemble": "NONE_SINGLE_MODEL", "shadow_labels_used_for_policy": False}
    policy_sha = sha256_json(policy)

    shadow_predict_df = shadow_df.copy()
    shadow_predict_df["label"] = 0
    shadow_ds = runner.build_dataset(shadow_predict_df, IMAGE_SIZE, BATCH_SIZE, False, seed, "none", 0.2, 2, 1)
    raw_shadow = runner.predict_dataset(model, shadow_ds, verbose=0)
    calibrated_shadow = apply_platt_calibrator(raw_shadow, calibrator)
    shadow_labels = shadow_df["label"].to_numpy(dtype=int)
    shadow_metrics = runner.calculate_metrics(shadow_labels, calibrated_shadow, frozen_threshold)
    shadow_metrics["ece"] = expected_calibration_error(shadow_labels, calibrated_shadow, bins=10)
    shadow_metrics["threshold"] = frozen_threshold

    cal_metrics = runner.calculate_metrics(cal_labels, calibrated_cal, frozen_threshold)
    cal_metrics["ece"] = expected_calibration_error(cal_labels, calibrated_cal, bins=10)
    cal_metrics["threshold"] = frozen_threshold

    pred_frame = shadow_df[["sample_id", "class_name", "label", "patient_id"]].copy()
    pred_frame["raw_probability_pneumonia"] = raw_shadow
    pred_frame["calibrated_probability_pneumonia"] = calibrated_shadow
    pred_frame["prediction"] = (calibrated_shadow >= frozen_threshold).astype(int)
    pred_path = OUT / f"P0_SHADOW_PREDICTIONS_seed{seed}.csv"
    pred_frame.to_csv(pred_path, index=False)

    seed_result = {"seed": int(seed), "training": {"head_epochs": HEAD_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS, "head_final_loss": float(head_history.history["loss"][-1]), "finetune_final_loss": float(finetune_history.history["loss"][-1]), "validation_data_used": False, "cal_used_during_training": False, "shadow_used_during_training": False}, "policy": policy, "policy_sha256": policy_sha, "cal_metrics": cal_metrics, "shadow_metrics": shadow_metrics, "shadow_predictions_sha256": sha256_file(pred_path), "model_parameter_count": int(model.count_params())}
    seed_results.append(seed_result)
    write_json(OUT / f"P0_SEED_{seed}_RECEIPT.json", seed_result)
    tf.keras.backend.clear_session()
    del model, backbone, train_ds, cal_ds, shadow_ds
    gc.collect()

shadow_ba = [safe_metric(r["shadow_metrics"], "balanced_accuracy") for r in seed_results]
shadow_mcc = [safe_metric(r["shadow_metrics"], "mcc") for r in seed_results]
shadow_normal = [safe_metric(r["shadow_metrics"], "recall_normal") for r in seed_results]
shadow_pneu = [safe_metric(r["shadow_metrics"], "recall_pneumonia") for r in seed_results]
thresholds = [float(r["policy"]["threshold"]) for r in seed_results]

if min(shadow_ba) >= 0.97 and min(shadow_normal) >= 0.95 and min(shadow_pneu) >= 0.95 and max(thresholds) - min(thresholds) <= 0.15:
    verdict = "STRONGLY_PROMISING"
elif min(shadow_ba) >= 0.93 and min(shadow_normal) >= 0.90 and min(shadow_pneu) >= 0.90:
    verdict = "PROMISING"
else:
    verdict = "NOT_YET_PROMISING"

aggregate_metrics = {"shadow_balanced_accuracy_min": float(min(shadow_ba)), "shadow_balanced_accuracy_mean": float(np.mean(shadow_ba)), "shadow_mcc_min": float(min(shadow_mcc)), "shadow_recall_normal_min": float(min(shadow_normal)), "shadow_recall_pneumonia_min": float(min(shadow_pneu)), "threshold_spread": float(max(thresholds) - min(thresholds)), "worst_stress_balanced_accuracy": None}
aggregate_sha256 = sha256_json(aggregate_metrics)
next_action = "BUILD_FULL_V623_WITH_NEW_BLIND_HOLDOUT" if verdict in {"STRONGLY_PROMISING", "PROMISING"} else "REVISE_DEVELOPMENT_METHOD_BEFORE_FULL_V623"

receipt = {"schema_version": 1, "project": PROJECT, "stage": "P0_SINGLE_MODEL_DIRECTIONAL_SANDBOX", "status": "PASS", "verdict": verdict, "official_result": False, "development_only": True, "raw_dataset": RAW_DATASET, "source_partitions_enumerated": ["train", "val"], "test_partition_enumerated": False, "locked_test_used": False, "external_data_used": False, "chexpert_used": False, "nih_used": False, "model_id": MODEL_ID, "backbone": BACKBONE, "image_size": IMAGE_SIZE, "sample_counts": POOL_TARGETS, "split_receipt_sha256": sha256_file(OUT / "P0_SPLIT_RECEIPT.json"), "canonical_source_zip_sha256": SOURCE_ZIP_SHA256, "training_recipe": {"seeds": SEEDS, "head_epochs": HEAD_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS, "dropout": DROPOUT, "head_lr": HEAD_LR, "finetune_lr": FINETUNE_LR, "weight_decay": WEIGHT_DECAY, "loss": "canonical FLSD-53 dynamic focal alpha=0.5", "validation_data_used_in_fit": False, "calibration": "Platt on P0_CAL only", "threshold": "balanced dual-class threshold on P0_CAL only"}, "seed_results": seed_results, "aggregate_metrics": aggregate_metrics, "aggregate_sha256": aggregate_sha256, "next_action": next_action, "stability": {"min_shadow_balanced_accuracy": float(min(shadow_ba)), "mean_shadow_balanced_accuracy": float(np.mean(shadow_ba)), "min_shadow_mcc": float(min(shadow_mcc)), "min_shadow_recall_normal": float(min(shadow_normal)), "min_shadow_recall_pneumonia": float(min(shadow_pneu)), "threshold_min": float(min(thresholds)), "threshold_max": float(max(thresholds)), "threshold_spread": float(max(thresholds) - min(thresholds))}, "governance": {"may_inform_v623_design": True, "may_be_reported_as_official_test": False, "old_locked_624_reopened": False, "old_external_cohorts_reused": False, "no_shadow_feedback_within_this_run": True}, "total_seconds": round(time.perf_counter() - START, 3)}
receipt["receipt_sha256"] = sha256_json(receipt)
write_json(OUT / "P0_RECEIPT.json", receipt)
print(json.dumps({"status": "PASS", "stage": receipt["stage"], "verdict": verdict, "sample_counts": POOL_TARGETS, "min_shadow_balanced_accuracy": receipt["stability"]["min_shadow_balanced_accuracy"], "mean_shadow_balanced_accuracy": receipt["stability"]["mean_shadow_balanced_accuracy"], "min_shadow_recall_normal": receipt["stability"]["min_shadow_recall_normal"], "min_shadow_recall_pneumonia": receipt["stability"]["min_shadow_recall_pneumonia"], "threshold_spread": receipt["stability"]["threshold_spread"], "locked_test_used": False, "external_data_used": False, "test_partition_enumerated": False}, indent=2))
