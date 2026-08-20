from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = "PNEUMONIA V6.2.3-P0R1"
STAGE = "SPLIT_DIAGNOSTIC_CPU_ONLY"
INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/V623_P0R1_DIAGNOSTIC")
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
TARGETS = {
    "NORMAL": {"shadow": 60, "cal": 60, "train": 240},
    "PNEUMONIA": {"shadow": 60, "cal": 60, "train": 240},
}
NEAR_DUP_HAMMING = 2
MAX_ATTEMPTS = 120
SEED_BASE = 623100


@dataclass(frozen=True)
class Sample:
    path: Path
    class_name: str
    label: int
    source_partition: str
    patient_id: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def patient_id_from_filename(name: str, class_name: str) -> str:
    stem = Path(name).stem
    if class_name == "PNEUMONIA":
        match = re.match(r"^(person\d+)(?:_|$)", stem, re.I)
        if match:
            return match.group(1).lower()
    for pattern in (r"^(NORMAL2-IM-\d+)", r"^(IM-\d+)"):
        match = re.match(pattern, stem, re.I)
        if match:
            return match.group(1).lower()
    # Fail closed to a single-image group rather than accidentally merging unrelated patients.
    return f"fallback:{class_name.lower()}:{stem.lower()}"


def looks_dev_root(root: Path) -> bool:
    return all(
        (root / partition / class_name).is_dir()
        for partition in ("train", "val")
        for class_name in ("NORMAL", "PNEUMONIA")
    )


walk_dirs = [INPUT]
for current, dirs, _files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d.lower() != "test" and d != "__MACOSX"]
    walk_dirs.extend(Path(current) / d for d in dirs)
roots = sorted(
    {p.resolve() for p in walk_dirs if looks_dev_root(p)},
    key=lambda path: (len(path.parts), str(path)),
)
roots = [p for p in roots if not any(p != q and q in p.parents for q in roots)]
if len(roots) != 1:
    raise RuntimeError(f"Expected one TRAIN/VAL development root, found: {roots}")
DATA_ROOT = roots[0]

samples: list[Sample] = []
for partition in ("train", "val"):
    for class_name, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
        for path in sorted((DATA_ROOT / partition / class_name).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                samples.append(
                    Sample(
                        path=path.resolve(),
                        class_name=class_name,
                        label=label,
                        source_partition=partition,
                        patient_id=patient_id_from_filename(path.name, class_name),
                    )
                )


def groups_for(class_name: str) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        if sample.class_name == class_name:
            groups.setdefault(sample.patient_id, []).append(sample)
    return groups


def exact_subset(groups: dict[str, list[Sample]], target: int, seed: int) -> list[str] | None:
    items = [(patient_id, len(rows)) for patient_id, rows in groups.items()]
    random.Random(seed).shuffle(items)
    parent: dict[int, tuple[int, str] | None] = {0: None}
    for patient_id, size in items:
        for total in sorted(list(parent), reverse=True):
            new_total = total + size
            if new_total > target or new_total in parent:
                continue
            parent[new_total] = (total, patient_id)
        if target in parent:
            break
    if target not in parent:
        return None
    chosen: list[str] = []
    cursor = target
    while cursor:
        link = parent[cursor]
        if link is None:
            raise RuntimeError("subset parent chain broken")
        cursor, patient_id = link
        chosen.append(patient_id)
    return chosen


def build_assignment(seed: int) -> tuple[dict[str, list[Sample]] | None, str | None]:
    assignment = {"train": [], "cal": [], "shadow": []}
    for class_index, class_name in enumerate(("NORMAL", "PNEUMONIA")):
        remaining = groups_for(class_name)
        for split_index, split_name in enumerate(("shadow", "cal", "train")):
            chosen = exact_subset(
                remaining,
                TARGETS[class_name][split_name],
                seed + class_index * 1009 + split_index * 101,
            )
            if chosen is None:
                return None, f"subset_fail:{class_name}:{split_name}"
            for patient_id in chosen:
                assignment[split_name].extend(remaining[patient_id])
            chosen_set = set(chosen)
            remaining = {
                patient_id: rows
                for patient_id, rows in remaining.items()
                if patient_id not in chosen_set
            }
    return assignment, None


print("CGP_PHASE:SPLIT_DIAGNOSTIC_HASHING", flush=True)
sha_cache: dict[str, str] = {}
dhash_cache: dict[str, int] = {}
for index, sample in enumerate(samples, start=1):
    key = str(sample.path)
    sha_cache[key] = sha256_file(sample.path)
    with Image.open(sample.path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    dhash_cache[key] = value
    if index % 500 == 0:
        print(f"CGP_DIAG_HASHED:{index}/{len(samples)}", flush=True)


def audit(assignment: dict[str, list[Sample]]) -> dict:
    patient_sets = {name: {row.patient_id for row in rows} for name, rows in assignment.items()}
    patient_overlap = []
    names = list(assignment)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(patient_sets[left] & patient_sets[right])
            if overlap:
                patient_overlap.append(
                    {
                        "a": left,
                        "b": right,
                        "count": len(overlap),
                        "examples": overlap[:10],
                    }
                )

    flat: list[tuple[str, Sample, str, int]] = []
    for split_name, rows in assignment.items():
        for row in rows:
            key = str(row.path)
            flat.append((split_name, row, sha_cache[key], dhash_cache[key]))

    by_sha: dict[str, list[tuple[str, Sample]]] = {}
    for split_name, row, digest, _dhash in flat:
        by_sha.setdefault(digest, []).append((split_name, row))
    exact_cross = []
    for digest, entries in by_sha.items():
        splits = sorted({entry[0] for entry in entries})
        if len(splits) > 1:
            exact_cross.append(
                {
                    "sha256": digest,
                    "splits": splits,
                    "count": len(entries),
                    "files": [entry[1].path.name for entry in entries[:6]],
                }
            )

    near_cross = []
    for index, left in enumerate(flat):
        for right in flat[index + 1 :]:
            if left[0] == right[0] or left[2] == right[2]:
                continue
            distance = (left[3] ^ right[3]).bit_count()
            if distance <= NEAR_DUP_HAMMING:
                near_cross.append(
                    {
                        "a": left[0],
                        "b": right[0],
                        "distance": distance,
                        "a_name": left[1].path.name,
                        "b_name": right[1].path.name,
                        "a_patient": left[1].patient_id,
                        "b_patient": right[1].patient_id,
                    }
                )
                if len(near_cross) >= 100:
                    break
        if len(near_cross) >= 100:
            break

    return {
        "patient_overlap": patient_overlap,
        "exact_duplicate_cross_split": exact_cross,
        "near_duplicate_cross_split": near_cross,
        "near_duplicate_hamming_threshold": NEAR_DUP_HAMMING,
        "pass": not patient_overlap and not exact_cross and not near_cross,
    }


group_summary = {}
for class_name in ("NORMAL", "PNEUMONIA"):
    groups = groups_for(class_name)
    histogram = Counter(len(rows) for rows in groups.values())
    group_summary[class_name] = {
        "samples": sum(len(rows) for rows in groups.values()),
        "patient_groups": len(groups),
        "group_size_histogram": dict(sorted(histogram.items())),
        "max_group_size": max(histogram) if histogram else 0,
        "fallback_patient_groups": sum(patient_id.startswith("fallback:") for patient_id in groups),
    }

print("CGP_PHASE:SPLIT_DIAGNOSTIC_SEARCH", flush=True)
reason_counts: Counter[str] = Counter()
audit_failure_counts: Counter[str] = Counter()
first_failures: list[dict] = []
winning_assignment = None
winning_audit = None
winning_seed = None
for attempt in range(MAX_ATTEMPTS):
    seed = SEED_BASE + attempt * 97
    assignment, reason = build_assignment(seed)
    if assignment is None:
        reason_counts[reason or "subset_fail:unknown"] += 1
        continue
    counts = {name: len(rows) for name, rows in assignment.items()}
    if counts != {"train": 480, "cal": 120, "shadow": 120}:
        reason_counts[f"count_mismatch:{counts}"] += 1
        continue
    result = audit(assignment)
    if result["pass"]:
        winning_assignment = assignment
        winning_audit = result
        winning_seed = seed
        break
    if result["patient_overlap"]:
        audit_failure_counts["patient_overlap"] += 1
    if result["exact_duplicate_cross_split"]:
        audit_failure_counts["exact_duplicate_cross_split"] += 1
    if result["near_duplicate_cross_split"]:
        audit_failure_counts["near_duplicate_cross_split"] += 1
    if len(first_failures) < 5:
        first_failures.append({"seed": seed, "audit": result})

receipt = {
    "schema_version": 1,
    "project": PROJECT,
    "stage": STAGE,
    "status": "PASS" if winning_assignment is not None else "FAIL",
    "gpu_used": False,
    "training_started": False,
    "development_only": True,
    "source_partitions_enumerated": ["train", "val"],
    "test_partition_enumerated": False,
    "locked_test_used": False,
    "external_data_used": False,
    "development_root": str(DATA_ROOT),
    "inventory_samples": len(samples),
    "group_summary": group_summary,
    "targets": TARGETS,
    "attempts": MAX_ATTEMPTS,
    "winning_seed": winning_seed,
    "failure_reason_counts": dict(reason_counts),
    "audit_failure_counts": dict(audit_failure_counts),
    "first_failures": first_failures,
}
if winning_assignment is not None:
    receipt["sample_counts"] = {name: len(rows) for name, rows in winning_assignment.items()}
    receipt["class_counts"] = {
        split_name: {
            class_name: sum(row.class_name == class_name for row in rows)
            for class_name in ("NORMAL", "PNEUMONIA")
        }
        for split_name, rows in winning_assignment.items()
    }
    receipt["audit"] = winning_audit

write_json(OUT / "SPLIT_DIAGNOSTIC.json", receipt)
print("CGP_PHASE:SPLIT_DIAGNOSTIC_DONE", flush=True)
print(json.dumps(receipt, sort_keys=True), flush=True)
