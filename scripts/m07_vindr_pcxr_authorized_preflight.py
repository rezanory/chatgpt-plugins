from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_TEST_N = 1397
DIAGNOSIS_LABELS = [
    "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease",
    "Bronchiolitis", "Situs inversus", "Pneumonia", "Pleuro-pneumonia",
    "Diaphragmatic hernia", "Tuberculosis", "Congenital emphysema", "CPAM",
    "Hyaline membrane disease", "Mediastinal tumor", "Lung tumor",
]
EXPECTED_DIAGNOSIS_COUNTS = np.array(
    [907, 174, 84, 77, 90, 2, 89, 0, 0, 1, 0, 1, 3, 1, 0],
    dtype=np.int64,
)
EXPECTED_PRIMARY = {"no_finding": 907, "pneumonia": 89, "n": 996}
OFFICIAL_REFERENCE = {
    "dataset": "VinDr-PCXR / PediCXR v1.0.0",
    "physionet_doi": "10.13026/k8qc-na36",
    "test_n": EXPECTED_TEST_N,
    "global_label_count": 15,
}

def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def parse_vector(value: object) -> np.ndarray:
    tokens = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", str(value))
    if not tokens:
        raise ValueError("EMPTY_LABEL_VECTOR")
    arr = np.array([float(x) for x in tokens], dtype=float)
    if not np.all(np.isin(arr, [0.0, 1.0])):
        raise ValueError("NON_BINARY_LABEL_VECTOR")
    return arr.astype(np.int8)

def fail(receipt: dict, output: Path, error: str) -> None:
    receipt["status"] = "BLOCKED"
    receipt["error"] = error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    raise SystemExit(error)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    receipt = {
        "schema": "m07.external.vindr_pcxr.authorized_preflight.v1",
        "status": "BLOCKED",
        "data_root": str(root),
        "official_reference": OFFICIAL_REFERENCE,
        "gpu_used": False,
        "training_started": False,
        "inference_started": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
    }

    if not root.is_dir():
        fail(receipt, output, "AUTHORIZED_DATA_ROOT_NOT_FOUND")

    candidates = []
    for labels_path in root.rglob("image_labels_test.csv"):
        annotations_path = labels_path.parent / "annotations_test.csv"
        if annotations_path.is_file():
            candidates.append((labels_path.resolve(), annotations_path.resolve()))
    if len(candidates) != 1:
        fail(receipt, output, f"AUTHORIZED_METADATA_ROOT_COUNT={len(candidates)}")

    labels_path, annotations_path = candidates[0]
    labels = pd.read_csv(labels_path)
    annotations = pd.read_csv(annotations_path)
    label_cols = {str(c).strip().lower(): c for c in labels.columns}
    ann_cols = {str(c).strip().lower(): c for c in annotations.columns}
    if not {"image_id", "labels"}.issubset(label_cols):
        fail(receipt, output, f"BAD_IMAGE_LABEL_COLUMNS={list(labels.columns)}")
    if not {"image_id", "class_name"}.issubset(ann_cols):
        fail(receipt, output, f"BAD_ANNOTATION_COLUMNS={list(annotations.columns)}")
    if len(labels) != EXPECTED_TEST_N:
        fail(receipt, output, f"TEST_METADATA_COUNT_MISMATCH={len(labels)}")

    iid_col = label_cols["image_id"]
    labels_col = label_cols["labels"]
    if labels[iid_col].astype(str).duplicated().any():
        fail(receipt, output, "DUPLICATE_TEST_IMAGE_ID")

    vectors = [parse_vector(v) for v in labels[labels_col]]
    widths = sorted({len(v) for v in vectors})
    if len(widths) != 1:
        fail(receipt, output, f"LABEL_VECTOR_WIDTHS={widths}")
    matrix = np.vstack(vectors)
    width = int(matrix.shape[1])

    matching_offsets = []
    for start in range(width - len(DIAGNOSIS_LABELS) + 1):
        observed = matrix[:, start:start + len(DIAGNOSIS_LABELS)].sum(axis=0).astype(np.int64)
        if np.array_equal(observed, EXPECTED_DIAGNOSIS_COUNTS):
            matching_offsets.append(start)
    if len(matching_offsets) != 1:
        fail(
            receipt,
            output,
            "DIAGNOSIS_OFFSET_NOT_UNIQUE="
            + json.dumps({"width": width, "offsets": matching_offsets}),
        )

    offset = int(matching_offsets[0])
    diagnosis = matrix[:, offset:offset + len(DIAGNOSIS_LABELS)]
    no_finding = diagnosis[:, 0].astype(bool)
    pneumonia = diagnosis[:, 6].astype(bool)
    primary_mask = np.logical_xor(no_finding, pneumonia)
    primary_n = int(primary_mask.sum())
    primary_no_finding = int(np.logical_and(primary_mask, no_finding).sum())
    primary_pneumonia = int(np.logical_and(primary_mask, pneumonia).sum())
    if {
        "no_finding": primary_no_finding,
        "pneumonia": primary_pneumonia,
        "n": primary_n,
    } != EXPECTED_PRIMARY:
        fail(
            receipt,
            output,
            "PRIMARY_COHORT_MISMATCH="
            + json.dumps(
                {
                    "no_finding": primary_no_finding,
                    "pneumonia": primary_pneumonia,
                    "n": primary_n,
                },
                sort_keys=True,
            ),
        )

    dicoms = [
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".dcm", ".dicom"}
    ]
    by_stem = {}
    for path in dicoms:
        by_stem.setdefault(path.stem, []).append(path)

    unresolved = []
    duplicate_stems = []
    selected = []
    for image_id in labels[iid_col].astype(str).str.strip():
        matches = by_stem.get(image_id, [])
        if not matches:
            unresolved.append(image_id)
        elif len(matches) > 1:
            duplicate_stems.append({"image_id": image_id, "count": len(matches)})
        else:
            selected.append(matches[0])
    if unresolved:
        fail(receipt, output, f"UNRESOLVED_DICOM_COUNT={len(unresolved)}")
    if duplicate_stems:
        fail(receipt, output, f"AMBIGUOUS_DICOM_STEM_COUNT={len(duplicate_stems)}")

    hashes = [sha256_file(p) for p in selected]
    duplicate_hash_count = len(hashes) - len(set(hashes))
    if duplicate_hash_count:
        fail(receipt, output, f"EXTERNAL_EXACT_DUPLICATE_DICOM_SHA={duplicate_hash_count}")

    identity = {
        "sampled": 0,
        "patient_id_present": 0,
        "study_instance_uid_present": 0,
        "sop_instance_uid_present": 0,
        "errors": [],
    }
    try:
        import pydicom
        for path in selected:
            try:
                ds = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    specific_tags=["PatientID", "StudyInstanceUID", "SOPInstanceUID"],
                )
                identity["sampled"] += 1
                if str(getattr(ds, "PatientID", "") or "").strip():
                    identity["patient_id_present"] += 1
                if str(getattr(ds, "StudyInstanceUID", "") or "").strip():
                    identity["study_instance_uid_present"] += 1
                if str(getattr(ds, "SOPInstanceUID", "") or "").strip():
                    identity["sop_instance_uid_present"] += 1
            except Exception as exc:
                identity["errors"].append(f"{type(exc).__name__}: {exc}"[:300])
                if len(identity["errors"]) >= 10:
                    break
    except Exception as exc:
        identity["probe_error"] = f"{type(exc).__name__}: {exc}"[:500]

    cluster_policy = "EXAM_IMAGE"
    if identity.get("sampled") == EXPECTED_TEST_N:
        if identity.get("patient_id_present") == EXPECTED_TEST_N:
            cluster_policy = "PATIENT_CLUSTER"
        elif identity.get("study_instance_uid_present") == EXPECTED_TEST_N:
            cluster_policy = "STUDY_CLUSTER"

    receipt.update({
        "status": "PASS_AUTHORIZED_DATA_PREFLIGHT",
        "metadata": {
            "image_labels_test": labels_path.relative_to(root).as_posix(),
            "annotations_test": annotations_path.relative_to(root).as_posix(),
            "image_labels_test_sha256": sha256_file(labels_path),
            "annotations_test_sha256": sha256_file(annotations_path),
            "test_metadata_n": int(len(labels)),
            "annotation_rows": int(len(annotations)),
            "label_vector_width": width,
            "selected_diagnosis_offset_zero_based": offset,
            "diagnosis_labels": DIAGNOSIS_LABELS,
            "observed_diagnosis_counts": diagnosis.sum(axis=0).astype(int).tolist(),
            "expected_diagnosis_counts": EXPECTED_DIAGNOSIS_COUNTS.astype(int).tolist(),
            "resolved_indices_zero_based_within_diagnosis_block": {
                "No finding": 0,
                "Pneumonia": 6,
                "Brocho-pneumonia": 2,
                "Pleuro-pneumonia": 7,
            },
        },
        "primary_endpoint": {
            "negative": "No finding",
            "positive": "Pneumonia",
            "other_diagnoses": "excluded from negative class",
            "n": primary_n,
            "no_finding": primary_no_finding,
            "pneumonia": primary_pneumonia,
        },
        "integrity": {
            "dicom_count_resolved": len(selected),
            "unresolved_image_ids": 0,
            "ambiguous_image_ids": 0,
            "external_exact_duplicate_dicom_sha": 0,
        },
        "identity_probe": identity,
        "recommended_ci_unit": cluster_policy,
        "next_action": "RUN_PREINFERENCE_INTERNAL_EXTERNAL_LEAKAGE_CHECK",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
