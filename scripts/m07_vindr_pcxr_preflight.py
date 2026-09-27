from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working/M07_VINDR_PCXR_PREFLIGHT.json")
print("CGP_PHASE:M07_VINDR_PREFLIGHT_BOOT", flush=True)

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
LABEL_DICTIONARY_SOURCE = {
    "dataset": "VinDr-PCXR / PediCXR",
    "official_reference": (
        "Scientific Data 2023 PediCXR Table 3 + "
        "Kaggle pediatric-cxr-analysis-challenge data description"
    ),
}


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_vector(value):
    tokens = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", str(value))
    if not tokens:
        raise ValueError("EMPTY_LABEL_VECTOR")
    arr = np.array([float(v) for v in tokens], dtype=float)
    if not np.all(np.isin(arr, [0.0, 1.0])):
        raise ValueError("NON_BINARY_LABEL_VECTOR")
    return arr.astype(np.int8)


files = [p for p in ROOT.rglob("*") if p.is_file()]
csvs = [p for p in files if p.suffix.lower() == ".csv"]
dicoms = [p for p in files if p.suffix.lower() in {".dcm", ".dicom"}]
rasters = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

receipt = {
    "schema": "m07.external.vindr_pcxr.preflight.v3",
    "status": "BLOCKED",
    "gpu_used": False,
    "training_started": False,
    "inference_started": False,
    "input_root_exists": ROOT.exists(),
    "file_count": len(files),
    "csv_count": len(csvs),
    "dicom_count": len(dicoms),
    "raster_image_count": len(rasters),
    "csvs": [],
    "metadata_sha256": {},
    "dicom_probe": {},
    "label_map_proof": {
        "status": "NOT_PROVED",
        "source": LABEL_DICTIONARY_SOURCE,
    },
}

image_labels_test_path = None
annotations_test_path = None

for path in sorted(csvs):
    rec = {"path": path.relative_to(ROOT).as_posix()}
    try:
        df = pd.read_csv(path)
        cols = [str(c) for c in df.columns]
        low = {c.lower().strip(): c for c in cols}
        rec["rows"] = int(len(df))
        rec["columns"] = cols
        receipt["metadata_sha256"][rec["path"]] = sha256_file(path)

        if path.name.casefold() == "annotations_test.csv":
            annotations_test_path = path
            if not {"image_id", "class_name"}.issubset(set(low)):
                raise RuntimeError("ANNOTATIONS_TEST_SCHEMA_INVALID")
            c = low["class_name"]
            vc = df[c].astype(str).value_counts(dropna=False).head(200)
            rec["class_name_counts"] = {str(k): int(v) for k, v in vc.items()}

        if path.name.casefold() == "image_labels_test.csv":
            image_labels_test_path = path
            if not {"image_id", "labels"}.issubset(set(low)):
                raise RuntimeError("IMAGE_LABELS_TEST_SCHEMA_INVALID")
            iid_col, labels_col = low["image_id"], low["labels"]
            if df[iid_col].astype(str).duplicated().any():
                raise RuntimeError("DUPLICATE_TEST_IMAGE_ID")
            if len(df) != EXPECTED_TEST_N:
                raise RuntimeError(f"VINDR_TEST_METADATA_COUNT_MISMATCH={len(df)}")

            vectors = [parse_vector(v) for v in df[labels_col]]
            widths = sorted({len(v) for v in vectors})
            rec["label_vector_widths"] = widths
            if len(widths) != 1:
                raise RuntimeError(f"LABEL_VECTOR_WIDTHS={widths}")
            matrix = np.vstack(vectors)
            width = int(matrix.shape[1])
            matching_offsets = []
            for start in range(width - len(DIAGNOSIS_LABELS) + 1):
                observed = matrix[:, start:start + len(DIAGNOSIS_LABELS)].sum(axis=0)
                if np.array_equal(observed.astype(np.int64), EXPECTED_DIAGNOSIS_COUNTS):
                    matching_offsets.append(int(start))

            if len(matching_offsets) == 1:
                offset = matching_offsets[0]
                diagnosis = matrix[:, offset:offset + len(DIAGNOSIS_LABELS)]
                receipt["label_map_proof"] = {
                    "status": "PROVED_BY_ACTUAL_METADATA_UNIQUE_ALIGNMENT",
                    "source": LABEL_DICTIONARY_SOURCE,
                    "actual_metadata_file": rec["path"],
                    "actual_metadata_sha256": sha256_file(path),
                    "test_metadata_n": int(len(df)),
                    "label_vector_width": width,
                    "candidate_diagnosis_offsets_zero_based": matching_offsets,
                    "selected_diagnosis_offset_zero_based": offset,
                    "diagnosis_labels": DIAGNOSIS_LABELS,
                    "observed_diagnosis_counts": diagnosis.sum(axis=0).astype(int).tolist(),
                    "official_diagnosis_counts": EXPECTED_DIAGNOSIS_COUNTS.tolist(),
                    "resolved_indices_zero_based_within_diagnosis_block": {
                        "No finding": 0,
                        "Brocho-pneumonia": 2,
                        "Pneumonia": 6,
                        "Pleuro-pneumonia": 7,
                    },
                    "primary_mapping": {
                        "negative": "No finding",
                        "positive": "Pneumonia",
                        "other_diagnoses": "EXCLUDED_NOT_NORMAL",
                    },
                }
            else:
                receipt["label_map_proof"] = {
                    "status": "BLOCKED_DIAGNOSIS_OFFSET_NOT_UNIQUE",
                    "source": LABEL_DICTIONARY_SOURCE,
                    "actual_metadata_file": rec["path"],
                    "test_metadata_n": int(len(df)),
                    "label_vector_width": width,
                    "candidate_diagnosis_offsets_zero_based": matching_offsets,
                }

        receipt["csvs"].append(rec)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        receipt["csvs"].append(rec)

if dicoms:
    try:
        import pydicom

        tag_presence = {
            "PatientID": 0,
            "StudyInstanceUID": 0,
            "SOPInstanceUID": 0,
            "PatientAge": 0,
            "ViewPosition": 0,
        }
        views = {}
        patient_values = set()
        study_values = set()
        errors = []
        for path in sorted(dicoms):
            try:
                ds = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    specific_tags=list(tag_presence),
                )
                for tag in tag_presence:
                    value = str(getattr(ds, tag, "") or "").strip()
                    if value:
                        tag_presence[tag] += 1
                patient = str(getattr(ds, "PatientID", "") or "").strip()
                study = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
                if patient:
                    patient_values.add(patient)
                if study:
                    study_values.add(study)
                view = str(getattr(ds, "ViewPosition", "") or "").strip()
                if view:
                    views[view] = views.get(view, 0) + 1
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}"[:300])

        if tag_presence["PatientID"] == len(dicoms):
            cluster_policy = "PATIENT_CLUSTER"
        elif tag_presence["StudyInstanceUID"] == len(dicoms):
            cluster_policy = "STUDY_CLUSTER"
        else:
            cluster_policy = "EXAM_IMAGE"

        receipt["dicom_probe"] = {
            "sampled": len(dicoms),
            "full_dataset_header_scan": True,
            "tag_presence_counts": tag_presence,
            "patient_id_unique_nonempty": len(patient_values),
            "study_instance_uid_unique_nonempty": len(study_values),
            "bootstrap_candidate": cluster_policy,
            "view_positions": views,
            "read_error_count": len(errors),
            "read_errors": errors[:10],
        }
    except Exception as exc:
        receipt["dicom_probe"] = {
            "probe_error": f"{type(exc).__name__}: {exc}"[:1000]
        }

schema_ok = bool(image_labels_test_path and annotations_test_path)
mapping_ok = (
    receipt.get("label_map_proof", {}).get("status")
    == "PROVED_BY_ACTUAL_METADATA_UNIQUE_ALIGNMENT"
)
mounted = bool(files and dicoms and csvs)

receipt["schema_evidence"] = {
    "annotations_test_found": bool(annotations_test_path),
    "image_labels_test_found": bool(image_labels_test_path),
    "mapping_proved_from_actual_metadata": bool(mapping_ok),
}
receipt["status"] = (
    "PASS_DATASET_SCHEMA_AND_LABEL_MAP"
    if mounted and schema_ok and mapping_ok
    else "BLOCKED_DATASET_SCHEMA_OR_LABEL_MAP"
)
receipt["next_action"] = (
    "BUILD_EXTERNAL_MANIFEST_AND_RUN_PREINFERENCE_INTEGRITY"
    if receipt["status"] == "PASS_DATASET_SCHEMA_AND_LABEL_MAP"
    else "RESOLVE_COMPETITION_ACCESS_SCHEMA_OR_LABEL_MAP"
)

OUT.write_text(
    json.dumps(receipt, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("CGP_PHASE:M07_VINDR_PREFLIGHT_DONE", flush=True)
print(json.dumps(receipt, sort_keys=True), flush=True)
