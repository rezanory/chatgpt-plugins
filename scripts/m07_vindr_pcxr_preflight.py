from pathlib import Path
import hashlib
import json

import pandas as pd

OUT = Path("/kaggle/working/M07_VINDR_PCXR_PREFLIGHT.json")
ROOT = Path("/kaggle/input")
print("CGP_PHASE:M07_VINDR_PREFLIGHT_BOOT", flush=True)

files = [p for p in ROOT.rglob("*") if p.is_file()]
csvs = [p for p in files if p.suffix.lower() == ".csv"]
dicoms = [p for p in files if p.suffix.lower() in {".dcm", ".dicom"}]
images = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

receipt = {
    "schema": "m07.external.vindr_pcxr.preflight.v1",
    "status": "BLOCKED",
    "gpu_used": False,
    "training_started": False,
    "inference_started": False,
    "input_root_exists": ROOT.exists(),
    "file_count": len(files),
    "csv_count": len(csvs),
    "dicom_count": len(dicoms),
    "raster_image_count": len(images),
    "csvs": [],
    "dicom_identity": {},
}

for path in sorted(csvs):
    try:
        df = pd.read_csv(path)
        cols = [str(c) for c in df.columns]
        pneumonia_cols = [c for c in cols if "pneumonia" in c.lower()]
        no_finding_cols = [
            c for c in cols
            if c.lower().strip().replace("_", " ") in {"no finding", "no findings"}
        ]
        patient_cols = [
            c for c in cols
            if c.lower().strip().replace("_", " ") in {"patient id", "patientid", "patient"}
        ]
        row = {
            "path": path.relative_to(ROOT).as_posix(),
            "rows": int(len(df)),
            "columns": cols,
            "pneumonia_columns": pneumonia_cols,
            "no_finding_columns": no_finding_cols,
            "patient_id_columns": patient_cols,
        }
        for col in pneumonia_cols + no_finding_cols:
            try:
                counts = df[col].value_counts(dropna=False).head(10).to_dict()
                row.setdefault("value_counts", {})[col] = {
                    str(key): int(value) for key, value in counts.items()
                }
            except Exception:
                pass
        receipt["csvs"].append(row)
    except Exception as exc:
        receipt["csvs"].append({
            "path": path.relative_to(ROOT).as_posix(),
            "error": repr(exc)[:500],
        })

if dicoms:
    try:
        import pydicom

        patient_hashes = []
        study_hashes = set()
        sop_hashes = set()
        views = {}
        ages_present = 0
        errors = []
        sample = dicoms[: min(len(dicoms), 200)]

        for path in sample:
            try:
                ds = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    specific_tags=[
                        "PatientID",
                        "StudyInstanceUID",
                        "SOPInstanceUID",
                        "PatientAge",
                        "ViewPosition",
                    ],
                )
                patient_id = str(getattr(ds, "PatientID", "")).strip()
                if patient_id:
                    patient_hashes.append(hashlib.sha256(patient_id.encode()).hexdigest())

                study_uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
                if study_uid:
                    study_hashes.add(hashlib.sha256(study_uid.encode()).hexdigest())

                sop_uid = str(getattr(ds, "SOPInstanceUID", "")).strip()
                if sop_uid:
                    sop_hashes.add(hashlib.sha256(sop_uid.encode()).hexdigest())

                if str(getattr(ds, "PatientAge", "")).strip():
                    ages_present += 1

                view = str(getattr(ds, "ViewPosition", "")).strip()
                if view:
                    views[view] = views.get(view, 0) + 1
            except Exception as exc:
                errors.append(repr(exc)[:200])

        receipt["dicom_identity"] = {
            "sampled": len(sample),
            "patient_id_present": len(patient_hashes),
            "unique_patient_ids_hashed": len(set(patient_hashes)),
            "study_uid_present_unique_hashed": len(study_hashes),
            "sop_uid_present_unique_hashed": len(sop_hashes),
            "patient_age_present": ages_present,
            "view_positions": views,
            "read_errors": errors[:5],
        }
    except Exception as exc:
        receipt["dicom_identity"] = {"probe_error": repr(exc)[:500]}

label_tables = [
    row for row in receipt["csvs"]
    if row.get("pneumonia_columns") and row.get("no_finding_columns")
]
patient_in_csv = any(row.get("patient_id_columns") for row in receipt["csvs"])
patient_in_dicom = int(receipt.get("dicom_identity", {}).get("patient_id_present", 0) or 0) > 0

receipt["label_tables_found"] = len(label_tables)
receipt["patient_identity_available"] = bool(patient_in_csv or patient_in_dicom)
receipt["status"] = (
    "PASS_DATASET_SCHEMA"
    if files and label_tables
    else "BLOCKED_DATASET_OR_LABEL_SCHEMA"
)
receipt["next_action"] = (
    "BUILD_EXTERNAL_MANIFEST"
    if receipt["status"] == "PASS_DATASET_SCHEMA" and receipt["patient_identity_available"]
    else (
        "RESOLVE_PATIENT_IDENTITY"
        if receipt["status"] == "PASS_DATASET_SCHEMA"
        else "RESOLVE_COMPETITION_ACCESS_OR_SCHEMA"
    )
)

OUT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
print("CGP_PHASE:M07_VINDR_PREFLIGHT_DONE", flush=True)
print(json.dumps(receipt, sort_keys=True), flush=True)
