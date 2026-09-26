from pathlib import Path
import json, re
import pandas as pd

ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working/M07_VINDR_PCXR_PREFLIGHT.json")
print("CGP_PHASE:M07_VINDR_PREFLIGHT_BOOT", flush=True)

files = [p for p in ROOT.rglob("*") if p.is_file()]
csvs = [p for p in files if p.suffix.lower() == ".csv"]
dicoms = [p for p in files if p.suffix.lower() in {".dcm", ".dicom"}]
rasters = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

receipt = {
    "schema": "m07.external.vindr_pcxr.preflight.v2",
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
    "dicom_probe": {},
}

pneumonia_evidence = False
no_finding_evidence = False
image_label_table = False
annotation_table = False

for p in sorted(csvs):
    rec = {"path": p.relative_to(ROOT).as_posix()}
    try:
        df = pd.read_csv(p)
        cols = [str(c) for c in df.columns]
        low = {c.lower().strip(): c for c in cols}
        rec["rows"] = int(len(df))
        rec["columns"] = cols

        if {"image_id", "class_name"}.issubset(set(low)):
            annotation_table = True
            c = low["class_name"]
            vc = df[c].astype(str).value_counts(dropna=False).head(100)
            counts = {str(k): int(v) for k, v in vc.items()}
            rec["class_name_counts"] = counts
            names = {str(k).strip().casefold() for k in counts}
            pneumonia_evidence = pneumonia_evidence or ("pneumonia" in names)
            no_finding_evidence = no_finding_evidence or ("no finding" in names)

        if {"image_id", "labels"}.issubset(set(low)):
            image_label_table = True
            c = low["labels"]
            vals = df[c].dropna().astype(str)
            lengths = {}
            token_domain = set()
            for value in vals.head(500):
                toks = [t for t in re.split(r"[\s,;|\[\]()]+", value.strip()) if t != ""]
                lengths[str(len(toks))] = lengths.get(str(len(toks)), 0) + 1
                token_domain.update(toks[:80])
            rec["label_vector_probe"] = {
                "sampled_rows": int(min(len(vals), 500)),
                "token_count_histogram": lengths,
                "token_domain_sample": sorted(token_domain)[:50],
            }

        for c in cols:
            lc = c.casefold()
            if "pneumonia" in lc:
                pneumonia_evidence = True
            if "no finding" in lc or "no_finding" in lc:
                no_finding_evidence = True

        receipt["csvs"].append(rec)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
        receipt["csvs"].append(rec)

if dicoms:
    try:
        import pydicom
        sample = dicoms[: min(200, len(dicoms))]
        tag_presence = {
            "PatientID": 0,
            "StudyInstanceUID": 0,
            "SOPInstanceUID": 0,
            "PatientAge": 0,
            "ViewPosition": 0,
        }
        views = {}
        errors = []
        for p in sample:
            try:
                ds = pydicom.dcmread(
                    str(p),
                    stop_before_pixels=True,
                    specific_tags=list(tag_presence),
                )
                for tag in tag_presence:
                    value = str(getattr(ds, tag, "")).strip()
                    if value:
                        tag_presence[tag] += 1
                view = str(getattr(ds, "ViewPosition", "")).strip()
                if view:
                    views[view] = views.get(view, 0) + 1
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:200])
        receipt["dicom_probe"] = {
            "sampled": len(sample),
            "tag_presence_counts": tag_presence,
            "view_positions": views,
            "read_errors": errors[:5],
        }
    except Exception as exc:
        receipt["dicom_probe"] = {"probe_error": f"{type(exc).__name__}: {exc}"[:500]}

receipt["schema_evidence"] = {
    "annotation_table_found": bool(annotation_table),
    "image_label_table_found": bool(image_label_table),
    "pneumonia_name_evidence": bool(pneumonia_evidence),
    "no_finding_name_evidence": bool(no_finding_evidence),
}

mounted = bool(files and dicoms and csvs)
schema_ok = bool(annotation_table and image_label_table)
receipt["status"] = "PASS_DATASET_SCHEMA" if mounted and schema_ok else "BLOCKED_DATASET_OR_SCHEMA"
receipt["next_action"] = (
    "BUILD_AND_VALIDATE_EXTERNAL_LABEL_MAP"
    if receipt["status"] == "PASS_DATASET_SCHEMA"
    else "RESOLVE_COMPETITION_ACCESS_OR_SCHEMA"
)
OUT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
print("CGP_PHASE:M07_VINDR_PREFLIGHT_DONE", flush=True)
print(json.dumps(receipt, sort_keys=True), flush=True)
