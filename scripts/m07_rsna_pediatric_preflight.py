from __future__ import annotations

import base64
import csv
import hashlib
import json
import pathlib
import zlib
from collections import Counter, defaultdict

INPUT = pathlib.Path("/kaggle/input").resolve()
OUT = pathlib.Path("/kaggle/working").resolve()
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_MANIFEST_SHA256 = "3450a101e626a09535d98e5aa9ca52dbf64ded584c2952f7613e3c22981cf5cd"
EXPECTED_COUNTS = {
    "expanded_images": 1099,
    "expanded_patients": 553,
    "expanded_negative": 578,
    "expanded_positive": 521,
    "primary_images": 282,
    "primary_patients": 158,
    "primary_negative": 146,
    "primary_positive": 136,
}
DATASET_REF = "nih-chest-xrays/data"

try:
    payload = zlib.decompress(base64.b64decode(EMBEDDED_MANIFEST_B64.encode("ascii")))
except Exception as exc:
    raise RuntimeError(f"RSNA_MANIFEST_DECODE_FAILED={type(exc).__name__}") from exc

manifest = json.loads(payload.decode("utf-8"))
canonical = json.dumps(
    manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")
manifest_sha = hashlib.sha256(canonical).hexdigest()
if manifest_sha != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("RSNA_MANIFEST_SHA_MISMATCH")
if not isinstance(manifest, list) or len(manifest) != EXPECTED_COUNTS["expanded_images"]:
    raise RuntimeError("RSNA_MANIFEST_CARDINALITY_MISMATCH")

image_ids = [str(row["img_id"]) for row in manifest]
if len(set(image_ids)) != len(image_ids):
    raise RuntimeError("RSNA_MANIFEST_DUPLICATE_IMAGE_ID")

labels = Counter(int(row["label"]) for row in manifest)
primary_rows = [row for row in manifest if bool(row["primary_peds_lt10"])]
primary_labels = Counter(int(row["label"]) for row in primary_rows)
counts = {
    "expanded_images": len(manifest),
    "expanded_patients": len({int(row["patient_id"]) for row in manifest}),
    "expanded_negative": labels[0],
    "expanded_positive": labels[1],
    "primary_images": len(primary_rows),
    "primary_patients": len({int(row["patient_id"]) for row in primary_rows}),
    "primary_negative": primary_labels[0],
    "primary_positive": primary_labels[1],
}
if counts != EXPECTED_COUNTS:
    raise RuntimeError("RSNA_MANIFEST_COUNTS_MISMATCH=" + json.dumps(counts, sort_keys=True))

meta_files = list(INPUT.rglob("Data_Entry_2017.csv"))
if len(meta_files) != 1:
    raise RuntimeError(f"RSNA_NIH_METADATA_COUNT={len(meta_files)}")

required = set(image_ids)
meta = {}
with meta_files[0].open("r", encoding="utf-8-sig", newline="") as stream:
    reader = csv.DictReader(stream)
    for row in reader:
        image_id = str(row.get("Image Index") or "")
        if image_id in required:
            meta[image_id] = {
                "age": int(float(row["Patient Age"])),
                "patient_id": int(float(row["Patient ID"])),
                "view": str(row["View Position"]),
                "sex": str(row["Patient Gender"]),
            }
if set(meta) != required:
    raise RuntimeError(f"RSNA_NIH_METADATA_MATCH_COUNT={len(meta)}")

for row in manifest:
    actual = meta[row["img_id"]]
    expected = {
        "age": int(row["age"]),
        "patient_id": int(row["patient_id"]),
        "view": str(row["view"]),
        "sex": str(row["sex"]),
    }
    if actual != expected:
        raise RuntimeError("RSNA_NIH_METADATA_DRIFT=" + str(row["img_id"]))

paths = defaultdict(list)
for path in INPUT.rglob("*.png"):
    if path.name in required:
        paths[path.name].append(path)
bad = {name: len(items) for name, items in paths.items() if len(items) != 1}
missing = sorted(required - set(paths))
if missing or bad or len(paths) != len(required):
    raise RuntimeError(
        "RSNA_IMAGE_RESOLUTION_MISMATCH="
        + json.dumps(
            {"resolved": len(paths), "missing": missing[:25], "bad": bad},
            sort_keys=True,
        )
    )

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

image_hashes = {}
hash_to_ids = defaultdict(list)
for image_id in sorted(required):
    digest = sha256_file(paths[image_id][0])
    image_hashes[image_id] = digest
    hash_to_ids[digest].append(image_id)
duplicates = {digest: ids for digest, ids in hash_to_ids.items() if len(ids) > 1}
if duplicates:
    raise RuntimeError("RSNA_EXTERNAL_DUPLICATE_IMAGE_SHA=" + json.dumps(duplicates, sort_keys=True))

receipt = {
    "schema": "m07.external.rsna_pediatric.preflight.v1",
    "status": "PASS_RSNA_PEDIATRIC_SCHEMA",
    "dataset_ref": DATASET_REF,
    "manifest_sha256": manifest_sha,
    "counts": counts,
    "metadata_file": str(meta_files[0]),
    "resolved_image_count": len(paths),
    "unique_image_sha256_count": len(hash_to_ids),
    "duplicate_image_sha256_count": 0,
    "training_performed": False,
    "hpo_performed": False,
    "inference_started": False,
    "external_threshold_tuning": False,
    "external_adaptation": False,
}
body = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
receipt["receipt_sha256"] = hashlib.sha256(body).hexdigest()
(OUT / "M07_RSNA_PEDIATRIC_PREFLIGHT.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8"
)
print("M07_RSNA_PEDIATRIC_PREFLIGHT", json.dumps(receipt, sort_keys=True), flush=True)
