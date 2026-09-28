from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "scripts" / "m07_rsna_pediatric_external_v1.py"
DISPATCH = ROOT / "scripts" / "m07_rsna_pediatric_external_dispatch_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "pneumonia-v17-m07-continuation-20260914.yml"
MANIFEST = ROOT / "evidence" / "m07_external" / "rsna_pediatric_v1" / "manifest.json"
RECEIPT = ROOT / "evidence" / "m07_external" / "rsna_pediatric_v1" / "manifest_receipt.json"

EXPECTED_MANIFEST_SHA256 = "3450a101e626a09535d98e5aa9ca52dbf64ded584c2952f7613e3c22981cf5cd"
TOKENS = (
    "M07_EXTERNAL_RSNA_PEDS_R224_V1",
    "M07_EXTERNAL_RSNA_PEDS_R320_V1",
    "M07_EXTERNAL_RSNA_PEDS_R384_V1",
)


def _canonical_manifest_sha() -> str:
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def test_rsna_manifest_is_frozen_before_inference():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN_PREINFERENCE"
    assert _canonical_manifest_sha() == EXPECTED_MANIFEST_SHA256
    assert receipt["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert receipt["counts"] == {
        "expanded_images": 1099,
        "expanded_patients": 553,
        "expanded_negative": 578,
        "expanded_positive": 521,
        "primary_images": 282,
        "primary_patients": 158,
        "primary_negative": 146,
        "primary_positive": 136,
    }
    assert receipt["selection"]["primary_age_lt"] == 10
    assert receipt["selection"]["age_min"] == 1
    assert receipt["selection"]["expanded_age_max"] == 18


def test_rsna_external_script_is_inference_only_and_not_vindr_loader():
    text = EXTERNAL.read_text(encoding="utf-8")
    assert "EXPECTED_MANIFEST_SHA256 = " in text
    assert EXPECTED_MANIFEST_SHA256 in text
    assert 'EXPECTED_PRIMARY_N = 282' in text
    assert 'EXPECTED_EXPANDED_N = 1099' in text
    assert '"training_performed": False' in text
    assert '"hpo_performed": False' in text
    assert '"external_threshold_tuning": False' in text
    assert '"external_adaptation": False' in text
    assert '"calibration_fitting_on_external": False' in text
    assert "model.fit(" not in text
    assert ".fit(" not in text
    assert "tune_threshold" not in text
    assert "image_labels_test.csv" not in text
    assert "annotations_test.csv" not in text
    assert "dicom_to_png" not in text
    assert "EXPECTED_DIAGNOSIS_COUNTS" not in text
    assert "Pneumonia Detection Challenge" in text
    assert "endpoint_semantics_differ" in text


def test_rsna_dispatch_uses_frozen_states_and_no_competition_source():
    text = DISPATCH.read_text(encoding="utf-8")
    for token in TOKENS:
        assert token in text
    for handle in (
        "rezanory/m07-final-5fold-fix2-d260914d",
        "trickermark/m07-gate-r320-state-v1-7",
        "trickermark/m07-gate-r384-state-v1-7",
    ):
        assert handle in text
    assert '"datasetDataSources": [state_handle, DATASET_REF]' in text
    assert '"competitionDataSources": []' in text
    assert '"enableGpu": True' in text
    assert '"external_threshold_tuning": False' in text
    assert '"calibration_fitting_on_external": False' in text
    assert EXPECTED_MANIFEST_SHA256 in text


def test_workflow_routes_exact_rsna_external_tokens():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "m07-rsna-pediatric-external-resolution-v1:" in text
    for token in TOKENS:
        assert token in text
    assert "scripts/m07_rsna_pediatric_external_dispatch_v1.py" in text
