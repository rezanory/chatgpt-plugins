from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "m07_rsna_r224_crossfit_calibration_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("calibration", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(patient, label, score):
    return {
        "age": 4,
        "patient_id": patient,
        "label": label,
        "pred": int(score >= 0),
        "score": float(score),
        "prob": 1.0 / (1.0 + __import__("math").exp(-float(score))),
    }


def synthetic_rows():
    rows = []
    for i in range(60):
        label = i % 2
        base = 1.2 if label else -0.4
        rows.append(row(f"p{i:03d}", label, base + (i % 7) * 0.03))
    return rows


def test_patient_fold_is_deterministic():
    m = load_module()
    assert m.patient_fold("abc") == m.patient_fold("abc")
    assert 0 <= m.patient_fold("abc") < 5


def test_threshold_selection_improves_shifted_operating_point():
    m = load_module()
    rows = synthetic_rows()
    baseline = m.confusion(rows, 0.0)
    threshold, selected = m.choose_balanced_threshold(rows)
    assert threshold > 0
    assert selected["balanced_accuracy"] >= baseline["balanced_accuracy"]


def test_platt_is_monotonic_and_bounded():
    m = load_module()
    rows = synthetic_rows()
    params = m.fit_platt(rows)
    p0 = m.platt_probability(-1.0, params)
    p1 = m.platt_probability(1.0, params)
    assert 0 < p0 < p1 < 1


def test_crossfit_has_no_patient_leakage_and_preserves_source_rows():
    m = load_module()
    rows = synthetic_rows()
    audit = m.validate_folds(rows)
    assert sum(audit["row_counts"].values()) == len(rows)
    report = m.crossfit(rows)
    assert report["patient_leakage"] is False
    assert report["model_weights_changed"] is False
    assert report["training_performed"] is False
    assert report["raw_external_result_preserved"] is True
    assert report["analysis_is_pure_external_validation"] is False


def test_source_contains_no_model_training_or_weight_change():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "model.fit(" not in source
    assert "load_weights" not in source
    assert "SaveKernel" not in source
    assert "model_weights_changed" in source


def test_authorized_workflow_routes_crossfit_calibration_marker():
    text = (ROOT / ".github" / "workflows" / "control-plane-v3-query.yml").read_text(
        encoding="utf-8"
    )
    assert "m07_rsna_r224_crossfit_calibration:" in text
    assert "M07_RSNA_R224_CROSSFIT_CALIBRATION_V1" in text
    assert "environment: cloudflare-production" in text
    assert "python scripts/m07_rsna_r224_crossfit_calibration_v1.py" in text
