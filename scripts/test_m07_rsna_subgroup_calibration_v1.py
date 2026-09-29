from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "m07_rsna_subgroup_calibration_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("subgroup", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(i, age, view, label, score):
    return {
        "image_id": f"img{i}",
        "patient_id": f"p{i}",
        "age": age,
        "sex": "M" if i % 2 else "F",
        "view": view,
        "label": label,
        "score": score,
        "pred": int(score >= 0),
        "prob": 0.5,
    }


def synthetic_rows():
    rows = []
    for i in range(120):
        label = i % 2
        age = 3 if i % 3 == 0 else (8 if i % 3 == 1 else 14)
        view = "AP" if i % 4 < 2 else "PA"
        base = 1.0 if label else -0.2
        shift = 0.35 if view == "AP" else -0.10
        rows.append(row(i, age, view, label, base + shift + (i % 7) * 0.02))
    return rows


def test_age_band_boundaries():
    m = load_module()
    assert m.age_band(1) == "age_1_5"
    assert m.age_band(5) == "age_1_5"
    assert m.age_band(6) == "age_6_9"
    assert m.age_band(9) == "age_6_9"
    assert m.age_band(10) == "age_10_18"
    assert m.age_band(18) == "age_10_18"


def test_optimizer_respects_global_sensitivity_floor():
    m = load_module()
    result = m.optimize_group_thresholds(synthetic_rows(), "view", 0.90)
    assert result["metrics"]["sensitivity"] >= 0.90
    assert set(result["thresholds"]) == {"AP", "PA"}


def test_crossfit_is_patient_isolated():
    m = load_module()
    result = m.crossfit(synthetic_rows(), "age", 0.90)
    assert result["patient_leakage"] is False
    assert result["metrics"]["n"] == 120
    assert all(
        receipt["train_metrics"]["sensitivity"] >= 0.90
        for receipt in result["fold_receipts"]
    )


def test_source_has_no_retraining_or_kaggle_write():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "SaveKernel" not in source
    assert "model.fit(" not in source
    assert "enableGpu" not in source
    assert "retraining_performed" in source


def test_authorized_workflow_routes_subgroup_calibration_marker():
    text = (ROOT / ".github" / "workflows" / "control-plane-v3-query.yml").read_text(
        encoding="utf-8"
    )
    assert "m07_rsna_subgroup_calibration:" in text
    assert "M07_RSNA_SUBGROUP_CALIBRATION_V1" in text
    assert "environment: cloudflare-production" in text
    assert "python scripts/m07_rsna_subgroup_calibration_v1.py" in text
