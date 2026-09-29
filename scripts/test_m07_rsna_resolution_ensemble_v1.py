from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "m07_rsna_resolution_ensemble_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ensemble", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(image_id, patient, label, s224, s320, s384):
    return {
        "image_id": image_id,
        "patient_id": patient,
        "age": 5,
        "label": label,
        "score_224": s224,
        "score_320": s320,
        "score_384": s384,
        "pred_224": int(s224 >= 0),
        "pred_320": int(s320 >= 0),
        "pred_384": int(s384 >= 0),
    }


def synthetic_rows():
    rows = []
    for i in range(120):
        label = i % 2
        direction = 1 if label else -1
        rows.append(
            row(
                f"img{i:03d}",
                f"p{i:03d}",
                label,
                direction * 1.0 + ((i % 5) - 2) * 0.15,
                direction * 0.9 + ((i % 7) - 3) * 0.12,
                direction * 1.1 + ((i % 9) - 4) * 0.10,
            )
        )
    return rows


def test_weight_grid_sums_to_one():
    m = load_module()
    grid = m.weight_grid()
    assert len(grid) > 10
    for weights in grid:
        assert abs(sum(weights) - 1.0) < 1e-12


def test_equal_weight_ensemble_is_deterministic():
    m = load_module()
    rows = synthetic_rows()
    m.add_label_free_ensembles(rows)
    first = rows[0]["score_equal"]
    m.add_label_free_ensembles(rows)
    assert rows[0]["score_equal"] == first


def test_crossfit_preserves_patient_isolation_and_constraint_training():
    m = load_module()
    report = m.crossfit(synthetic_rows(), optimize_weights=True)
    assert report["patient_leakage"] is False
    assert report["metrics"]["n"] == 120
    assert all(
        receipt["train_sensitivity"] >= 0.90
        for receipt in report["fold_receipts"]
    )


def test_source_has_no_retraining_or_kaggle_write():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "SaveKernel" not in source
    assert "model.fit(" not in source
    assert "enableGpu" not in source
    assert "retraining_performed" in source


def test_authorized_workflow_routes_ensemble_marker():
    text = (ROOT / ".github" / "workflows" / "control-plane-v3-query.yml").read_text(
        encoding="utf-8"
    )
    assert "m07_rsna_resolution_ensemble:" in text
    assert "M07_RSNA_RESOLUTION_ENSEMBLE_V1" in text
    assert "environment: cloudflare-production" in text
    assert "python scripts/m07_rsna_resolution_ensemble_v1.py" in text
