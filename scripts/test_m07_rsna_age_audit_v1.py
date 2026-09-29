from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "m07_rsna_age_audit_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("age_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(age, label, pred, score, patient):
    return {
        "age": age,
        "label": label,
        "pred": pred,
        "score": score,
        "prob": 0.8 if pred else 0.2,
        "patient_id": patient,
    }


def test_age_band_selection_is_disjoint_and_complete():
    m = load_module()
    rows = [
        row(1, 0, 0, -1, "a"),
        row(5, 1, 1, 1, "b"),
        row(6, 0, 1, 0.3, "c"),
        row(9, 1, 1, 1.2, "d"),
        row(10, 0, 0, -0.8, "e"),
        row(18, 1, 0, -0.1, "f"),
    ]
    assert len(m.select(rows, 1, 5)) == 2
    assert len(m.select(rows, 6, 9)) == 2
    assert len(m.select(rows, 10, 18)) == 2
    assert len(m.select(rows, 1, 18)) == 6


def test_confusion_metrics_are_exact():
    m = load_module()
    rows = [
        row(3, 0, 0, -1.0, "a"),
        row(3, 0, 1, 0.2, "b"),
        row(4, 1, 0, -0.3, "c"),
        row(5, 1, 1, 1.0, "d"),
    ]
    result = m.metrics(rows)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["accuracy"] == 0.5
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["auroc"] == 0.75


def test_workflow_is_read_only_and_uses_frozen_source():
    text = (ROOT / ".github" / "workflows" / "m07-rsna-r224-age-audit-v1.yml").read_text(
        encoding="utf-8"
    )
    assert "cgp-control-plane-v3-action" not in text
    assert "CGP_READ_OIDC_TOKEN" in text
    assert "--bootstraps 1000" in text
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "training_performed" in source
    assert '"model_weights_changed": False' in source
    assert '"thresholds_changed": False' in source
    assert "SaveKernel" not in source
