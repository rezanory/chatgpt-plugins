from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "m07_rsna_fold_consensus_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("consensus", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(i, label, v224, v320, v384):
    return {
        "image_id": f"img{i}",
        "patient_id": f"p{i}",
        "age": 5,
        "label": label,
        "votes_224": v224,
        "votes_320": v320,
        "votes_384": v384,
        "votes_total": v224 + v320 + v384,
    }


def synthetic_rows():
    rows = []
    for i in range(100):
        label = i % 2
        if label:
            votes = (5, 4, 5) if i % 6 else (4, 4, 4)
        else:
            votes = (1, 2, 1) if i % 5 else (3, 2, 2)
        rows.append(row(i, label, *votes))
    return rows


def test_rule_catalog_contains_expected_families():
    m = load_module()
    families = {rule["family"] for rule in m.rule_catalog()}
    assert "total_15" in families
    assert "at_least_2_resolutions" in families
    assert "all_3_resolutions" in families
    assert "single_224" in families


def test_prediction_rules_are_deterministic():
    m = load_module()
    r = row(1, 1, 4, 3, 5)
    assert m.predict(r, {"family": "total_15", "k": 8}) == 1
    assert m.predict(r, {"family": "all_3_resolutions", "k": 4}) == 0
    assert m.predict(r, {"family": "at_least_2_resolutions", "k": 4}) == 1


def test_crossfit_is_patient_isolated():
    m = load_module()
    result = m.crossfit(synthetic_rows())
    assert result["patient_leakage"] is False
    assert result["metrics"]["n"] == 100
    assert all(
        receipt["train_sensitivity"] >= 0.90
        for receipt in result["fold_receipts"]
    )


def test_source_has_no_training_or_kaggle_write():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "SaveKernel" not in source
    assert "model.fit(" not in source
    assert "enableGpu" not in source
    assert "retraining_performed" in source


def test_authorized_workflow_routes_fold_consensus_marker():
    text = (ROOT / ".github" / "workflows" / "control-plane-v3-query.yml").read_text(
        encoding="utf-8"
    )
    assert "m07_rsna_fold_consensus:" in text
    assert "M07_RSNA_FOLD_CONSENSUS_V1" in text
    assert "environment: cloudflare-production" in text
    assert "python scripts/m07_rsna_fold_consensus_v1.py" in text
