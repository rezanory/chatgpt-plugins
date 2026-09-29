from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "m07_rsna_resolution_reconcile_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("reconcile", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metrics():
    return {
        "accuracy": 0.8,
        "balanced_accuracy": 0.81,
        "precision_positive": 0.73,
        "recall_positive": 0.92,
        "f1_positive": 0.81,
        "auroc": 0.93,
        "auprc_positive": 0.92,
        "tn": 400,
        "fp": 178,
        "fn": 37,
        "tp": 484,
    }


def test_extract_accepts_pure_scientific_receipt():
    m = load_module()
    receipt = {
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "resolution": 320,
        "expanded_n": 1099,
        "primary_n": 282,
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
        "external_calibration_fitting": False,
        "receipt_sha256": "abc",
        "source_state": "state",
    }
    report = {
        **{k: receipt[k] for k in (
            "status", "resolution", "training_performed", "hpo_performed",
            "external_threshold_tuning", "external_adaptation",
            "external_calibration_fitting",
        )},
        "expanded_pediatric_le18": {"metrics": metrics()},
        "primary_pediatric_lt10": {"metrics": metrics()},
    }
    row = m.extract(320, receipt, report)
    assert row["status"] == "SCIENTIFIC_RECEIPT_PASS"
    assert row["expanded"]["precision_positive"] == 0.73


def test_extract_rejects_external_tuning():
    m = load_module()
    receipt = {
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "resolution": 384,
        "expanded_n": 1099,
        "primary_n": 282,
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": True,
        "external_adaptation": False,
        "external_calibration_fitting": False,
    }
    report = {
        **{k: receipt[k] for k in (
            "status", "resolution", "training_performed", "hpo_performed",
            "external_threshold_tuning", "external_adaptation",
            "external_calibration_fitting",
        )},
        "expanded_pediatric_le18": {"metrics": metrics()},
        "primary_pediatric_lt10": {"metrics": metrics()},
    }
    try:
        m.extract(384, receipt, report)
    except RuntimeError as exc:
        assert "scientific purity" in str(exc)
    else:
        raise AssertionError("expected purity failure")


def test_source_is_read_only():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "SaveKernel" not in source
    assert "enableGpu" not in source
    assert "training_performed" in source


def test_authorized_workflow_routes_reconcile_marker():
    text = (ROOT / ".github" / "workflows" / "control-plane-v3-query.yml").read_text(
        encoding="utf-8"
    )
    assert "m07_rsna_resolution_reconcile:" in text
    assert "M07_RSNA_RESOLUTION_RECONCILE_V1" in text
    assert "environment: cloudflare-production" in text
    assert "python scripts/m07_rsna_resolution_reconcile_v1.py" in text
