from __future__ import annotations

import importlib.util
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TTA = ROOT / "scripts" / "m07_rsna_r224_tta_v1.py"
DISPATCH = ROOT / "scripts" / "m07_rsna_r224_tta_dispatch_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "pneumonia-v17-m07-continuation-20260914.yml"


def load_dispatch():
    spec = importlib.util.spec_from_file_location("tta_dispatch", DISPATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dispatch_contract_is_exact_frozen_r224():
    module = load_dispatch()
    assert set(module.CONTRACTS) == {"M07_EXTERNAL_RSNA_R224_TTA_V1"}
    contract = module.CONTRACTS["M07_EXTERNAL_RSNA_R224_TTA_V1"]
    assert contract == {
        "resolution": 224,
        "account_id": "kg-03",
        "owner": "rezanory",
        "state_handle": "rezanory/m07-final-5fold-fix2-d260914d",
        "batch_size": 12,
    }


def test_tta_family_is_bounded_and_geometry_preserving():
    source = TTA.read_text(encoding="utf-8")
    for name in (
        "canonical",
        "gamma_085",
        "gamma_115",
        "contrast_085",
        "contrast_115",
        "brightness_m10",
        "brightness_p10",
    ):
        assert f'"{name}"' in source
    block = source[source.index("def apply_tta_view"):source.index("def build_eval_dataset")]
    for forbidden in (
        "flip_left_right",
        "flip_up_down",
        "rot90",
        "central_crop",
        "crop_and_resize",
        "random_",
    ):
        assert forbidden not in block


def test_canonical_replay_is_hard_gate_and_raw_external_is_preserved():
    source = TTA.read_text(encoding="utf-8")
    assert 'expected_confusion = {"tn": 400, "fp": 178, "fn": 37, "tp": 484}' in source
    assert "R224_CANONICAL_REPLAY_DRIFT" in source
    assert '"raw_external_result_preserved": True' in source
    assert '"analysis_is_pure_external_validation": False' in source
    assert '"external_threshold_tuning": False' in source
    assert '"external_model_adaptation": False' in source
    assert '"training_performed": False' in source


def test_dispatch_verifies_tta_receipt_and_has_no_cpu_fallback():
    source = DISPATCH.read_text(encoding="utf-8")
    assert '"m07.external.rsna_pediatric.r224_tta.terminal.v1"' in source
    assert '"PASS_TTA_ANALYSIS"' in source
    assert 'root / "scripts" / "m07_rsna_r224_tta_v1.py"' in source
    assert "CPU_FALLBACK_GPU_QUOTA" not in source
    assert "M07_RSNA_TTA_GPU_QUOTA_EXHAUSTED" in source


def test_workflow_routes_tta_to_isolated_kg03_concurrency_lane():
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "M07_EXTERNAL_RSNA_R224_TTA_V1" in source
    assert "pneumonia-v17-m07-rsna-tta-kg03" in source
    assert "m07-rsna-r224-tta-v1:" in source
    assert "python scripts/m07_rsna_r224_tta_dispatch_v1.py" in source
    assert (chr(92) + "$" + "{{") not in source
    match = re.search(
        r"m07-rsna-r224-tta-v1:.*?(?=\n  [A-Za-z0-9_-]+:|\Z)",
        source,
        flags=re.S,
    )
    assert match is not None
    job = match.group(0)
    assert "timeout-minutes: 240" in job
    assert "M07_RSNA_R224_TTA_REPORT.json" in job
    assert "M07_RSNA_PEDIATRIC_R224_TTA_V1_COMPLETE.zip" in job
