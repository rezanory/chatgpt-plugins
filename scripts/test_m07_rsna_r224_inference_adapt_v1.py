from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "m07_rsna_r224_tta_v1.py"
TAIL = ROOT / "scripts" / "m07_rsna_r224_inference_adapt_tail_v1.py"
DISPATCH = ROOT / "scripts" / "m07_rsna_r224_inference_adapt_dispatch_v1.py"
WORKFLOW = ROOT / ".github" / "workflows" / "pneumonia-v17-m07-continuation-20260914.yml"


def assembled_source() -> str:
    base = BASE.read_text(encoding="utf-8")
    marker = "integrity_receipt = {"
    assert marker in base
    prefix = base[: base.index(marker)].replace(
        'OUT = WORK / "M07_RSNA_PEDIATRIC_R224_TTA_V1"',
        'OUT = WORK / "M07_RSNA_PEDIATRIC_R224_INFERENCE_ADAPT_V1"',
    )
    return prefix + TAIL.read_text(encoding="utf-8")


def test_assembled_analysis_compiles() -> None:
    compile(assembled_source(), "assembled_m07_rsna_inference_adapt", "exec")


def test_lung_crop_is_fixed_label_free_inference() -> None:
    text = TAIL.read_text(encoding="utf-8")
    assert "tf.image.central_crop" in text
    assert "LUNG_CROP_FRACTIONS=(0.90,0.84)" in text
    assert '"labels_used":False' in text
    assert '"model_weights_changed":False' in text


def test_test_time_adaptation_is_episodic_and_nonpersistent() -> None:
    text = TAIL.read_text(encoding="utf-8")
    assert 'model.get_layer("head_norm")' in text
    assert "tf.GradientTape()" in text
    assert "variable.assign" not in text  # updates use local alias v, and anchors are restored
    assert "v.assign(a)" in text
    assert "v.assign_sub" in text
    assert '"persistent_model_weight_change":False' in text
    assert '"adaptation_labels_used":False' in text


def test_dispatch_and_workflow_route_exact_token() -> None:
    dispatch = DISPATCH.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    token = "M07_EXTERNAL_RSNA_R224_INFERENCE_ADAPT_V1"
    assert token in dispatch
    assert token in workflow
    assert "m07-rsna-r224-inference-adapt-v1:" in workflow
    assert "m07_rsna_r224_inference_adapt_dispatch_v1.py" in workflow
    assert '"enableGpu": True' in dispatch


def test_raw_external_replay_is_guarded() -> None:
    text = TAIL.read_text(encoding="utf-8")
    assert 'expected={"tn":400,"fp":178,"fn":37,"tp":484}' in text
    assert "R224_CANONICAL_REPLAY_DRIFT" in text
