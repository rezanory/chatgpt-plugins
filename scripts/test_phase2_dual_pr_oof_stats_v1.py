"""Synthetic fixtures test the algorithm only; real acceptance uses sealed OOF."""

import copy
import hashlib
import json
import pathlib
import sys

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, precision_score, recall_score

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import phase2_dual_pr_oof_stats_v1 as dual  # noqa: E402


def vec(pred=None):
    return {
        "identity_sequence_sha256": "a" * 64,
        "patient_id": np.array(["a", "a", "b", "c", "c", "d", "e", "e"], dtype=object),
        "label": np.array([0, 0, 0, 1, 1, 1, 0, 1]),
        "prediction": np.array(pred if pred is not None else [0, 0, 1, 1, 0, 1, 0, 1]),
        "score": np.array([0.1, 0.2, 0.6, 0.8, 0.4, 0.9, 0.3, 0.7]),
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
    }


@pytest.mark.parametrize("pred", [[0, 0, 1, 1, 0, 1, 0, 1], [0] * 8, [1] * 8])
def test_metrics_match_sklearn(pred):
    v = vec(pred)
    cm = confusion_matrix(v["label"], v["prediction"], labels=[0, 1]).ravel()
    got = dual.pr_metrics(cm)
    assert got["macro_precision"][0] == pytest.approx(
        precision_score(v["label"], pred, average="macro", zero_division=0)
    )
    assert got["macro_recall"][0] == pytest.approx(
        recall_score(v["label"], pred, average="macro", zero_division=0)
    )


def test_identical_predictions_no_effect():
    rows = dual.paired_pr(vec(), vec(), n_boot=200, seed=15)
    assert len(rows) == 2
    for r in rows:
        assert r["delta_candidate_minus_reference"] == 0
        assert r["ci95_low"] == r["ci95_high"] == 0
        assert r["p_raw_two_sided"] == 1
        assert r["n_patients"] == 5


def test_paired_swapping_reverses_delta():
    a, b = vec(), vec([0, 0, 0, 1, 1, 1, 0, 1])
    forward = dual.paired_pr(a, b, n_boot=200, seed=22)
    backward = dual.paired_pr(b, a, n_boot=200, seed=22)
    for f, b in zip(forward, backward, strict=True):
        assert f["delta_candidate_minus_reference"] == pytest.approx(
            -b["delta_candidate_minus_reference"]
        )
        assert f["ci95_low"] == pytest.approx(-b["ci95_high"])
        assert f["p_raw_two_sided"] == b["p_raw_two_sided"]


def test_bootstrap_matches_original_balanced_accuracy_estimator():
    a, b = vec(), vec([0, 0, 0, 1, 1, 1, 0, 1])
    original = dual.base.paired_patient_bootstrap_metrics(
        a["patient_id"],
        a["label"],
        a["prediction"],
        a["score"],
        b["prediction"],
        b["score"],
        ("balanced_accuracy",),
        n_boot=200,
        seed=45,
    )[0]
    updated = dual.paired_pr(a, b, n_boot=200, seed=45)[1]
    for key in (
        "reference",
        "candidate",
        "delta_candidate_minus_reference",
        "ci95_low",
        "ci95_high",
        "p_raw_two_sided",
        "n_bootstrap_valid",
    ):
        assert updated[key] == pytest.approx(original[key], abs=1e-12)


@pytest.mark.parametrize(
    "key,value",
    [
        ("identity_sequence_sha256", "b" * 64),
        ("patient_id", np.array(["z"] * 8)),
        ("label", np.array([1] * 8)),
        ("prediction", np.array([2] * 8)),
        ("locked_test_used_for_selection", True),
        ("external_used_for_selection", None),
    ],
)
def test_fail_closed_pair(key, value):
    candidate = copy.deepcopy(vec())
    candidate[key] = value
    with pytest.raises(ValueError):
        dual.paired_pr(vec(), candidate, n_boot=100)


def test_holm_joint_family_not_smaller_than_raw():
    rows = [{"metric": m, "p_raw_two_sided": p} for m in dual.METRICS for p in (0.001, 0.04, 0.7)]
    adjusted = dual.adjust(rows)
    assert len(adjusted) == 6
    assert all(r["p_holm_dual_global"] >= r["p_raw_two_sided"] for r in adjusted)
    assert adjusted[0]["p_holm_dual_global"] == pytest.approx(0.006)


def test_source_lock_detects_tamper(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    hashes = {}
    for i in range(40):
        f = root / (str(i) + ".json")
        f.write_text("{}", encoding="utf-8")
        hashes[f.name] = hashlib.sha256(b"{}").hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"source_run_id": dual.SOURCE_RUN, "files": hashes}), encoding="utf-8"
    )
    dual.verify_source(root, lock)
    (root / "0.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_FILE_SHA_MISMATCH"):
        dual.verify_source(root, lock)


def test_no_inference_or_training_api_in_addition():
    source = pathlib.Path(dual.__file__).read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert ".predict(" not in source
    assert "POST_RESULTS_EXPLORATORY" in source
