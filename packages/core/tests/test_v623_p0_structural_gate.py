from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DIAGNOSTIC = ROOT / "scripts" / "v623_p0_split_diagnostic.py"
TEMPLATE = ROOT / "scripts" / "v623_p0_sandbox_v2.py"
BUILDER = ROOT / "scripts" / "v623_build_p0_payload.mjs"


def test_p0r1_diagnostic_uses_structural_confirmation() -> None:
    source = DIAGNOSTIC.read_text(encoding="utf-8")
    required = [
        "DHASH_CANDIDATE_HAMMING = 2",
        "PHASH_CANDIDATE_HAMMING = 4",
        "PIXEL_CORR_MIN = 0.97",
        "GRADIENT_CORR_MIN = 0.75",
        '"candidate_rule": "dhash64_hamming<=2 OR fft_phash64_hamming<=4"',
        '"confirmation_rule": "pixel_corr>=0.97 AND gradient_corr>=0.75"',
        '"near_duplicate_candidate_pairs_checked"',
    ]
    for marker in required:
        assert marker in source
    assert "NEAR_DUP_HAMMING" not in source


def test_p0_builder_fail_closed_transform_contract() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    builder = BUILDER.read_text(encoding="utf-8")

    # The builder intentionally transforms the frozen legacy template rather than
    # silently accepting drift. Each legacy anchor must remain unique until build.
    assert template.count("NEAR_DUP_HAMMING = 2") == 1
    assert template.count('_dhash_cache: dict[str, int] = {}') == 1
    assert template.count('"dhash": dhash(sample.path)') == 1
    assert template.count("dist <= NEAR_DUP_HAMMING") == 1

    required_builder_markers = [
        "function replaceExactly",
        "function applyStructuralNearDuplicateGate",
        "legacy dHash-only near-duplicate gate remains after patch",
        "DHASH_CANDIDATE_HAMMING = 2",
        "PHASH_CANDIDATE_HAMMING = 4",
        "PIXEL_CORR_MIN = 0.97",
        "GRADIENT_CORR_MIN = 0.75",
        "near_duplicate_gate: 'structural-confirmed-v2'",
    ]
    for marker in required_builder_markers:
        assert marker in builder


def test_p0_builder_javascript_syntax() -> None:
    node = shutil.which("node") or shutil.which("node.exe")
    assert node, "Node.js is required by the official P0 payload builder workflow"
    completed = subprocess.run(
        [node, "--check", str(BUILDER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
