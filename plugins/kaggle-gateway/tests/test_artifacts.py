from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chatgpt_plugin_kaggle_gateway.artifacts import build_output_manifest


class FakePool:
    def __init__(self):
        self.output_path: Path | None = None
        self.file_pattern = ""

    def kernels_output(
        self,
        account_id: str,
        kernel_ref: str,
        path: str,
        *,
        file_pattern: str | None = None,
        force: bool = False,
        quiet: bool = True,
    ):
        assert account_id == "kg-01"
        assert kernel_ref == "owner-one/kernel"
        assert force is False
        assert quiet is True
        self.output_path = Path(path)
        self.file_pattern = file_pattern or ""
        (self.output_path / "KAGGLE_EXECUTION_V62_2.json").write_text(
            '{"fingerprint":"fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838"}',
            encoding="utf-8",
        )
        (self.output_path / "fingerprint.txt").write_text("other", encoding="utf-8")


def test_manifest_hashes_selected_outputs_and_finds_fingerprint():
    pool = FakePool()
    fingerprint = "fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838"
    result = build_output_manifest(
        pool,
        "kg-01",
        "owner-one/kernel",
        ["KAGGLE_EXECUTION_V62_2", "fingerprint.txt"],
        expected_fingerprint=fingerprint,
    )

    assert result["file_count"] == 2
    assert result["fingerprint_hits"] == ["KAGGLE_EXECUTION_V62_2.json"]
    by_name = {item["path"]: item for item in result["files"]}
    expected_bytes = (
        '{"fingerprint":"fe64ed64fc0a0bba80c55e343206046aa13edf87722a41494dd384b1d06b1838"}'
    ).encode()
    assert by_name["KAGGLE_EXECUTION_V62_2.json"]["sha256"] == hashlib.sha256(
        expected_bytes
    ).hexdigest()
    assert "KAGGLE_EXECUTION_V62_2" in pool.file_pattern
    assert r"fingerprint\.txt" in pool.file_pattern
    assert pool.output_path is not None
    assert not pool.output_path.exists()


def test_manifest_rejects_unsafe_artifact_names():
    with pytest.raises(ValueError, match="unsafe artifact name"):
        build_output_manifest(
            FakePool(),
            "kg-01",
            "owner-one/kernel",
            ["../secret"],
        )


def test_manifest_rejects_non_sha256_fingerprint():
    with pytest.raises(ValueError, match="64-character"):
        build_output_manifest(
            FakePool(),
            "kg-01",
            "owner-one/kernel",
            ["result.json"],
            expected_fingerprint="abc",
        )
