from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any

from .api_pool import KaggleApiPool

_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_ARTIFACTS = 16
_MAX_FILES = 64
_MAX_TOTAL_BYTES = 25 * 1024 * 1024
_MAX_FINGERPRINT_SCAN_BYTES = 2 * 1024 * 1024


def _validate_artifact_names(artifact_names: list[str]) -> list[str]:
    if not artifact_names or len(artifact_names) > _MAX_ARTIFACTS:
        raise ValueError(f"artifact_names must contain 1-{_MAX_ARTIFACTS} entries")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in artifact_names:
        name = raw.strip()
        if not _ARTIFACT_NAME.fullmatch(name):
            raise ValueError(f"unsafe artifact name: {raw!r}")
        if name not in seen:
            cleaned.append(name)
            seen.add(name)
    return cleaned


def _validate_fingerprint(expected_fingerprint: str | None) -> str | None:
    if not expected_fingerprint:
        return None
    value = expected_fingerprint.strip().lower()
    if not _SHA256.fullmatch(value):
        raise ValueError("expected_fingerprint must be a 64-character SHA-256 hex value")
    return value


def build_output_manifest(
    pool: KaggleApiPool,
    account_id: str,
    kernel_ref: str,
    artifact_names: list[str],
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Download selected existing output files, hash them, then discard the local copy.

    This is a read-only recovery helper. Artifact names are treated as literal fragments and are
    escaped before becoming Kaggle's file-pattern regex. No returned filename is executed or used
    outside a fresh temporary directory.
    """

    names = _validate_artifact_names(artifact_names)
    fingerprint = _validate_fingerprint(expected_fingerprint)
    file_pattern = "(?:" + "|".join(re.escape(name) for name in names) + ")"

    with tempfile.TemporaryDirectory(prefix="cgp-kaggle-output-") as tmp:
        root = Path(tmp).resolve()
        pool.kernels_output(
            account_id,
            kernel_ref,
            str(root),
            file_pattern=file_pattern,
            force=False,
            quiet=True,
        )

        files: list[dict[str, Any]] = []
        fingerprint_hits: list[str] = []
        total_bytes = 0
        candidates = sorted(path for path in root.rglob("*") if path.is_file())
        if len(candidates) > _MAX_FILES:
            raise RuntimeError(f"selected Kaggle output exceeds {_MAX_FILES} files")

        for path in candidates:
            if path.is_symlink():
                raise RuntimeError("refusing symlink in Kaggle output recovery")
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise RuntimeError("Kaggle output escaped the recovery directory") from exc

            size = resolved.stat().st_size
            total_bytes += size
            if total_bytes > _MAX_TOTAL_BYTES:
                raise RuntimeError(
                    f"selected Kaggle output exceeds {_MAX_TOTAL_BYTES} total bytes"
                )

            digest = hashlib.sha256()
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)

            files.append(
                {
                    "path": relative,
                    "size": size,
                    "sha256": digest.hexdigest(),
                }
            )

            if fingerprint and size <= _MAX_FINGERPRINT_SCAN_BYTES:
                content = resolved.read_bytes()
                if fingerprint.encode("ascii") in content.lower():
                    fingerprint_hits.append(relative)

        return {
            "account_id": account_id,
            "kernel_ref": kernel_ref,
            "requested_artifacts": names,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "files": files,
            "expected_fingerprint": fingerprint,
            "fingerprint_hits": fingerprint_hits,
        }
