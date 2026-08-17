from __future__ import annotations

import fnmatch
import json
import os
import stat
import zipfile
from pathlib import Path

DEFAULT_EXCLUDES = (
    ".git/*",
    ".git",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.crt",
    ".venv/*",
    "venv/*",
    "node_modules/*",
    "__pycache__/*",
    ".pytest_cache/*",
    ".mypy_cache/*",
    ".ruff_cache/*",
    "dist/*",
    "build/*",
)


def _excluded(relative: str) -> bool:
    normalized = relative.replace(os.sep, "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in DEFAULT_EXCLUDES)


def package_source(
    source_dir: Path,
    output_zip: Path,
    *,
    max_file_bytes: int = 100_000_000,
    max_total_bytes: int = 1_000_000_000,
    max_files: int = 50_000,
) -> int:
    source_dir = source_dir.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(source_dir).as_posix()
            if relative.startswith("/") or ".." in Path(relative).parts or _excluded(relative):
                continue
            mode = path.stat().st_mode
            if not stat.S_ISREG(mode):
                continue
            size = path.stat().st_size
            if size > max_file_bytes:
                raise ValueError(f"refusing to package oversized file: {relative}")
            total += size
            count += 1
            if total > max_total_bytes:
                raise ValueError("source package exceeds total size limit")
            if count > max_files:
                raise ValueError("source package exceeds file count limit")
            zf.write(path, relative)
    if count == 0:
        raise ValueError("source package is empty")
    return count


def write_dataset_metadata(path: Path, *, owner: str, slug: str, job_id: str) -> str:
    dataset_ref = f"{owner}/{slug}"
    payload = {
        "title": f"CGP Source {job_id}"[:50],
        "id": dataset_ref,
        "licenses": [{"name": "other"}],
        "description": "Private ephemeral source package created by chatgpt-plugins V0.1.",
    }
    (path / "dataset-metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dataset_ref
