#!/usr/bin/env bash
set -euo pipefail
uv sync --all-packages --dev
uv run python scripts/security_gate.py
uv run ruff check .
uv run pytest
uv run python -m compileall -q packages plugins
