from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any


def load_github_event() -> dict[str, Any]:
    path = os.getenv("GITHUB_EVENT_PATH")
    if not path:
        raise RuntimeError("GITHUB_EVENT_PATH is not set")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("GitHub event payload must be an object")
    return data


def write_github_output(name: str, value: str) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        print(f"{name}={value}")
        return
    delimiter = f"CGP_{secrets.token_hex(8)}"
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")
