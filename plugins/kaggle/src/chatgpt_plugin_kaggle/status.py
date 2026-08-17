from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from chatgpt_plugins_repair_engine import classify_failure

from .sanitize import sanitize_text


@dataclass(frozen=True, slots=True)
class KernelStatus:
    state: str
    raw: str


def normalize_kernel_status(raw: str) -> KernelStatus:
    text = sanitize_text(raw, max_chars=8000)
    lower = text.lower()
    # Be conservative: unknown output remains non-terminal rather than claiming success.
    if re.search(r"\b(error|failed|failure|cancelled|canceled)\b", lower):
        state = "failed"
    elif re.search(r"\b(complete|completed|success|succeeded)\b", lower):
        state = "succeeded"
    elif re.search(r"\b(running|executing)\b", lower):
        state = "running"
    elif re.search(r"\b(queued|pending)\b", lower):
        state = "queued"
    else:
        state = "unknown"
    return KernelStatus(state=state, raw=text)


def load_result_files(path: Path) -> tuple[dict, str]:
    result_candidates = sorted(path.rglob("result.json"))
    log_candidates = sorted(path.rglob("job.log"))
    result: dict = {}
    if result_candidates:
        try:
            loaded = json.loads(result_candidates[0].read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                result = loaded
        except json.JSONDecodeError:
            result = {}
    log = ""
    if log_candidates:
        log = sanitize_text(log_candidates[0].read_text(encoding="utf-8", errors="replace"), max_chars=16000)
    if "log_excerpt" in result:
        result["log_excerpt"] = sanitize_text(str(result["log_excerpt"]), max_chars=16000)
    return result, log


def classify_result_failure(result: dict, log: str):
    summary = str(result.get("summary") or "Kaggle execution failed")
    excerpt = str(result.get("log_excerpt") or log)
    return classify_failure(summary, excerpt)
