from __future__ import annotations

import re

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PATTERNS = [
    re.compile(r"(?i)(KAGGLE_API_TOKEN\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(KAGGLE_KEY\s*[=:]\s*)\S+"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)\S+"),
    re.compile(r"(?i)KGAT_[A-Za-z0-9._-]{8,}"),
    re.compile(r"(?i)(token|secret|password)(\s*[=:]\s*)['\"]?[^\s'\"]{8,}"),
]


def sanitize_text(value: str, *, max_chars: int = 20_000) -> str:
    text = _ANSI.sub("", value or "")
    text = text.replace("\x00", "")
    for pattern in _PATTERNS:
        if pattern.groups >= 1:
            text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    # Avoid pathological single-line output / prompt stuffing while retaining useful tail evidence.
    lines = []
    for line in text.splitlines():
        lines.append(line[:4000])
    text = "\n".join(lines)
    return text[-max_chars:]
