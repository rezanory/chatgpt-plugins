from __future__ import annotations

import re
import secrets

_SAFE = re.compile(r"[^a-z0-9-]+")


def new_job_id(prefix: str = "job") -> str:
    return f"{slugify(prefix, 12)}-{secrets.token_hex(6)}"


def slugify(value: str, max_length: int = 48) -> str:
    value = value.lower().strip().replace("_", "-")
    value = _SAFE.sub("-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    value = value[:max_length].strip("-")
    return value or "job"
