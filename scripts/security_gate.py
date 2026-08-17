from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []


def fail(message: str) -> None:
    ERRORS.append(message)


# Source-code execution hazards that should never appear in bridge/provider Python.
for path in list((ROOT / "packages").rglob("*.py")) + list((ROOT / "plugins").rglob("*.py")):
    text = path.read_text(encoding="utf-8")
    for pattern in (r"shell\s*=\s*True", r"\bos\.system\s*\(", r"\beval\s*\(", r"\bexec\s*\("):
        if re.search(pattern, text):
            fail(f"unsafe execution primitive {pattern!r} in {path.relative_to(ROOT)}")

# The trusted profile registry must use steps/argv and never legacy command strings.
profiles_path = ROOT / "plugins/kaggle/config/profiles.json"
profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
for name, profile in profiles.items():
    if "commands" in profile:
        fail(f"profile {name} uses forbidden commands field")
    for step in profile.get("steps", []):
        argv = step.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            fail(f"profile {name} has invalid argv")

# Credential material must not be present in account metadata.
accounts_path = ROOT / "plugins/kaggle/config/accounts.json"
accounts_text = accounts_path.read_text(encoding="utf-8")
for forbidden in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "api_token", "password", "private_key"):
    if forbidden.lower() in accounts_text.lower():
        fail(f"accounts.json contains forbidden credential-like field/text: {forbidden}")

# Never interpolate raw Issue body/comment text into a workflow shell script.
for path in (ROOT / ".github/workflows").glob("*.yml"):
    text = path.read_text(encoding="utf-8")
    for forbidden in (
        "${{ github.event.issue.body }}",
        "${{ github.event.comment.body }}",
        "${{ toJSON(github.event.issue) }}",
    ):
        if forbidden in text:
            fail(f"workflow directly interpolates untrusted event content: {path.name}: {forbidden}")

if ERRORS:
    for error in ERRORS:
        print(f"SECURITY_GATE_FAIL: {error}", file=sys.stderr)
    raise SystemExit(1)
print("SECURITY_GATE_PASS")
