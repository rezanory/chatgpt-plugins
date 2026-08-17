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

# Direct Kaggle gateway invariant: runtime code uses the Python KaggleApi class directly. It must
# never spawn the `kaggle` CLI or any subprocess. Tests are allowed to use test helpers, but src is
# held to this boundary.
gateway_src = ROOT / "plugins/kaggle-gateway/src"
if gateway_src.exists():
    for path in gateway_src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in (
            r"\bimport\s+subprocess\b",
            r"\bfrom\s+subprocess\s+import\b",
            r"subprocess\.",
        ):
            if re.search(pattern, text):
                fail(
                    "direct Kaggle gateway must not use subprocess/CLI: "
                    f"{path.relative_to(ROOT)}"
                )

# Operational Kaggle GitHub Actions are intentionally forbidden. GitHub Actions may validate this
# repository, but Kaggle authentication/execution/status must occur in the long-running direct MCP
# gateway, never in CI.
for path in (ROOT / ".github/workflows").glob("kaggle-*.yml"):
    fail(f"obsolete Kaggle runtime workflow is forbidden: {path.relative_to(ROOT)}")

# The legacy trusted profile registry remains checked while the old relay package is retained for
# migration/reference. It is not on the runtime path.
profiles_path = ROOT / "plugins/kaggle/config/profiles.json"
if profiles_path.exists():
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    for name, profile in profiles.items():
        if "commands" in profile:
            fail(f"profile {name} uses forbidden commands field")
        for step in profile.get("steps", []):
            argv = step.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
                fail(f"profile {name} has invalid argv")

# Credential material must not be present in legacy account metadata.
accounts_path = ROOT / "plugins/kaggle/config/accounts.json"
if accounts_path.exists():
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
            fail(
                f"workflow directly interpolates untrusted event content: {path.name}: {forbidden}"
            )

if ERRORS:
    for error in ERRORS:
        print(f"SECURITY_GATE_FAIL: {error}", file=sys.stderr)
    raise SystemExit(1)
print("SECURITY_GATE_PASS")
