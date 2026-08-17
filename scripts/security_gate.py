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

# Direct Kaggle gateway invariant: runtime code uses KaggleApi directly and never shells out to
# the Kaggle CLI.
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

# Operational Kaggle GitHub Actions are forbidden. CI may validate source/deployment packages but
# Kaggle auth/execution/status must happen only inside the remote direct gateway runtime.
for path in (ROOT / ".github/workflows").glob("kaggle-*.yml"):
    fail(f"obsolete Kaggle runtime workflow is forbidden: {path.relative_to(ROOT)}")

# Legacy trusted profiles remain checked while old relay code is retained for migration/reference.
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

# Cloudflare is the active production boundary. The Worker must protect /mcp with a validated
# Access JWT before forwarding to the Python Container. Secret *names* are allowed; credential
# values must never be hard-coded in the deployment package.
cloudflare_root = ROOT / "deploy/cloudflare-container"
cloudflare_worker = cloudflare_root / "src/index.ts"
if cloudflare_worker.exists():
    worker_text = cloudflare_worker.read_text(encoding="utf-8")
    required_fragments = (
        'url.pathname.startsWith("/mcp")',
        'request.headers.get("cf-access-jwt-assertion")',
        "jwtVerify(",
        "audience: env.POLICY_AUD",
        'forwarded.headers.delete("cf-access-jwt-assertion")',
        'forwarded.headers.delete("cookie")',
    )
    for fragment in required_fragments:
        if fragment not in worker_text:
            fail(f"Cloudflare /mcp security boundary missing required fragment: {fragment}")

    suspicious_literals = re.findall(
        r"CGP_KAGGLE_[A-Z0-9_]+\s*[:=]\s*[\"'][^\"'$][^\"']{12,}[\"']",
        worker_text,
    )
    if suspicious_literals:
        fail("Cloudflare Worker appears to hard-code a Kaggle credential value")
else:
    fail("active Cloudflare Worker entrypoint is missing")

wrangler_path = cloudflare_root / "wrangler.jsonc"
if wrangler_path.exists():
    wrangler_text = wrangler_path.read_text(encoding="utf-8")
    if '"class_name": "KaggleGatewayContainer"' not in wrangler_text:
        fail("Cloudflare Container class binding is missing")
    if '"max_instances": 1' not in wrangler_text:
        fail("V0.1 Cloudflare runtime must keep a single gateway container instance")
else:
    fail("Cloudflare wrangler.jsonc is missing")

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
