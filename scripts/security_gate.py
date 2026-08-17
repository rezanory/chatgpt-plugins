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

# Active Kaggle runtime invariant: the gateway uses KaggleApi directly and never shells out to the
# Kaggle CLI. GitHub may be the control plane, but Actions/CI must never authenticate to Kaggle.
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

for path in (ROOT / ".github/workflows").glob("kaggle-*.yml"):
    fail(f"operational Kaggle GitHub Actions workflow is forbidden: {path.relative_to(ROOT)}")

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

# Credential material must never be committed to account metadata.
accounts_path = ROOT / "plugins/kaggle/config/accounts.json"
if accounts_path.exists():
    accounts_text = accounts_path.read_text(encoding="utf-8")
    for forbidden in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "api_token", "password", "private_key"):
        if forbidden.lower() in accounts_text.lower():
            fail(f"accounts.json contains forbidden credential-like field/text: {forbidden}")

# V0.1 production hosting is deliberately free-only. Cloudflare Containers require a paid Workers
# plan, so that runtime must not reappear. Render is used only as a free web service.
cloudflare_container = ROOT / "deploy/cloudflare-container"
if cloudflare_container.exists() and any(cloudflare_container.rglob("*")):
    fail("paid Cloudflare Container runtime must not exist in V0.1")

render_blueprint = ROOT / "render.yaml"
render_dockerfile = ROOT / "deploy/render-free/Dockerfile"
if not render_blueprint.is_file():
    fail("render.yaml is missing")
else:
    render_text = render_blueprint.read_text(encoding="utf-8")
    required_render = (
        "plan: free",
        "runtime: docker",
        "dockerfilePath: ./deploy/render-free/Dockerfile",
        "healthCheckPath: /healthz",
        "CGP_GITHUB_WEBHOOK_SECRET",
        "CGP_GITHUB_TOKEN",
        "CGP_MCP_PATH_SECRET",
    )
    for fragment in required_render:
        if fragment not in render_text:
            fail(f"Render Free blueprint missing required fragment: {fragment}")
    for paid_plan in ("plan: starter", "plan: standard", "plan: pro"):
        if paid_plan in render_text:
            fail(f"paid Render plan is forbidden in V0.1: {paid_plan}")
if not render_dockerfile.is_file():
    fail("Render Free Dockerfile is missing")

# Signed GitHub webhook is the Pro-compatible write bridge. The webhook must be authenticated,
# repository/actor constrained, and only the narrow rerun_existing action is accepted in V0.1.
server_path = gateway_src / "chatgpt_plugin_kaggle_gateway/server.py"
control_path = gateway_src / "chatgpt_plugin_kaggle_gateway/control.py"
if server_path.is_file():
    server_text = server_path.read_text(encoding="utf-8")
    required_server = (
        '@_mcp.custom_route("/github/webhook", methods=["POST"])',
        "CGP_GITHUB_WEBHOOK_SECRET",
        "verify_github_signature(",
        "CGP_CONTROL_REPOSITORY",
        "CGP_GITHUB_ALLOWED_ACTORS",
        "CGP_MCP_PATH_SECRET",
    )
    for fragment in required_server:
        if fragment not in server_text:
            fail(f"Render/GitHub control boundary missing required fragment: {fragment}")
else:
    fail("Kaggle gateway server.py is missing")

if control_path.is_file():
    control_text = control_path.read_text(encoding="utf-8")
    required_control = (
        'self.action != "rerun_existing"',
        "verify_github_signature",
        "kernels_pull(",
        "kernels_push(",
        "CLAIM_MARKER",
        "RECEIPT_MARKER",
        "CGP_GITHUB_TOKEN",
    )
    for fragment in required_control:
        if fragment not in control_text:
            fail(f"direct write control missing required fragment: {fragment}")
else:
    fail("Kaggle gateway control.py is missing")

# Never interpolate raw Issue body/comment text into an Actions shell script.
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
