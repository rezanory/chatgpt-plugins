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

# Retained Python reference gateway must remain direct KaggleApi only and never shell out to a CLI.
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
                    "Kaggle gateway reference code must not use subprocess/CLI: "
                    f"{path.relative_to(ROOT)}"
                )

# Kaggle runtime credentials must not be referenced directly by ordinary GitHub Actions workflows.
# The temporary Cloudflare bootstrap workflow constructs binding names at runtime so the Worker can
# be activated without exposing the provider tokens as first-class Actions secret names.
for path in (ROOT / ".github/workflows").glob("kaggle-*.yml"):
    fail(f"operational Kaggle GitHub Actions workflow is forbidden: {path.relative_to(ROOT)}")
for path in (ROOT / ".github/workflows").glob("*.yml"):
    text = path.read_text(encoding="utf-8")
    for forbidden in (
        "CGP_KAGGLE_KG02_TOKEN",
        "CGP_KAGGLE_KG03_TOKEN",
        "CGP_KAGGLE_KG04_TOKEN",
        "CGP_KAGGLE_KG05_TOKEN",
        "CGP_KAGGLE_KG06_TOKEN",
        "CGP_KAGGLE_KG07_TOKEN",
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
    ):
        if forbidden in text:
            fail(f"GitHub Actions must never receive Kaggle runtime credentials: {path.name}")

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

# V0.1 production hosting is zero-cost only. Previously rejected hosting paths stay forbidden.
for abandoned in (
    ROOT / "render.yaml",
    ROOT / "deploy/render-free",
    ROOT / "deploy/cloudflare-container",
):
    if abandoned.is_file() or (abandoned.is_dir() and any(abandoned.rglob("*"))):
        fail(f"abandoned or paid runtime must not exist: {abandoned.relative_to(ROOT)}")

# Active production boundary: one plain Cloudflare Worker on the Workers Free plan, with no
# Container, Durable Object binding, queue, or paid runtime dependency.
worker_root = ROOT / "deploy/cloudflare-worker-free"
wrangler_path = worker_root / "wrangler.jsonc"
worker_entry = worker_root / "src/index.ts"
kaggle_client = worker_root / "src/kaggle.ts"
github_control = worker_root / "src/github.ts"
for required in (wrangler_path, worker_entry, kaggle_client, github_control):
    if not required.is_file():
        fail(f"Cloudflare Workers Free runtime file is missing: {required.relative_to(ROOT)}")

if wrangler_path.is_file():
    wrangler_text = wrangler_path.read_text(encoding="utf-8")
    required_wrangler = (
        '"name": "chatgpt-kaggle-gateway"',
        '"workers_dev": true',
        '"preview_urls": false',
        '"CGP_WRITE_ENABLED": "0"',
    )
    for fragment in required_wrangler:
        if fragment not in wrangler_text:
            fail(f"Workers Free wrangler config missing required fragment: {fragment}")
    for forbidden in ('"containers"', '"durable_objects"', '"queues"'):
        if forbidden in wrangler_text:
            fail(f"Workers Free runtime contains forbidden paid/stateful binding: {forbidden}")

if kaggle_client.is_file():
    kaggle_text = kaggle_client.read_text(encoding="utf-8")
    required_kaggle = (
        'const KAGGLE_API_ROOT = "https://api.kaggle.com/v1"',
        'account.token.startsWith("KGAT_")',
        '`Bearer ${account.token}`',
        '`Basic ${btoa(`${account.username}:${account.token}`)}`',
        '"ListKernels"',
        '"GetKernelSessionStatus"',
        '"ListKernelSessionOutput"',
        '"GetKernel"',
        '"SaveKernel"',
        'kernelExecutionType: "SAVE_AND_RUN_ALL"',
        "CGP_MCP_PATH_TOKEN",
    )
    for fragment in required_kaggle:
        if fragment not in kaggle_text:
            fail(f"direct Kaggle HTTPS client missing required fragment: {fragment}")
    if "KaggleApi(" in kaggle_text or "kaggle cli" in kaggle_text.lower():
        fail("active Worker runtime must use direct HTTPS rather than Python/CLI")

if worker_entry.is_file():
    entry_text = worker_entry.read_text(encoding="utf-8")
    required_entry = (
        'url.pathname === "/healthz"',
        'url.pathname === "/github/webhook"',
        "CGP_MCP_PATH_TOKEN",
        "privateMcpPath(",
        "createMcpHandler(",
        'transport: "cloudflare-workers-free"',
        "kaggle_auth_check_all",
        "kaggle_kernels_inventory_all",
    )
    for fragment in required_entry:
        if fragment not in entry_text:
            fail(f"Workers Free MCP entry missing required fragment: {fragment}")

if github_control.is_file():
    control_text = github_control.read_text(encoding="utf-8")
    required_control = (
        "x-hub-signature-256",
        "CGP_GITHUB_WEBHOOK_SECRET",
        'env.CGP_WRITE_ENABLED !== "1"',
        "write_disabled_until_recovery",
        'action !== "rerun_existing"',
        "hasJobMarker(",
        "rerunExisting(",
        "CGP_GITHUB_TOKEN",
    )
    for fragment in required_control:
        if fragment not in control_text:
            fail(f"signed GitHub write bridge missing required fragment: {fragment}")

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
