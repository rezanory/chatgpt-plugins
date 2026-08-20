from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .sanitize import sanitize_text


class KaggleCliError(RuntimeError):
    pass


def ensure_auth_present() -> None:
    """Require the current non-interactive Kaggle API token flow.

    The provider accepts the official non-interactive ``KAGGLE_API_TOKEN`` path. Interactive
    OAuth/browser login belongs in explicit bootstrap/admin workflows rather than normal provider
    actions.
    """

    if os.getenv("KAGGLE_API_TOKEN"):
        return
    raise KaggleCliError(
        "Kaggle API token missing: configure KAGGLE_API_TOKEN in the selected GitHub Environment"
    )


def _run_cli(
    command: list[str],
    *,
    timeout: int,
    require_auth: bool,
    max_chars: int = 30_000,
) -> str:
    if require_auth:
        ensure_auth_present()
    proc = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=os.environ.copy(),
        shell=False,
    )
    output = sanitize_text(proc.stdout or "", max_chars=max_chars)
    if proc.returncode != 0:
        rendered_command = " ".join(command[:4])
        raise KaggleCliError(
            f"{rendered_command} failed ({proc.returncode}): {output[-8000:]}"
        )
    return output


def run_kaggle(args: list[str], *, timeout: int = 120) -> str:
    """Universal authenticated Kaggle CLI transport.

    This function is deliberately not restricted to a small command subset. Higher layers must
    apply Control Plane v3 safety classification and user-authorization rules before using it for
    write, compute, destructive, or privileged operations. ``shell=False`` prevents shell-command
    injection; the executable is always the official ``kaggle`` CLI.
    """

    return _run_cli(["kaggle", *args], timeout=timeout, require_auth=True)


def kaggle_version() -> str:
    """Return the installed Kaggle CLI version without requiring credentials."""

    return _run_cli(["kaggle", "--version"], timeout=30, require_auth=False, max_chars=4000)


def kaggle_help(path: list[str] | None = None) -> str:
    """Return current CLI help for a command path without performing a mutation."""

    parts = list(path or [])
    return _run_cli(
        ["kaggle", *parts, "--help"],
        timeout=45,
        require_auth=False,
        max_chars=30_000,
    )


def create_private_dataset(path: Path) -> str:
    # Kaggle datasets create is private unless --public is explicitly passed.
    return run_kaggle(["datasets", "create", "-p", str(path), "-q"], timeout=600)


def dataset_status(dataset_ref: str) -> str:
    return run_kaggle(["datasets", "status", dataset_ref], timeout=120)


def wait_dataset_ready(dataset_ref: str, *, timeout: int = 300, interval: int = 5) -> str:
    deadline = time.monotonic() + timeout
    latest = ""
    while time.monotonic() < deadline:
        latest = dataset_status(dataset_ref)
        lower = latest.lower()
        if any(word in lower for word in ("error", "failed", "failure")):
            raise KaggleCliError(f"dataset creation failed: {latest[-4000:]}")
        if any(word in lower for word in ("ready", "complete", "completed", "success")):
            return latest
        time.sleep(interval)
    raise KaggleCliError(f"dataset did not become ready within {timeout}s: {latest[-4000:]}")


def push_kernel(path: Path, accelerator: str | None = None) -> str:
    args = ["kernels", "push", "-p", str(path)]
    if accelerator:
        args += ["--accelerator", accelerator]
    return run_kaggle(args, timeout=600)


def kernel_status(kernel_ref: str) -> str:
    return run_kaggle(["kernels", "status", kernel_ref], timeout=120)


def kernel_output(kernel_ref: str, path: Path, *, file_pattern: str | None = None) -> str:
    path.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", kernel_ref, "-p", str(path), "-o", "-q"]
    if file_pattern:
        args += ["--file-pattern", file_pattern]
    return run_kaggle(args, timeout=600)
