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

    V0.1 intentionally does not use browser sessions, cookies, OAuth login, or legacy
    username/key credentials. Every account is expected to expose exactly one GitHub Environment
    secret named ``KAGGLE_API_TOKEN`` generated from Kaggle Settings -> API.
    """

    if os.getenv("KAGGLE_API_TOKEN"):
        return
    raise KaggleCliError(
        "Kaggle API token missing: configure KAGGLE_API_TOKEN in the selected GitHub Environment"
    )


def run_kaggle(args: list[str], *, timeout: int = 120) -> str:
    ensure_auth_present()
    command = ["kaggle", *args]
    proc = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=os.environ.copy(),
    )
    output = sanitize_text(proc.stdout or "", max_chars=30_000)
    if proc.returncode != 0:
        rendered_command = " ".join(command[:3])
        raise KaggleCliError(f"{rendered_command} failed ({proc.returncode}): {output[-8000:]}")
    return output


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
