from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .sanitize import sanitize_text


class KaggleCliError(RuntimeError):
    pass


def ensure_auth_present() -> None:
    token = os.getenv("KAGGLE_API_TOKEN")
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if token:
        return
    if username and key:
        return
    raise KaggleCliError(
        "Kaggle credential missing: configure KAGGLE_API_TOKEN or legacy KAGGLE_USERNAME/KAGGLE_KEY"
    )


def _run_once(command: list[str], *, timeout: int, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
    )


def run_kaggle(args: list[str], *, timeout: int = 120) -> str:
    """Run the official Kaggle CLI without exposing credentials.

    V0.1 prefers the current ``KAGGLE_API_TOKEN`` flow. Some operators still possess the legacy
    Kaggle API key that historically lived in ``kaggle.json``. During bootstrap those keys can be
    accidentally placed in the new-token secret slot. If (and only if) the CLI explicitly reports
    that authentication is required, we make one in-memory compatibility retry using the selected
    account's public owner slug as ``KAGGLE_USERNAME`` and the supplied secret as ``KAGGLE_KEY``.
    The secret is never printed, persisted, or returned to ChatGPT.
    """

    ensure_auth_present()
    command = ["kaggle", *args]
    base_env = os.environ.copy()
    proc = _run_once(command, timeout=timeout, env=base_env)
    output = sanitize_text(proc.stdout or "", max_chars=30_000)
    if proc.returncode == 0:
        return output

    token = base_env.get("KAGGLE_API_TOKEN", "")
    owner = base_env.get("KAGGLE_OWNER", "")
    legacy_key = base_env.get("KAGGLE_KEY", "")
    auth_required = "authentication required" in output.lower()

    # Compatibility bridge for a legacy API key stored in the new-token slot. The fallback is
    # deliberately narrow: only an explicit authentication-required response may trigger it.
    if token and owner and not legacy_key and auth_required:
        fallback_env = base_env.copy()
        fallback_env.pop("KAGGLE_API_TOKEN", None)
        fallback_env["KAGGLE_USERNAME"] = fallback_env.get("KAGGLE_USERNAME") or owner
        fallback_env["KAGGLE_KEY"] = token
        retry = _run_once(command, timeout=timeout, env=fallback_env)
        retry_output = sanitize_text(retry.stdout or "", max_chars=30_000)
        if retry.returncode == 0:
            return retry_output
        rendered_command = " ".join(command[:3])
        raise KaggleCliError(
            f"{rendered_command} failed after current-token auth and legacy-key compatibility retry "
            f"({retry.returncode}): {retry_output[-8000:]}"
        )

    rendered_command = " ".join(command[:3])
    raise KaggleCliError(f"{rendered_command} failed ({proc.returncode}): {output[-8000:]}")


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
