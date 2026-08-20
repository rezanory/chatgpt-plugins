from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence

from .capability_registry import SafetyClass


class CliTransportError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CliReceipt:
    provider: str
    argv: tuple[str, ...]
    safety: SafetyClass
    returncode: int
    output: str


_EXECUTABLE = {
    "github": "gh",
    "cloudflare": "wrangler",
    "kaggle": "kaggle",
}

_READ_HINTS = {
    "help",
    "--help",
    "-h",
    "version",
    "--version",
    "list",
    "ls",
    "get",
    "view",
    "show",
    "status",
    "search",
    "find",
    "describe",
    "info",
    "inspect",
    "logs",
    "log",
    "output",
    "download",
    "whoami",
    "quota",
    "read",
    "tail",
}

_MUTATION_HINTS = {
    "create",
    "update",
    "edit",
    "set",
    "add",
    "remove",
    "delete",
    "destroy",
    "push",
    "submit",
    "cancel",
    "rerun",
    "run",
    "deploy",
    "publish",
    "upload",
    "merge",
    "close",
    "reopen",
    "enable",
    "disable",
    "rollback",
    "restore",
    "put",
    "write",
    "execute",
    "apply",
    "approve",
    "reject",
    "lock",
    "unlock",
    "invite",
    "grant",
    "revoke",
}


def _allowed_classes(env: Mapping[str, str] | None = None) -> set[SafetyClass]:
    source = os.environ if env is None else env
    raw = source.get("CONTROL_PLANE_ALLOWED_CLASSES", "read")
    values = {item.strip().lower() for item in raw.replace(",", " ").split() if item.strip()}
    try:
        return {SafetyClass(value) for value in values}
    except ValueError as exc:
        raise CliTransportError(f"invalid CONTROL_PLANE_ALLOWED_CLASSES: {sorted(values)}") from exc


def _validate_read_argv(args: Sequence[str]) -> None:
    words = {str(arg).strip().lower() for arg in args if str(arg).strip()}
    if words & _MUTATION_HINTS:
        raise CliTransportError(
            "command contains a mutation-like verb and cannot be classified as read"
        )
    if not words & _READ_HINTS:
        raise CliTransportError(
            "read-only classification is not proven for this command; use an explicit non-read safety class"
        )


def run_provider_cli(
    provider: str,
    args: Sequence[str],
    *,
    safety: SafetyClass = SafetyClass.READ,
    timeout: int = 120,
    env: Mapping[str, str] | None = None,
    max_chars: int = 100_000,
) -> CliReceipt:
    """Run an official provider CLI with a fixed executable and safety-class gate.

    The transport is intentionally generic: it does not enumerate a closed set of commands.
    Unknown/new commands remain usable. If their read-only nature cannot be proven, callers must
    classify them as write/compute/destructive/privileged and explicitly grant that class through
    CONTROL_PLANE_ALLOWED_CLASSES.
    """

    try:
        executable = _EXECUTABLE[provider]
    except KeyError as exc:
        raise CliTransportError(f"unsupported provider: {provider}") from exc
    resolved = shutil.which(executable)
    if not resolved:
        raise CliTransportError(f"{executable} CLI is not installed")
    if safety is SafetyClass.READ:
        _validate_read_argv(args)
    if safety not in _allowed_classes(env):
        raise CliTransportError(f"safety class {safety.value!r} is not granted")

    child_env = dict(os.environ if env is None else env)
    command = [resolved, *[str(arg) for arg in args]]
    proc = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        shell=False,
        env=child_env,
    )
    output = (proc.stdout or "")[-max_chars:]
    if proc.returncode != 0:
        preview = " ".join([executable, *[str(arg) for arg in args[:3]]])
        raise CliTransportError(
            f"{preview} failed ({proc.returncode}): {output[-8000:]}"
        )
    return CliReceipt(
        provider=provider,
        argv=(executable, *tuple(str(arg) for arg in args)),
        safety=safety,
        returncode=proc.returncode,
        output=output,
    )
