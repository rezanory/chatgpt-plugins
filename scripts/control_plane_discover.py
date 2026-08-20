#!/usr/bin/env python3
"""Discover provider CLI capability trees without executing mutations.

This script is intentionally read-only: it invokes only --help/help/version style commands.
It sanitizes environment-variable names that may contain credentials and never prints values.

Examples:
    python scripts/control_plane_discover.py --provider all --depth 2
    python scripts/control_plane_discover.py --provider kaggle --json-out cap.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SECRET_NAME = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|ACCESS_KEY)", re.I)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class CommandNode:
    argv: list[str]
    available: bool
    returncode: int | None
    commands: list[str]
    help_excerpt: str
    error: str | None = None


def clean(text: str, limit: int = 24000) -> str:
    text = ANSI.sub("", text or "")
    text = text.replace("\r\n", "\n")
    return text[-limit:]


def run_help(argv: list[str], timeout: int = 45) -> tuple[int, str, str]:
    env = {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}
    proc = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        env=env,
    )
    return proc.returncode, clean(proc.stdout), clean(proc.stderr)


def executable_for(provider: str) -> list[str] | None:
    if provider == "kaggle":
        exe = shutil.which("kaggle")
        return [exe] if exe else None
    if provider == "github":
        exe = shutil.which("gh")
        return [exe] if exe else None
    if provider == "cloudflare":
        exe = shutil.which("wrangler")
        if exe:
            return [exe]
        npx = shutil.which("npx")
        return [npx, "wrangler"] if npx else None
    raise ValueError(provider)


def parse_commands(provider: str, text: str) -> list[str]:
    """Best-effort parser; runtime help text remains the evidence source."""
    lines = text.splitlines()
    candidates: list[str] = []

    # Common `command  description` formats used by kaggle/gh/wrangler.
    patterns = [
        re.compile(r"^\s{2,}([a-zA-Z0-9][a-zA-Z0-9:_-]*)\s{2,}.+"),
        re.compile(r"^\s{0,4}([a-zA-Z0-9][a-zA-Z0-9:_-]*)\s+-\s+.+"),
    ]
    reserved = {
        "usage", "options", "flags", "commands", "examples", "arguments", "aliases",
        "global", "help", "learn", "environment", "resources", "version",
    }
    for line in lines:
        for pattern in patterns:
            m = pattern.match(line)
            if not m:
                continue
            name = m.group(1)
            if name.lower() in reserved or name.startswith("-"):
                continue
            if name not in candidates:
                candidates.append(name)
            break

    # Provider-specific known section fallback is only used to make discovery resilient;
    # no command is ever executed except with help flags.
    if provider == "kaggle":
        baseline = [
            "competitions", "datasets", "kernels", "models", "files", "forums",
            "benchmarks", "config", "auth", "quota", "search",
        ]
    elif provider == "github":
        baseline = [
            "api", "auth", "browse", "cache", "codespace", "gist", "issue", "org",
            "pr", "project", "release", "repo", "ruleset", "run", "search", "secret",
            "ssh-key", "status", "variable", "workflow",
        ]
    else:
        baseline = [
            "deploy", "dev", "versions", "rollback", "pages", "kv", "d1", "r2",
            "queues", "workflows", "pipelines", "hyperdrive", "vectorize", "containers",
            "tunnel", "secrets-store", "vpc",
        ]
    for name in baseline:
        if name in text and name not in candidates:
            candidates.append(name)
    return candidates


def help_argv(provider: str, base: list[str], path: list[str]) -> list[str]:
    if provider == "github":
        return [*base, *path, "--help"]
    return [*base, *path, "--help"]


def discover(provider: str, depth: int) -> dict:
    base = executable_for(provider)
    if not base:
        return {
            "provider": provider,
            "available": False,
            "executable": None,
            "nodes": [],
            "note": "CLI executable not found; use connector/API fallback.",
        }

    nodes: list[CommandNode] = []
    queue: list[tuple[list[str], int]] = [([], 0)]
    seen: set[tuple[str, ...]] = set()
    while queue:
        path, level = queue.pop(0)
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        argv = help_argv(provider, base, path)
        try:
            rc, out, err = run_help(argv)
            text = out if out.strip() else err
            commands = parse_commands(provider, text) if rc == 0 else []
            nodes.append(
                CommandNode(
                    argv=[Path(base[0]).name, *base[1:], *path, "--help"],
                    available=rc == 0,
                    returncode=rc,
                    commands=commands,
                    help_excerpt=text,
                    error=None if rc == 0 else clean(err or out, 4000),
                )
            )
            if rc == 0 and level < depth:
                for command in commands:
                    queue.append(([*path, command], level + 1))
        except (subprocess.TimeoutExpired, OSError) as exc:
            nodes.append(
                CommandNode(
                    argv=[Path(base[0]).name, *base[1:], *path, "--help"],
                    available=False,
                    returncode=None,
                    commands=[],
                    help_excerpt="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    version_argv = [*base, "--version"]
    try:
        rc, out, err = run_help(version_argv, timeout=20)
        version = clean((out or err).strip(), 2000) if rc == 0 else None
    except Exception as exc:  # discovery should not fail because version probing failed
        version = f"probe-error:{type(exc).__name__}"

    return {
        "provider": provider,
        "available": True,
        "executable": [Path(base[0]).name, *base[1:]],
        "version": version,
        "nodes": [asdict(node) for node in nodes],
    }


def iter_providers(value: str) -> Iterable[str]:
    if value == "all":
        return ("kaggle", "cloudflare", "github")
    return (value,)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["all", "kaggle", "cloudflare", "github"], default="all")
    parser.add_argument("--depth", type=int, default=1, choices=range(0, 4))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = {
        "schema_version": 1,
        "read_only_discovery": True,
        "secret_values_emitted": False,
        "providers": [discover(provider, args.depth) for provider in iter_providers(args.provider)],
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
