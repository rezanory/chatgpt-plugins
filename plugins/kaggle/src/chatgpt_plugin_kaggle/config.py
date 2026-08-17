from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chatgpt_plugins_core import AccountDescriptor, RepoPolicy

_SAFE_ARG = re.compile(r"^[^\x00\r\n]{1,1000}$")


@dataclass(frozen=True, slots=True)
class KaggleAccountConfig:
    descriptor: AccountDescriptor
    owner_slug: str
    default_accelerator: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    argv: tuple[str, ...]
    cwd: str = "."

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("execution step argv must not be empty")
        if len(self.argv) > 64:
            raise ValueError("execution step argv is too large")
        for arg in self.argv:
            if not _SAFE_ARG.fullmatch(arg):
                raise ValueError("execution step contains unsafe control characters or oversized argument")
        if self.cwd.startswith("/") or ".." in Path(self.cwd).parts:
            raise ValueError("execution step cwd must stay inside the packaged source")


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    name: str
    description: str
    capabilities: frozenset[str]
    steps: tuple[ExecutionStep, ...]
    artifact_globs: tuple[str, ...] = ()
    internet: bool = False

    @property
    def commands(self) -> tuple[str, ...]:
        """Compatibility view for older callers; never feed this property to a shell."""
        return tuple(" ".join(step.argv) for step in self.steps)


@dataclass(frozen=True, slots=True)
class Settings:
    db_path: str
    public_base_url: str
    accounts_file: str
    profiles_file: str
    repo_policy: RepoPolicy
    github_repository: str | None
    github_workflow: str
    github_ref: str
    github_token: str | None


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_accounts(path: str) -> list[KaggleAccountConfig]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError("accounts config must be a list")
    result: list[KaggleAccountConfig] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("account entry must be an object")
        account_id = str(item["account_id"]).strip()
        if account_id in seen:
            raise ValueError(f"duplicate account_id: {account_id}")
        seen.add(account_id)
        owner_slug = str(item["owner_slug"]).strip()
        if not owner_slug:
            raise ValueError(f"owner_slug required for {account_id}")
        descriptor = AccountDescriptor(
            account_id=account_id,
            provider="kaggle",
            secret_scope=str(item["secret_scope"]),
            enabled=bool(item.get("enabled", True)),
            max_parallel=int(item.get("max_parallel", 1)),
            capabilities=frozenset(str(v) for v in item.get("capabilities", ["cpu"])),
            labels=frozenset(str(v) for v in item.get("labels", [])),
            weight=float(item.get("weight", 1.0)),
        )
        result.append(
            KaggleAccountConfig(
                descriptor=descriptor,
                owner_slug=owner_slug,
                default_accelerator=(str(item["default_accelerator"]) if item.get("default_accelerator") else None),
            )
        )
    return result


def load_profiles(path: str) -> dict[str, ExecutionProfile]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError("profiles config must be an object")
    profiles: dict[str, ExecutionProfile] = {}
    for name, item in data.items():
        if not isinstance(item, dict):
            raise ValueError(f"profile {name!r} must be an object")
        raw_steps = item.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"profile {name!r} must define non-empty steps")
        steps: list[ExecutionStep] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict) or set(raw_step) - {"argv", "cwd"}:
                raise ValueError(f"profile {name!r} has invalid step shape")
            argv = raw_step.get("argv")
            if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
                raise ValueError(f"profile {name!r} step argv must be a list of strings")
            steps.append(ExecutionStep(tuple(argv), str(raw_step.get("cwd", "."))))
        profiles[name] = ExecutionProfile(
            name=name,
            description=str(item.get("description", "")),
            capabilities=frozenset(str(v) for v in item.get("capabilities", ["cpu"])),
            steps=tuple(steps),
            artifact_globs=tuple(str(v) for v in item.get("artifact_globs", [])),
            internet=bool(item.get("internet", False)),
        )
    return profiles



def load_repo_policy(path: str) -> RepoPolicy:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError("source repo policy must be an object")
    patterns = data.get("allowed_repositories", [])
    if not isinstance(patterns, list) or not all(isinstance(v, str) for v in patterns):
        raise ValueError("allowed_repositories must be a list of strings")
    allow_any = bool(data.get("allow_any", False))
    return RepoPolicy(patterns, allow_any=allow_any)

def load_settings() -> Settings:
    allowlist = json.loads(os.getenv("SOURCE_REPO_ALLOWLIST", "[]"))
    allow_any = os.getenv("ALLOW_ANY_SOURCE_REPO", "false").lower() in {"1", "true", "yes"}
    return Settings(
        db_path=os.getenv("CHATGPT_PLUGINS_DB", ".local/chatgpt-plugins.sqlite3"),
        public_base_url=os.getenv("CHATGPT_PLUGINS_PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
        accounts_file=os.getenv("KAGGLE_ACCOUNTS_FILE", "plugins/kaggle/config/accounts.example.json"),
        profiles_file=os.getenv("KAGGLE_PROFILES_FILE", "plugins/kaggle/config/profiles.example.json"),
        repo_policy=RepoPolicy(allowlist, allow_any=allow_any),
        github_repository=os.getenv("GITHUB_ORCHESTRATOR_REPOSITORY") or None,
        github_workflow=os.getenv("GITHUB_ORCHESTRATOR_WORKFLOW", "kaggle-dispatch.yml"),
        github_ref=os.getenv("GITHUB_ORCHESTRATOR_REF", "main"),
        github_token=os.getenv("GITHUB_ORCHESTRATOR_TOKEN") or None,
    )
