from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

_ACCOUNT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_ENV = re.compile(r"^[A-Z][A-Z0-9_]{1,119}$")


@dataclass(frozen=True, slots=True)
class GatewayAccount:
    account_id: str
    owner_slug: str
    username_env: str
    token_env: str
    enabled: bool = True

    def __post_init__(self) -> None:
        if not _ACCOUNT_ID.fullmatch(self.account_id):
            raise ValueError(f"invalid account_id: {self.account_id!r}")
        if not _OWNER.fullmatch(self.owner_slug):
            raise ValueError(f"invalid owner_slug for {self.account_id!r}")
        if not _ENV.fullmatch(self.username_env):
            raise ValueError(f"invalid username_env for {self.account_id!r}")
        if not _ENV.fullmatch(self.token_env):
            raise ValueError(f"invalid token_env for {self.account_id!r}")

    @property
    def credentials_configured(self) -> bool:
        username = os.getenv(self.username_env, "")
        token = os.getenv(self.token_env, "")
        return bool(username and token and not any(ch.isspace() for ch in token))

    def require_credentials(self) -> tuple[str, str]:
        username = os.getenv(self.username_env, "").strip()
        token = os.getenv(self.token_env, "")
        if not username:
            raise RuntimeError(f"Kaggle username is not configured for {self.account_id}")
        if not token:
            raise RuntimeError(f"Kaggle API token is not configured for {self.account_id}")
        if any(ch.isspace() for ch in token):
            raise RuntimeError(f"Kaggle API token contains whitespace for {self.account_id}")
        return username, token


@dataclass(frozen=True, slots=True)
class GatewayRegistry:
    accounts: tuple[GatewayAccount, ...]

    def __post_init__(self) -> None:
        ids = [account.account_id for account in self.accounts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate account_id in gateway registry")
        username_envs = [account.username_env for account in self.accounts]
        if len(username_envs) != len(set(username_envs)):
            raise ValueError("duplicate username_env in gateway registry")
        token_envs = [account.token_env for account in self.accounts]
        if len(token_envs) != len(set(token_envs)):
            raise ValueError("duplicate token_env in gateway registry")

    def get(self, account_id: str, *, require_enabled: bool = True) -> GatewayAccount:
        for account in self.accounts:
            if account.account_id == account_id:
                if require_enabled and not account.enabled:
                    raise ValueError(f"Kaggle account is disabled: {account_id}")
                return account
        raise ValueError(f"unknown Kaggle account: {account_id}")

    def public_view(self) -> list[dict[str, str | bool]]:
        return [
            {
                "account_id": account.account_id,
                "owner_slug": account.owner_slug,
                "enabled": account.enabled,
                "credentials_configured": account.credentials_configured,
            }
            for account in self.accounts
        ]


def default_registry_path() -> Path:
    override = os.getenv("KAGGLE_GATEWAY_ACCOUNTS_FILE")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).with_name("accounts.json")


def load_registry(path: str | Path | None = None) -> GatewayRegistry:
    source = Path(path).resolve() if path else default_registry_path()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("gateway account registry must be a JSON array")

    accounts: list[GatewayAccount] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each gateway account entry must be an object")
        allowed = {"account_id", "owner_slug", "username_env", "token_env", "enabled"}
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"unsupported gateway account fields: {sorted(unknown)}")
        accounts.append(
            GatewayAccount(
                account_id=str(item["account_id"]),
                owner_slug=str(item["owner_slug"]),
                username_env=str(item["username_env"]),
                token_env=str(item["token_env"]),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return GatewayRegistry(tuple(accounts))
