from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import GatewayAccount, GatewayRegistry

# KaggleApi.authenticate() reads KAGGLE_* environment variables after the config file. Global
# credentials would therefore override an account's isolated username/key. The gateway stores
# account secrets under CGP_KAGGLE_* instead and rejects conflicting globals at startup.
_CONFLICTING_GLOBAL_AUTH = (
    "KAGGLE_API_TOKEN",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
)


def _kaggle_api_class():
    # Importing the kaggle package creates a module-level API instance and performs a best-effort
    # authenticate(). Suppress only that import-time helper output; our per-account instances below
    # authenticate explicitly and errors from those instances are never suppressed.
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        from kaggle.api.kaggle_api_extended import KaggleApi

    return KaggleApi


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


@dataclass(slots=True)
class _ClientSlot:
    account: GatewayAccount
    api: Any
    config_dir: Path
    lock: threading.RLock


class KaggleApiPool:
    """One isolated KaggleApi instance per configured account.

    Authentication intentionally follows the operator-proven sequence exactly::

        api = KaggleApi()
        api.set_config_value(api.CONFIG_NAME_USER, username)
        api.set_config_value(api.CONFIG_NAME_KEY, token)
        api.authenticate()
        api.kernels_list(page_size=1)

    `set_config_value()` writes to `api.config`, while KaggleApi's defaults are class-level values.
    Each account therefore receives an instance-local temporary config directory and config file
    before credentials are injected. This preserves the proven authentication method while making
    six-account parallel operation safe from shared `~/.kaggle/kaggle.json` overwrite races.
    """

    def __init__(self, registry: GatewayRegistry) -> None:
        polluted = [name for name in _CONFLICTING_GLOBAL_AUTH if os.getenv(name)]
        if polluted:
            names = ", ".join(polluted)
            raise RuntimeError(
                "global Kaggle authentication variables are forbidden in multi-account gateway: "
                f"{names}"
            )
        self._registry = registry
        self._slots: dict[str, _ClientSlot] = {}
        self._creation_locks = {
            account.account_id: threading.Lock() for account in registry.accounts
        }
        self._closed = False
        self._pool_lock = threading.RLock()

    def _build_slot(self, account: GatewayAccount) -> _ClientSlot:
        username, token = account.require_credentials()
        config_dir = Path(tempfile.mkdtemp(prefix=f"cgp-kaggle-{account.account_id}-"))
        try:
            if os.name != "nt":
                config_dir.chmod(0o700)

            KaggleApi = _kaggle_api_class()
            api = KaggleApi()

            # Make every mutable auth/config field instance-local before set_config_value().
            api.config_dir = str(config_dir)
            api.config_file = "kaggle.json"
            api.config = str(config_dir / api.config_file)
            api.config_values = {}
            api._authenticated = False

            # Proven direct Python API authentication flow supplied by the operator.
            api.set_config_value(api.CONFIG_NAME_USER, username, quiet=True)
            api.set_config_value(api.CONFIG_NAME_KEY, token, quiet=True)
            config_path = Path(api.config)
            if os.name != "nt" and config_path.exists():
                config_path.chmod(0o600)
            api.authenticate()

            # Same harmless probe already observed as auth_ok on all six active accounts.
            api.kernels_list(page_size=1)
            return _ClientSlot(
                account=account,
                api=api,
                config_dir=config_dir,
                lock=threading.RLock(),
            )
        except BaseException:
            shutil.rmtree(config_dir, ignore_errors=True)
            raise

    def _slot(self, account_id: str) -> _ClientSlot:
        account = self._registry.get(account_id)
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("KaggleApiPool is closed")
            existing = self._slots.get(account_id)
            if existing is not None:
                return existing

        # Serialize creation only for the same logical account. Different accounts authenticate
        # concurrently and therefore preserve the desired six-way parallel connection behavior.
        creation_lock = self._creation_locks[account_id]
        with creation_lock:
            with self._pool_lock:
                if self._closed:
                    raise RuntimeError("KaggleApiPool is closed")
                existing = self._slots.get(account_id)
                if existing is not None:
                    return existing

            slot = self._build_slot(account)
            with self._pool_lock:
                if self._closed:
                    shutil.rmtree(slot.config_dir, ignore_errors=True)
                    raise RuntimeError("KaggleApiPool was closed during account initialization")
                self._slots[account_id] = slot
            return slot

    def _safe_error(self, account_id: str, exc: Exception) -> str:
        text = str(exc)
        account = self._registry.get(account_id, require_enabled=False)
        token = os.getenv(account.token_env, "")
        if token:
            text = text.replace(token, "[REDACTED]")
        return text[:1000]

    def auth_check(self, account_id: str) -> dict[str, Any]:
        slot = self._slot(account_id)
        with slot.lock:
            kernels = slot.api.kernels_list(page_size=1)
        return {
            "account_id": account_id,
            "owner_slug": slot.account.owner_slug,
            "auth_ok": True,
            "probe_count": len(kernels or []),
        }

    def auth_check_all(self, *, max_workers: int | None = None) -> list[dict[str, Any]]:
        enabled = [account for account in self._registry.accounts if account.enabled]
        if not enabled:
            return []
        workers = max_workers or len(enabled)
        workers = max(1, min(workers, len(enabled), 16))
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kaggle-auth") as executor:
            futures = {
                executor.submit(self.auth_check, account.account_id): account.account_id
                for account in enabled
            }
            for future in as_completed(futures):
                account_id = futures[future]
                try:
                    results[account_id] = future.result()
                except Exception as exc:
                    results[account_id] = {
                        "account_id": account_id,
                        "auth_ok": False,
                        "error_type": exc.__class__.__name__,
                        "error": self._safe_error(account_id, exc),
                    }
        return [results[account.account_id] for account in enabled]

    def kernels_list(
        self,
        account_id: str,
        *,
        search: str | None = None,
        page_size: int = 20,
        mine: bool = True,
        sort_by: str = "dateRun",
    ) -> list[Any]:
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        slot = self._slot(account_id)
        with slot.lock:
            values = slot.api.kernels_list(
                search=search,
                page_size=page_size,
                mine=mine,
                sort_by=sort_by,
            )
        return _jsonable(values or [])

    def kernels_status(self, account_id: str, kernel_ref: str) -> Any:
        self._validate_kernel_owner(account_id, kernel_ref)
        slot = self._slot(account_id)
        with slot.lock:
            result = slot.api.kernels_status(kernel_ref)
        return _jsonable(result)

    def kernels_logs(self, account_id: str, kernel_ref: str) -> str:
        self._validate_kernel_owner(account_id, kernel_ref)
        slot = self._slot(account_id)
        with slot.lock:
            value = slot.api.kernels_logs(kernel_ref)
        return str(value or "")[:200_000]

    def kernels_output(
        self,
        account_id: str,
        kernel_ref: str,
        path: str,
        *,
        file_pattern: str | None = None,
        force: bool = False,
        quiet: bool = True,
    ) -> Any:
        self._validate_kernel_owner(account_id, kernel_ref)
        slot = self._slot(account_id)
        output_path = Path(path).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        with slot.lock:
            result = slot.api.kernels_output(
                kernel_ref,
                path=str(output_path),
                file_pattern=file_pattern,
                force=force,
                quiet=quiet,
            )
        return _jsonable(result)

    def kernels_push(self, account_id: str, folder: str) -> Any:
        slot = self._slot(account_id)
        folder_path = Path(folder).expanduser().resolve()
        if not folder_path.is_dir():
            raise ValueError(f"kernel folder does not exist: {folder_path}")
        with slot.lock:
            result = slot.api.kernels_push(str(folder_path))
        return _jsonable(result)

    def _validate_kernel_owner(self, account_id: str, kernel_ref: str) -> None:
        account = self._registry.get(account_id)
        parts = kernel_ref.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError("kernel_ref must use owner/slug form")
        if parts[0].casefold() != account.owner_slug.casefold():
            raise ValueError(
                f"kernel owner {parts[0]!r} does not match account {account_id!r}"
            )

    def close(self) -> None:
        with self._pool_lock:
            if self._closed:
                return
            self._closed = True
            slots = list(self._slots.values())
            self._slots.clear()
        for slot in slots:
            shutil.rmtree(slot.config_dir, ignore_errors=True)

    def __enter__(self) -> KaggleApiPool:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
