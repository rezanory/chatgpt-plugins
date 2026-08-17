from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from .config import GatewayAccount, GatewayRegistry

# These global Kaggle variables would override per-account config during api.authenticate().
# The gateway deliberately uses CGP_KAGGLE_* variables instead.
_CONFLICTING_GLOBAL_AUTH = (
    "KAGGLE_API_TOKEN",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
)


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
    api: KaggleApi
    config_dir: Path
    lock: threading.RLock


class KaggleApiPool:
    """One isolated KaggleApi instance per configured account.

    The exact authentication flow is intentionally the one proven by the operator:

      api = KaggleApi()
      api.set_config_value(api.CONFIG_NAME_USER, username)
      api.set_config_value(api.CONFIG_NAME_KEY, token)
      api.authenticate()
      api.kernels_list(page_size=1)

    KaggleApi.set_config_value writes to api.config. We therefore override api.config_dir/api.config
    per account before injecting credentials, preventing six concurrent clients from racing on the
    process-wide default ~/.kaggle/kaggle.json file.
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
        self._closed = False
        self._pool_lock = threading.RLock()

    def _build_slot(self, account: GatewayAccount) -> _ClientSlot:
        username, token = account.require_credentials()
        config_dir = Path(tempfile.mkdtemp(prefix=f"cgp-kaggle-{account.account_id}-"))
        try:
            if os.name != "nt":
                config_dir.chmod(0o700)

            api = KaggleApi()
            # KaggleApi's defaults are class attributes; make every mutable credential/config
            # field instance-local before set_config_value() is called.
            api.config_dir = str(config_dir)
            api.config_file = "kaggle.json"
            api.config = str(config_dir / api.config_file)
            api.config_values = {}
            api._authenticated = False

            api.set_config_value(api.CONFIG_NAME_USER, username, quiet=True)
            api.set_config_value(api.CONFIG_NAME_KEY, token, quiet=True)
            api.authenticate()

            # Same harmless health probe that already succeeded for the user's six accounts.
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
        if self._closed:
            raise RuntimeError("KaggleApiPool is closed")
        account = self._registry.get(account_id)
        with self._pool_lock:
            slot = self._slots.get(account_id)
            if slot is None:
                slot = self._build_slot(account)
                self._slots[account_id] = slot
            return slot

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

    def auth_check_all(self) -> list[dict[str, Any]]:
        return [
            self.auth_check(account.account_id)
            for account in self._registry.accounts
            if account.enabled
        ]

    def kernels_list(
        self,
        account_id: str,
        *,
        search: str | None = None,
        page_size: int = 20,
        mine: bool = True,
        sort_by: str = "dateRun",
    ) -> list[Any]:
        if not 1 <= page_size <= 200:
            raise ValueError("page_size must be between 1 and 200")
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
            method = getattr(slot.api, "kernels_status", None) or getattr(
                slot.api, "kernel_status", None
            )
            if method is None:
                raise RuntimeError("installed KaggleApi does not expose a kernel status method")
            result = method(kernel_ref)
        return _jsonable(result)

    def kernels_output(
        self,
        account_id: str,
        kernel_ref: str,
        path: str,
        *,
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
