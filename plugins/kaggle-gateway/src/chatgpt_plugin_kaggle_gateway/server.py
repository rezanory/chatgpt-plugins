from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mcp.server import MCPServer
from mcp_types import ToolAnnotations

from .api_pool import KaggleApiPool
from .artifacts import build_output_manifest
from .config import load_registry

_mcp = MCPServer(
    "Kaggle Direct Gateway",
    instructions=(
        "Direct multi-account Kaggle gateway. Each logical account uses its own isolated "
        "KaggleApi instance authenticated with username + API token. No Kaggle CLI, browser "
        "session, or GitHub Actions runtime relay is used for Kaggle API calls."
    ),
)
_pool_instance: KaggleApiPool | None = None
_pool_lock = threading.Lock()
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


def _pool() -> KaggleApiPool:
    global _pool_instance
    if _pool_instance is not None:
        return _pool_instance
    with _pool_lock:
        if _pool_instance is None:
            _pool_instance = KaggleApiPool(load_registry())
    return _pool_instance


def _close_pool() -> None:
    global _pool_instance
    with _pool_lock:
        pool = _pool_instance
        _pool_instance = None
    if pool is not None:
        pool.close()


atexit.register(_close_pool)


@_mcp.tool(annotations=_READ_ONLY)
def kaggle_accounts() -> list[dict[str, str | bool]]:
    """List configured logical Kaggle accounts without exposing credentials."""
    return load_registry().public_view()


@_mcp.tool(annotations=_READ_ONLY)
def kaggle_auth_check(account_id: str) -> dict[str, Any]:
    """Authenticate one account and verify it with KaggleApi.kernels_list(page_size=1)."""
    return _pool().auth_check(account_id)


@_mcp.tool(annotations=_READ_ONLY)
def kaggle_auth_check_all(max_workers: int = 6) -> list[dict[str, Any]]:
    """Verify all enabled Kaggle accounts in parallel using isolated KaggleApi instances."""
    return _pool().auth_check_all(max_workers=max_workers)


@_mcp.tool(annotations=_READ_ONLY)
def kaggle_kernels_list(
    account_id: str,
    search: str = "",
    page_size: int = 20,
) -> list[Any]:
    """List kernels for one authenticated account through the direct Python API."""
    return _pool().kernels_list(
        account_id,
        search=search or None,
        page_size=page_size,
        mine=True,
        sort_by="dateRun",
    )


@_mcp.tool(annotations=_READ_ONLY)
def kaggle_kernels_inventory_all(
    search: str,
    page_size: int = 20,
    max_workers: int = 6,
) -> list[dict[str, Any]]:
    """Search each enabled account's own kernels in parallel without submitting new compute."""
    registry = load_registry()
    enabled = [account for account in registry.accounts if account.enabled]
    workers = max(1, min(max_workers, len(enabled), 16)) if enabled else 1
    results: dict[str, dict[str, Any]] = {}
    pool = _pool()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kaggle-inventory") as executor:
        futures = {
            executor.submit(
                pool.kernels_list,
                account.account_id,
                search=search or None,
                page_size=page_size,
                mine=True,
                sort_by="dateRun",
            ): account.account_id
            for account in enabled
        }
        for future in as_completed(futures):
            account_id = futures[future]
            try:
                results[account_id] = {
                    "account_id": account_id,
                    "ok": True,
                    "kernels": future.result(),
                }
            except BaseException as exc:
                results[account_id] = {
                    "account_id": account_id,
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error": pool.safe_error(account_id, exc),
                }

    return [results[account.account_id] for account in enabled]


@_mcp.tool(annotations=_READ_ONLY)
def kaggle_kernel_status(account_id: str, kernel_ref: str) -> Any:
    """Get the latest run status for an existing owner/kernel through the direct Python API."""
    return _pool().kernels_status(account_id, kernel_ref)


@_mcp.tool(annotations=_READ_ONLY)
def kaggle_kernel_logs(account_id: str, kernel_ref: str) -> str:
    """Read the execution log for an existing owner/kernel through the direct Python API."""
    return _pool().kernels_logs(account_id, kernel_ref)


@_mcp.tool(annotations=_READ_ONLY)
def kaggle_kernel_output_manifest(
    account_id: str,
    kernel_ref: str,
    artifact_names: list[str],
    expected_fingerprint: str = "",
) -> dict[str, Any]:
    """Hash selected existing outputs and report exact fingerprint hits without retaining files."""
    return build_output_manifest(
        _pool(),
        account_id,
        kernel_ref,
        artifact_names,
        expected_fingerprint=expected_fingerprint or None,
    )


def main() -> None:
    host = os.getenv("CGP_GATEWAY_HOST", "127.0.0.1")
    port = int(os.getenv("CGP_GATEWAY_PORT", "8000"))
    if host not in {"127.0.0.1", "localhost", "::1"} and os.getenv(
        "CGP_GATEWAY_ALLOW_UNAUTHENTICATED_REMOTE"
    ) != "1":
        raise RuntimeError(
            "refusing unauthenticated non-loopback MCP binding; keep the gateway local/private "
            "or place it behind an authenticated tunnel/proxy"
        )
    _mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
