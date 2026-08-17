from __future__ import annotations

import atexit
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mcp.server import MCPServer
from mcp_types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

from .api_pool import KaggleApiPool
from .artifacts import build_output_manifest
from .config import load_registry
from .control import ControlCommand, ControlDispatcher, verify_github_signature

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
_dispatcher_instance: ControlDispatcher | None = None
_dispatcher_lock = threading.Lock()
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


def _dispatcher() -> ControlDispatcher:
    global _dispatcher_instance
    if _dispatcher_instance is not None:
        return _dispatcher_instance
    with _dispatcher_lock:
        if _dispatcher_instance is None:
            _dispatcher_instance = ControlDispatcher(_pool())
    return _dispatcher_instance


def _close_pool() -> None:
    global _pool_instance
    with _pool_lock:
        pool = _pool_instance
        _pool_instance = None
    if pool is not None:
        pool.close()


atexit.register(_close_pool)


@_mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request: Request) -> JSONResponse:
    """Public liveness endpoint for the free hosting platform."""
    return JSONResponse({"service": "chatgpt-kaggle-gateway", "status": "ready"})


@_mcp.custom_route("/github/webhook", methods=["POST"])
async def github_webhook(request: Request) -> JSONResponse:
    """Accept signed GitHub Issue events and dispatch direct KaggleApi writes."""
    body = await request.body()
    secret = os.getenv("CGP_GITHUB_WEBHOOK_SECRET", "")
    signature = request.headers.get("x-hub-signature-256", "")
    if not verify_github_signature(secret, body, signature):
        return JSONResponse({"accepted": False, "reason": "invalid_signature"}, status_code=401)

    event = request.headers.get("x-github-event", "")
    if event != "issues":
        return JSONResponse({"accepted": False, "reason": "ignored_event"}, status_code=202)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse({"accepted": False, "reason": "invalid_json"}, status_code=400)

    if payload.get("action") != "opened":
        return JSONResponse({"accepted": False, "reason": "ignored_action"}, status_code=202)

    expected_repo = os.getenv("CGP_CONTROL_REPOSITORY", "rezanory/chatgpt-plugins")
    repository = str((payload.get("repository") or {}).get("full_name") or "")
    if repository != expected_repo:
        return JSONResponse({"accepted": False, "reason": "wrong_repository"}, status_code=403)

    issue = payload.get("issue") or {}
    actor = str((issue.get("user") or {}).get("login") or "")
    allowed = {
        item.strip().casefold()
        for item in os.getenv("CGP_GITHUB_ALLOWED_ACTORS", "rezanory").split(",")
        if item.strip()
    }
    if actor.casefold() not in allowed:
        return JSONResponse({"accepted": False, "reason": "actor_not_allowed"}, status_code=403)

    try:
        issue_number = int(issue["number"])
        command = ControlCommand.from_issue(str(issue.get("title") or ""), str(issue.get("body") or ""))
        accepted = _dispatcher().accept(issue_number, command)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return JSONResponse(
            {"accepted": False, "reason": exc.__class__.__name__, "detail": str(exc)[:500]},
            status_code=400,
        )

    return JSONResponse(
        {"accepted": accepted, "job_id": command.job_id, "duplicate": not accepted},
        status_code=202,
    )


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


def _mcp_path_secret() -> tuple[str, bool]:
    secret = os.getenv("CGP_MCP_PATH_SECRET", "").strip()
    if not secret:
        return "/mcp", False
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return f"/mcp/{digest}", True


def main() -> None:
    host = os.getenv("CGP_GATEWAY_HOST", "127.0.0.1")
    port = int(os.getenv("CGP_GATEWAY_PORT") or os.getenv("PORT") or "8000")
    mcp_path, has_path_secret = _mcp_path_secret()
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    render_runtime = os.getenv("RENDER", "").lower() == "true"
    explicit_unsafe_override = os.getenv("CGP_GATEWAY_ALLOW_UNAUTHENTICATED_REMOTE") == "1"

    if render_runtime and not has_path_secret:
        raise RuntimeError("Render runtime requires CGP_MCP_PATH_SECRET")

    protected_remote = render_runtime and has_path_secret
    if not loopback and not protected_remote and not explicit_unsafe_override:
        raise RuntimeError(
            "refusing non-loopback MCP binding without a protected hosted-runtime boundary"
        )

    if render_runtime:
        print(f"CGP_MCP_ENDPOINT_PATH={mcp_path}", flush=True)

    _mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        streamable_http_path=mcp_path,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
