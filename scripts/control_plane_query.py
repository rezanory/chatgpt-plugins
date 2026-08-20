from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MAX_HTTP_BYTES = 2_000_000
MAX_RESULT_CHARS = 200_000
SAFE_PATH = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/?=-]{1,4000}$")
SECRET_KEY = re.compile(r"token|secret|password|authorization|credential|api[_-]?key|cookie|signed[_-]?url", re.I)
SECRET_VALUE = re.compile(
    r"KGAT_[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9._~-]+|Basic\s+[A-Za-z0-9+/=]+|"
    r"X-Goog-Signature=|X-Amz-Signature=",
    re.I,
)
ACCOUNTS = {
    "master": "azadka",
    "kg-02": "radlinaradlina",
    "kg-03": "rezanory",
    "kg-04": "reyhanehazad",
    "kg-05": "trickermark",
    "kg-06": "msdenis",
    "kg-07": "nisabulutmark",
}


class QueryError(RuntimeError):
    pass


def sanitize(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return "<depth-limit>"
    if isinstance(value, str):
        if SECRET_VALUE.search(value):
            return "<redacted>"
        return value[:120_000] + ("<truncated>" if len(value) > 120_000 else "")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        return [sanitize(item, depth + 1) for item in value[:2000]]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in list(value.items())[:2000]:
            text_key = str(key)
            result[text_key] = "<redacted>" if SECRET_KEY.search(text_key) else sanitize(child, depth + 1)
        return result
    return str(value)


def bounded(value: Any) -> Any:
    clean = sanitize(value)
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= MAX_RESULT_CHARS:
        return clean
    return {"truncated": True, "original_json_chars": len(encoded), "preview": encoded[:MAX_RESULT_CHARS]}


def safe_path(path: str) -> str:
    if not SAFE_PATH.fullmatch(path) or ".." in path or "://" in path or "\\" in path:
        raise QueryError("unsafe API path")
    return path


def http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read(MAX_HTTP_BYTES + 1)
            if len(raw) > MAX_HTTP_BYTES:
                raise QueryError("HTTP response exceeded bounded read limit")
            text = raw.decode("utf-8", errors="replace")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read(20_000).decode("utf-8", errors="replace")
        raise QueryError(f"HTTP {exc.code}: {sanitize(raw)}") from exc
    except urllib.error.URLError as exc:
        raise QueryError(f"HTTP transport error: {exc.reason}") from exc


def github_query(payload: dict[str, Any]) -> Any:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise QueryError("GITHUB_TOKEN unavailable")
    action = str(payload.get("action", "rest_get"))
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "chatgpt-control-plane-v3-query/1.0",
    }
    if action == "rest_get":
        path = safe_path(str(payload.get("path", "")))
        return http_json(f"https://api.github.com{path}", headers=headers)
    if action == "graphql_query":
        query = str(payload.get("query", "")).strip()
        if not query or query.lower().startswith("mutation") or " mutation " in f" {query.lower()} ":
            raise QueryError("GitHub GraphQL read broker accepts query operations only")
        variables = payload.get("variables", {})
        if not isinstance(variables, dict):
            raise QueryError("GitHub GraphQL variables must be an object")
        return http_json(
            "https://api.github.com/graphql",
            method="POST",
            headers={**headers, "Content-Type": "application/json"},
            body={"query": query, "variables": variables},
        )
    raise QueryError("unsupported GitHub read action")


def cloudflare_query(payload: dict[str, Any]) -> Any:
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not token:
        raise QueryError("CLOUDFLARE_API_TOKEN unavailable")
    action = str(payload.get("action", "rest_get"))
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "chatgpt-control-plane-v3-query/1.0",
    }
    if action == "rest_get":
        path = safe_path(str(payload.get("path", "")))
        if not path.startswith("/client/v4/"):
            raise QueryError("Cloudflare REST path must start with /client/v4/")
        return http_json(f"https://api.cloudflare.com{path}", headers=headers)
    if action == "graphql_query":
        query = str(payload.get("query", "")).strip()
        if not query or query.lower().startswith("mutation") or " mutation " in f" {query.lower()} ":
            raise QueryError("Cloudflare GraphQL broker accepts query operations only")
        variables = payload.get("variables", {})
        if not isinstance(variables, dict):
            raise QueryError("Cloudflare GraphQL variables must be an object")
        return http_json(
            "https://api.cloudflare.com/client/v4/graphql",
            method="POST",
            headers={**headers, "Content-Type": "application/json"},
            body={"query": query, "variables": variables},
        )
    raise QueryError("unsupported Cloudflare read action")


def worker_kaggle_read(payload: dict[str, Any]) -> Any:
    oidc = os.environ.get("CGP_OIDC_TOKEN", "").strip()
    base = os.environ.get(
        "CGP_CONTROL_PLANE_WORKER_BASE",
        "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev",
    ).rstrip("/")
    if not oidc:
        raise QueryError("GitHub OIDC token unavailable")
    action = str(payload.get("action", "raw_read"))
    request_payload = dict(payload)
    request_payload.pop("provider", None)
    if action == "kernel_status":
        account_id = str(payload.get("account_id", ""))
        owner = ACCOUNTS.get(account_id)
        kernel_ref = str(payload.get("kernel_ref", ""))
        if not owner or "/" not in kernel_ref:
            raise QueryError("kernel_status requires valid account_id and kernel_ref")
        ref_owner, slug = kernel_ref.split("/", 1)
        if ref_owner.lower() != owner.lower() or not slug:
            raise QueryError("kernel_ref owner does not match account_id")
        request_payload = {
            "action": "raw_read",
            "account_id": account_id,
            "service": "kernels.KernelsApiService",
            "method": "GetKernelSessionStatus",
            "body": {"userName": owner, "kernelSlug": slug},
        }
    return http_json(
        f"{base}/control-plane/v3/read/kaggle",
        method="POST",
        headers={
            "Authorization": f"Bearer {oidc}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "chatgpt-control-plane-v3-query/1.0",
        },
        body=request_payload,
    )


def local_kaggle_kernel_status(payload: dict[str, Any]) -> Any:
    executable = shutil.which("kaggle") or shutil.which("kaggle.exe")
    if not executable:
        raise QueryError("local Kaggle CLI unavailable")
    kernel_ref = str(payload.get("kernel_ref", ""))
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", kernel_ref):
        raise QueryError("invalid kernel_ref")
    completed = subprocess.run(
        [executable, "kernels", "status", kernel_ref],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        raise QueryError(f"local Kaggle status failed: {sanitize(output.strip())}")
    return {"transport": "local_kaggle_cli", "kernel_ref": kernel_ref, "output": output.strip()}


def kaggle_query(payload: dict[str, Any]) -> Any:
    action = str(payload.get("action", "kernel_status"))
    try:
        return worker_kaggle_read(payload)
    except QueryError as worker_error:
        if action != "kernel_status":
            raise
        try:
            local = local_kaggle_kernel_status(payload)
            return {"worker_read_error": str(worker_error), "fallback": local}
        except QueryError as local_error:
            raise QueryError(f"Kaggle read unavailable: worker={worker_error}; local={local_error}") from local_error


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider", "")).lower()
    request_id = str(payload.get("request_id", ""))[:120]
    if provider == "github":
        result = github_query(payload)
    elif provider == "cloudflare":
        result = cloudflare_query(payload)
    elif provider == "kaggle":
        result = kaggle_query(payload)
    else:
        raise QueryError("provider must be github, cloudflare, or kaggle")
    return {
        "ok": True,
        "request_id": request_id,
        "provider": provider,
        "read_only": True,
        "result": bounded(result),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QueryError("query payload must be an object")
    try:
        result = execute(payload)
    except Exception as exc:  # noqa: BLE001 - broker must emit bounded structured diagnostics
        result = {
            "ok": False,
            "request_id": str(payload.get("request_id", ""))[:120],
            "provider": str(payload.get("provider", ""))[:40],
            "read_only": True,
            "error": str(sanitize(str(exc)))[:4000],
        }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": result.get("ok"), "provider": result.get("provider"), "out": args.out}))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
