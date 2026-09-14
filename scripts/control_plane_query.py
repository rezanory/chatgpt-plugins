from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MAX_HTTP_BYTES = 2_000_000
MAX_RESULT_CHARS = 200_000
SAFE_PATH = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/?={}-]{1,4000}$")
SECRET_KEY = re.compile(
    r"token|secret|password|authorization|credential|api[_-]?key|cookie|signed[_-]?url",
    re.I,
)
SECRET_VALUE = re.compile(
    r"KGAT_[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9._~-]+|Basic\s+[A-Za-z0-9+/=]+|"
    r"X-Goog-Signature=|X-Amz-Signature=|https://www\.kaggleusercontent\.com/kf/\S+",
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
    "kg-08": "azadkk",
    "kg-09": "mylovevpn1",
    "kg-10": "computstu1",
    "kg-11": "jobreza1",
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
            result[text_key] = (
                "<redacted>"
                if SECRET_KEY.search(text_key)
                else sanitize(child, depth + 1)
            )
        return result
    return str(value)


def bounded(value: Any) -> Any:
    clean = sanitize(value)
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= MAX_RESULT_CHARS:
        return clean
    return {
        "truncated": True,
        "original_json_chars": len(encoded),
        "preview": encoded[:MAX_RESULT_CHARS],
    }


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
        lowered = f" {query.lower()} "
        if not query or query.lower().startswith("mutation") or " mutation " in lowered:
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
        path = str(payload.get("path", ""))
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        if "{account_id}" in path:
            if not account_id:
                raise QueryError("CLOUDFLARE_ACCOUNT_ID unavailable for placeholder expansion")
            path = path.replace("{account_id}", account_id)
        path = safe_path(path)
        if not path.startswith("/client/v4/"):
            raise QueryError("Cloudflare REST path must start with /client/v4/")
        return http_json(f"https://api.cloudflare.com{path}", headers=headers)
    if action == "graphql_query":
        query = str(payload.get("query", "")).strip()
        lowered = f" {query.lower()} "
        if not query or query.lower().startswith("mutation") or " mutation " in lowered:
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
    request_payload.pop("request_id", None)
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


def _kaggle_kernels(account_id: str) -> list[dict[str, Any]]:
    owner = ACCOUNTS.get(account_id)
    if not owner:
        raise QueryError("unknown Kaggle account_id")
    response = worker_kaggle_read(
        {
            "provider": "kaggle",
            "action": "raw_read",
            "account_id": account_id,
            "service": "kernels.KernelsApiService",
            "method": "ListKernels",
            "body": {
                "group": "PROFILE",
                "user": owner,
                "sortBy": "DATE_RUN",
                "page": 1,
                "pageSize": 100,
            },
        }
    )
    if not isinstance(response, dict) or not response.get("ok"):
        raise QueryError("Kaggle ListKernels read did not return ok=true")
    result = response.get("result")
    if not isinstance(result, dict):
        raise QueryError("Kaggle ListKernels result missing")
    kernels = result.get("kernels")
    if not isinstance(kernels, list):
        raise QueryError("Kaggle ListKernels kernels list missing")
    return [item for item in kernels if isinstance(item, dict)]


def resolve_kaggle_kernel_ref(payload: dict[str, Any]) -> dict[str, Any]:
    account_id = str(payload.get("account_id", ""))
    owner = ACCOUNTS.get(account_id)
    requested_ref = str(payload.get("kernel_ref", "")).strip()
    if not owner or "/" not in requested_ref:
        raise QueryError("kernel resolver requires valid account_id and kernel_ref")
    requested_owner, requested_slug = requested_ref.split("/", 1)
    if requested_owner.lower() != owner.lower() or not requested_slug:
        raise QueryError("kernel_ref owner does not match account_id")

    kernels = _kaggle_kernels(account_id)
    exact = [item for item in kernels if str(item.get("ref", "")).lower() == requested_ref.lower()]

    title = str(payload.get("kernel_title", "")).strip()
    if not title and requested_slug == "m07-gate-224-pkg-v1":
        title = f"M07 Gate 224 Package - {owner}"
    title_matches: list[dict[str, Any]] = []
    if not exact and title:
        title_matches = [item for item in kernels if str(item.get("title", "")).strip().lower() == title.lower()]

    candidates = exact or title_matches
    not_before = str(payload.get("not_before", "")).strip()
    if not_before:
        candidates = [item for item in candidates if str(item.get("lastRunTime", "")) >= not_before]

    if not candidates:
        recent = [
            {"ref": str(item.get("ref", "")), "title": str(item.get("title", "")), "lastRunTime": str(item.get("lastRunTime", ""))}
            for item in kernels[:8]
        ]
        raise QueryError(
            "CURRENT_PACKAGE_NOT_FOUND: no exact Kaggle kernel object matched "
            f"requested_ref={requested_ref!r}, title={title!r}, not_before={not_before!r}; "
            f"recent_candidates={json.dumps(recent, ensure_ascii=False, separators=(',', ':'))}"
        )

    candidates.sort(key=lambda item: str(item.get("lastRunTime", "")), reverse=True)
    chosen = candidates[0]
    resolved_ref = str(chosen.get("ref", "")).strip()
    if "/" not in resolved_ref:
        raise QueryError("resolved Kaggle kernel ref invalid")
    return {
        "requested_ref": requested_ref,
        "resolved_ref": resolved_ref,
        "matched_by": "exact_ref" if exact else "exact_title",
        "title": str(chosen.get("title", "")),
        "lastRunTime": str(chosen.get("lastRunTime", "")),
    }


def resolved_kaggle_telemetry(payload: dict[str, Any], target_action: str) -> Any:
    resolution = resolve_kaggle_kernel_ref(payload)
    forwarded = dict(payload)
    forwarded["action"] = target_action
    forwarded["kernel_ref"] = resolution["resolved_ref"]
    forwarded.pop("kernel_title", None)
    forwarded.pop("not_before", None)
    result = worker_kaggle_read(forwarded)
    return {"resolution": resolution, "telemetry": result}


def _fetch_output_json_url(url: str, max_bytes: int) -> Any:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "www.kaggleusercontent.com":
        raise QueryError("Kaggle output file URL host is not allowed")
    request = urllib.request.Request(
        url, method="GET",
        headers={"User-Agent": "chatgpt-control-plane-v3-query/1.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise QueryError("Kaggle output JSON exceeded bounded file limit")
            if response.status != 200:
                raise QueryError(f"Kaggle output file HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise QueryError(f"Kaggle output file HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise QueryError(f"Kaggle output file transport error: {exc.reason}") from exc
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QueryError("Kaggle output file is not valid UTF-8 JSON") from exc
    if not isinstance(value, (dict, list)):
        raise QueryError("Kaggle output JSON must be an object or array")
    return value

def kaggle_output_json_files(payload: dict[str, Any]) -> Any:
    account_id = str(payload.get("account_id", ""))
    owner = ACCOUNTS.get(account_id)
    kernel_ref = str(payload.get("kernel_ref", "")).strip()
    if not owner or "/" not in kernel_ref:
        raise QueryError("kernel_output_json_files requires valid account_id and kernel_ref")
    ref_owner, slug = kernel_ref.split("/", 1)
    if ref_owner.lower() != owner.lower() or not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", slug):
        raise QueryError("kernel_ref owner/slug does not match account_id")
    names = payload.get("file_names")
    if not isinstance(names, list) or not 1 <= len(names) <= 10:
        raise QueryError("file_names must contain between 1 and 10 exact JSON paths")
    exact_names: list[str] = []
    for item in names:
        name = str(item).strip()
        if (not name.endswith(".json") or len(name) > 500 or name.startswith("/")
                or ".." in name or "\\" in name
                or not re.fullmatch(r"[A-Za-z0-9._/-]+", name)):
            raise QueryError("file_names contains an unsafe or non-JSON path")
        if name in exact_names:
            raise QueryError("file_names must be unique")
        exact_names.append(name)
    max_bytes = int(payload.get("max_bytes_per_file", 65536))
    if max_bytes < 1024 or max_bytes > 262144:
        raise QueryError("max_bytes_per_file must be between 1024 and 262144")
    by_name: dict[str, dict[str, Any]] = {}
    previous_signature: tuple[str, ...] | None = None
    for page in range(1, 21):
        response = worker_kaggle_read({
            "provider": "kaggle", "action": "raw_read", "account_id": account_id,
            "service": "kernels.KernelsApiService", "method": "ListKernelSessionOutput",
            "body": {"userName": owner, "kernelSlug": slug, "page": page, "pageSize": 10},
        })
        if not isinstance(response, dict) or not response.get("ok"):
            raise QueryError("ListKernelSessionOutput did not return ok=true")
        result = response.get("result")
        if not isinstance(result, dict):
            raise QueryError("ListKernelSessionOutput result missing")
        raw_files = result.get("files")
        if not isinstance(raw_files, list):
            raise QueryError("ListKernelSessionOutput files list missing")
        signature = tuple(str(item.get("fileName", "")) for item in raw_files if isinstance(item, dict))
        if page > 1 and signature and signature == previous_signature:
            raise QueryError("ListKernelSessionOutput pagination did not advance")
        previous_signature = signature
        for item in raw_files:
            if isinstance(item, dict) and item.get("fileName"):
                by_name[str(item["fileName"])] = item
        if all(name in by_name for name in exact_names):
            break
        if len(raw_files) < 10:
            break
    else:
        raise QueryError("Kaggle output pagination exceeded bounded page limit")
    outputs = []
    for name in exact_names:
        item = by_name.get(name)
        if item is None:
            raise QueryError(f"requested Kaggle output JSON not found: {name}")
        url = str(item.get("url", ""))
        if not url:
            raise QueryError(f"requested Kaggle output JSON has no download URL: {name}")
        outputs.append({"file_name": name, "json": _fetch_output_json_url(url, max_bytes)})
    return {
        "transport": "cloudflare_kaggle_api_then_bounded_output_fetch",
        "account_id": account_id,
        "kernel_ref": kernel_ref,
        "file_count": len(outputs),
        "files": outputs,
        "signed_urls_returned": False,
    }

def resolve_kaggle_executable() -> str | None:
    executable = shutil.which("kaggle") or shutil.which("kaggle.exe")
    if executable:
        return executable
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidate = Path(scripts) / "kaggle.exe"
        if candidate.exists():
            return str(candidate)
    user_scripts = Path(os.environ.get("APPDATA", "")) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "kaggle.exe"
    return str(user_scripts) if user_scripts.exists() else None


def local_kaggle_kernel_status(payload: dict[str, Any]) -> Any:
    executable = resolve_kaggle_executable()
    if not executable:
        raise QueryError("local Kaggle CLI unavailable")
    kernel_ref = str(payload.get("kernel_ref", ""))
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", kernel_ref):
        raise QueryError("invalid kernel_ref")
    completed = subprocess.run([executable, "kernels", "status", kernel_ref], capture_output=True, text=True, timeout=90, check=False)
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        raise QueryError(f"local Kaggle status failed: {sanitize(output.strip())}")
    return {"transport": "local_kaggle_cli", "kernel_ref": kernel_ref, "output": output.strip()}


def kaggle_query(payload: dict[str, Any]) -> Any:
    action = str(payload.get("action", "kernel_status"))
    if action == "kernel_output_json_files":
        return kaggle_output_json_files(payload)
    if action in {"live_log", "kernel_status", "phase_probe"}:
        return resolved_kaggle_telemetry(payload, action)
    if action == "resolved_live_log":
        return resolved_kaggle_telemetry(payload, "live_log")
    if action == "resolved_kernel_status":
        return resolved_kaggle_telemetry(payload, "kernel_status")
    if action == "resolved_phase_probe":
        return resolved_kaggle_telemetry(payload, "phase_probe")
    return worker_kaggle_read(payload)


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
    return {"ok": True, "request_id": request_id, "provider": provider, "read_only": True, "result": bounded(result)}


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
    except Exception as exc:  # noqa: BLE001 - broker emits bounded structured diagnostics
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
