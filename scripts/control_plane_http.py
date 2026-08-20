#!/usr/bin/env python3
"""Fixed-host REST/GraphQL fallback for GitHub and Cloudflare.

This is not a general-purpose HTTP client. Hosts are fixed by provider, credential values are read
only from environment variables, and non-read operations require an explicit safety-class grant.
A POST classified as read is accepted only for the provider's GraphQL endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


SENSITIVE_KEY = re.compile(
    r"(token|secret|password|passwd|private[_-]?key|access[_-]?key|authorization|cookie)",
    re.I,
)
SAFETY_CLASSES = {"read", "write", "compute", "destructive", "privileged"}
GRAPHQL_PATH = {"github": "/graphql", "cloudflare": "/graphql"}


class ControlPlaneHttpError(RuntimeError):
    pass


def allowed_classes() -> set[str]:
    raw = os.environ.get("CONTROL_PLANE_ALLOWED_CLASSES", "read")
    values = {item.strip().lower() for item in re.split(r"[\s,]+", raw) if item.strip()}
    unknown = values - SAFETY_CLASSES
    if unknown:
        raise ControlPlaneHttpError(f"unknown allowed safety classes: {sorted(unknown)}")
    return values


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if SENSITIVE_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def provider_config(provider: str) -> tuple[str, dict[str, str]]:
    if provider == "github":
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            raise ControlPlaneHttpError("GITHUB_TOKEN is not configured")
        return (
            "https://api.github.com",
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2026-03-10",
                "User-Agent": "chatgpt-control-plane-v3/1.0",
            },
        )
    if provider == "cloudflare":
        token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
        if not token:
            raise ControlPlaneHttpError("CLOUDFLARE_API_TOKEN is not configured")
        return (
            "https://api.cloudflare.com/client/v4",
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": "chatgpt-control-plane-v3/1.0",
            },
        )
    raise ControlPlaneHttpError(f"unsupported provider: {provider}")


def safe_relative_path(path: str) -> str:
    value = path.strip()
    if not value.startswith("/"):
        value = "/" + value
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme or parsed.netloc:
        raise ControlPlaneHttpError("absolute URLs are forbidden")
    if ".." in parsed.path.split("/"):
        raise ControlPlaneHttpError("parent path traversal is forbidden")
    return urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))


def require_safety(provider: str, operation_class: str, method: str, relative_path: str) -> None:
    operation_class = operation_class.lower()
    if operation_class not in SAFETY_CLASSES:
        raise ControlPlaneHttpError(f"invalid safety class: {operation_class}")
    method = method.upper()
    parsed_path = urllib.parse.urlsplit(relative_path).path.rstrip("/") or "/"

    if method in {"GET", "HEAD"} and operation_class != "read":
        raise ControlPlaneHttpError("GET/HEAD operations must be classified as read")

    if operation_class == "read":
        if method in {"GET", "HEAD"}:
            pass
        elif method == "POST" and parsed_path == GRAPHQL_PATH[provider]:
            pass
        else:
            raise ControlPlaneHttpError(
                "read operations may use GET/HEAD, or POST only to the provider GraphQL endpoint"
            )

    if operation_class not in allowed_classes():
        raise ControlPlaneHttpError(
            f"safety class {operation_class!r} is not granted by CONTROL_PLANE_ALLOWED_CLASSES"
        )


def read_body(args: argparse.Namespace) -> bytes | None:
    if args.body_json is not None and args.body_file is not None:
        raise ControlPlaneHttpError("use only one of --body-json or --body-file")
    if args.body_json is not None:
        parsed = json.loads(args.body_json)
        return json.dumps(parsed, separators=(",", ":")).encode("utf-8")
    if args.body_file is not None:
        parsed = json.loads(args.body_file.read_text(encoding="utf-8"))
        return json.dumps(parsed, separators=(",", ":")).encode("utf-8")
    return None


def request_api(args: argparse.Namespace) -> Any:
    method = args.method.upper()
    relative = safe_relative_path(args.path)
    require_safety(args.provider, args.operation_class, method, relative)
    base, headers = provider_config(args.provider)
    url = base + relative
    body = read_body(args)
    if body is not None:
        headers = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw:
                return {"http_status": response.status}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"http_status": response.status, "text": raw[:100_000]}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            detail = raw[:20_000]
        raise ControlPlaneHttpError(
            f"{args.provider} HTTP {exc.code}: {json.dumps(redact(detail), ensure_ascii=False)[:12000]}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["github", "cloudflare"], required=True)
    parser.add_argument("--method", default="GET")
    parser.add_argument("--path", required=True)
    parser.add_argument("--operation-class", choices=sorted(SAFETY_CLASSES), default="read")
    parser.add_argument("--body-json")
    parser.add_argument("--body-file", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    if args.body_file is not None:
        content = args.body_file.read()
        args.body_file.close()

        class _Body:
            def read_text(self, encoding: str = "utf-8") -> str:
                del encoding
                return content

        args.body_file = _Body()

    try:
        result = request_api(args)
    except (ControlPlaneHttpError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "result": redact(result)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
