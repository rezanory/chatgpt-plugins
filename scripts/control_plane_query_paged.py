from __future__ import annotations

import sys
from typing import Any

import control_plane_query as base


def _find_output_listing(value: Any) -> tuple[list[Any], str] | None:
    """Find the Kaggle output listing inside broker/API wrapper envelopes.

    The Cloudflare broker can wrap the Kaggle API payload one or more times
    (for example result.result.files). Keep this read helper tolerant to those
    transport envelopes without weakening the exact-path allowlist below.
    """
    if isinstance(value, dict):
        files = value.get("files")
        if isinstance(files, list):
            if not files or any(isinstance(item, dict) and item.get("fileName") for item in files):
                token = str(value.get("nextPageToken") or "")
                return files, token
        for child in value.values():
            found = _find_output_listing(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_output_listing(child)
            if found is not None:
                return found
    return None


def _find_next_page_token(value: Any) -> str:
    if isinstance(value, dict):
        token = value.get("nextPageToken")
        if token:
            return str(token)
        for child in value.values():
            found = _find_next_page_token(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_next_page_token(child)
            if found:
                return found
    return ""


def kaggle_output_json_files(payload: dict[str, Any]) -> Any:
    account_id = str(payload.get("account_id", ""))
    owner = base.ACCOUNTS.get(account_id)
    kernel_ref = str(payload.get("kernel_ref", "")).strip()
    if not owner or "/" not in kernel_ref:
        raise base.QueryError("kernel_output_json_files requires valid account_id and kernel_ref")
    ref_owner, slug = kernel_ref.split("/", 1)
    if ref_owner.lower() != owner.lower() or not base.re.fullmatch(r"[A-Za-z0-9._-]{1,200}", slug):
        raise base.QueryError("kernel_ref owner/slug does not match account_id")

    names = payload.get("file_names")
    if not isinstance(names, list) or not 1 <= len(names) <= 10:
        raise base.QueryError("file_names must contain between 1 and 10 exact JSON paths")
    exact_names: list[str] = []
    for item in names:
        name = str(item).strip()
        if (
            not name.endswith(".json")
            or len(name) > 500
            or name.startswith("/")
            or ".." in name
            or "\\" in name
            or not base.re.fullmatch(r"[A-Za-z0-9._/-]+", name)
        ):
            raise base.QueryError("file_names contains an unsafe or non-JSON path")
        if name in exact_names:
            raise base.QueryError("file_names must be unique")
        exact_names.append(name)

    max_bytes = int(payload.get("max_bytes_per_file", 65536))
    if max_bytes < 1024 or max_bytes > 262144:
        raise base.QueryError("max_bytes_per_file must be between 1024 and 262144")

    by_name: dict[str, dict[str, Any]] = {}
    page_token = ""
    seen_tokens: set[str] = set()
    for _ in range(20):
        body: dict[str, Any] = {
            "userName": owner,
            "kernelSlug": slug,
            "pageSize": 50,
        }
        if page_token:
            body["pageToken"] = page_token
        response = base.worker_kaggle_read(
            {
                "provider": "kaggle",
                "action": "raw_read",
                "account_id": account_id,
                "service": "kernels.KernelsApiService",
                "method": "ListKernelSessionOutput",
                "body": body,
            }
        )
        if not isinstance(response, dict) or not response.get("ok"):
            raise base.QueryError("ListKernelSessionOutput did not return ok=true")
        listing = _find_output_listing(response)
        if listing is None:
            raise base.QueryError("ListKernelSessionOutput files list missing")
        raw_files, listing_token = listing
        for item in raw_files:
            if isinstance(item, dict) and item.get("fileName"):
                by_name[str(item["fileName"])] = item
        if all(name in by_name for name in exact_names):
            break
        next_token = listing_token or _find_next_page_token(response)
        if not next_token:
            break
        if next_token == page_token or next_token in seen_tokens:
            raise base.QueryError("ListKernelSessionOutput page token did not advance")
        seen_tokens.add(next_token)
        page_token = next_token
    else:
        raise base.QueryError("Kaggle output pagination exceeded bounded page limit")

    outputs = []
    for name in exact_names:
        item = by_name.get(name)
        if item is None:
            raise base.QueryError(f"requested Kaggle output JSON not found: {name}")
        url = str(item.get("url", ""))
        if not url:
            raise base.QueryError(f"requested Kaggle output JSON has no download URL: {name}")
        outputs.append({"file_name": name, "json": base._fetch_output_json_url(url, max_bytes)})

    return {
        "transport": "cloudflare_kaggle_api_token_paging_then_bounded_output_fetch",
        "account_id": account_id,
        "kernel_ref": kernel_ref,
        "file_count": len(outputs),
        "files": outputs,
        "signed_urls_returned": False,
    }


base.kaggle_output_json_files = kaggle_output_json_files

if __name__ == "__main__":
    sys.exit(base.main())
