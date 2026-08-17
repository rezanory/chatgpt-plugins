from __future__ import annotations

import argparse
import json
from typing import Any

from .api_pool import KaggleApiPool
from .config import load_registry


def _compact_kernel(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"value": str(item)[:300]}
    keep = (
        "ref",
        "title",
        "lastRunTime",
        "last_run_time",
        "lastRunStatus",
        "last_run_status",
    )
    return {key: item[key] for key in keep if key in item}


def run(search: str, page_size: int, max_workers: int) -> dict[str, Any]:
    registry = load_registry()
    enabled = [account for account in registry.accounts if account.enabled]
    result: dict[str, Any] = {
        "schema": "chatgpt.kaggle.direct-selftest/v1",
        "enabled_accounts": [account.account_id for account in enabled],
        "auth": [],
        "inventory": [],
    }

    with KaggleApiPool(registry) as pool:
        auth = pool.auth_check_all(max_workers=max_workers)
        result["auth"] = auth

        for account in enabled:
            auth_row = next((row for row in auth if row["account_id"] == account.account_id), None)
            if not auth_row or not auth_row.get("auth_ok"):
                result["inventory"].append(
                    {
                        "account_id": account.account_id,
                        "ok": False,
                        "skipped": "authentication_failed",
                    }
                )
                continue
            try:
                kernels = pool.kernels_list(
                    account.account_id,
                    search=search or None,
                    page_size=page_size,
                    mine=True,
                    sort_by="dateRun",
                )
                result["inventory"].append(
                    {
                        "account_id": account.account_id,
                        "ok": True,
                        "kernel_count": len(kernels),
                        "kernels": [_compact_kernel(item) for item in kernels],
                    }
                )
            except BaseException as exc:
                result["inventory"].append(
                    {
                        "account_id": account.account_id,
                        "ok": False,
                        "error_type": exc.__class__.__name__,
                        "error": pool.safe_error(account.account_id, exc),
                    }
                )

    result["all_auth_ok"] = bool(result["auth"]) and all(
        row.get("auth_ok") is True for row in result["auth"]
    )
    result["all_inventory_ok"] = bool(result["inventory"]) and all(
        row.get("ok") is True for row in result["inventory"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct KaggleApi multi-account readiness check. No Kaggle CLI is used."
    )
    parser.add_argument("--search", default="pneumonia-v6-2-2")
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    if not 1 <= args.page_size <= 100:
        raise SystemExit("--page-size must be between 1 and 100")
    if not 1 <= args.max_workers <= 16:
        raise SystemExit("--max-workers must be between 1 and 16")

    result = run(args.search, args.page_size, args.max_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_auth_ok"] else 2)


if __name__ == "__main__":
    main()
