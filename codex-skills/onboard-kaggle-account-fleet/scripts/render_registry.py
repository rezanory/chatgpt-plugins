#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def existing_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = load(path)
    if not isinstance(value, list):
        raise ValueError("existing registry must be a JSON list")
    result: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict) or not row.get("account_id"):
            raise ValueError("invalid existing registry row")
        result[str(row["account_id"])] = dict(row)
    return result


def clean_labels(labels: Any, role: str) -> list[str]:
    values = [str(v) for v in labels] if isinstance(labels, list) else []
    values = [v for v in values if not v.startswith("route:")]
    if "route:cloudflare-oidc" not in values:
        values.append("route:cloudflare-oidc")
    if not any(v.startswith("pool:") for v in values):
        values.append("pool:kaggle-fleet")
    if role == "master" and "role:master" not in values:
        values.append("role:master")
    return values

def render(plan: dict[str, Any], existing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    accounts = plan.get("accounts")
    if not isinstance(accounts, list):
        raise ValueError("plan.accounts must be a list")
    rows: list[dict[str, Any]] = []
    for item in accounts:
        if not isinstance(item, dict):
            raise ValueError("plan account row must be an object")
        account_id = str(item.get("account_id", ""))
        if not account_id:
            raise ValueError("plan account_id missing")
        role = str(item.get("role", "worker"))
        base = existing.get(account_id, {})
        row = {
            "account_id": account_id,
            "owner_slug": str(item.get("owner_slug", "")),
            "secret_scope": str(item.get("environment", "")),
            "enabled": True,
            "max_parallel": int(base.get("max_parallel", 1)),
            "capabilities": base.get("capabilities", ["cpu", "gpu"]),
            "labels": clean_labels(base.get("labels", []), role),
            "default_accelerator": base.get("default_accelerator", "NvidiaTeslaT4"),
        }
        rows.append(row)
    rows.sort(key=lambda row: int(str(row["account_id"]).split("-", 1)[1]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Render canonical Kaggle accounts.json from a fleet plan")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = load(args.plan)
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    rows = render(plan, existing_map(args.existing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"REGISTRY_READY={len(rows)}/{len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"REGISTRY_ERROR: {exc}") from exc
