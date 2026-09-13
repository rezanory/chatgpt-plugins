#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACCOUNT_RE = re.compile(r"^kg-(\d{2,3})$")
FORBIDDEN_KEYS = re.compile(r"(token|password|private[_-]?key|secret[_-]?value|credential)", re.I)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def account_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and isinstance(value.get("accounts"), list):
        rows = value["accounts"]
    else:
        raise ValueError("input must be an account list or an object with accounts[]")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every account entry must be an object")
    return rows

def reject_secret_material(row: dict[str, Any]) -> None:
    bad = [key for key in row if FORBIDDEN_KEYS.search(str(key)) and key != "secret_scope"]
    if bad:
        raise ValueError(f"credential-like fields are forbidden in planner input: {bad}")


def normalize(row: dict[str, Any], capacity: int) -> dict[str, Any]:
    reject_secret_material(row)
    account_id = str(row.get("account_id", "")).strip()
    match = ACCOUNT_RE.fullmatch(account_id)
    if not match:
        raise ValueError(f"invalid account_id {account_id!r}; expected kg-NN or kg-NNN")
    numeric_id = int(match.group(1))
    if numeric_id < 1:
        raise ValueError("account numeric ID must be positive")
    owner = str(row.get("owner_slug", "")).strip()
    role = str(row.get("role", "worker")).strip() or "worker"
    if role not in {"worker", "master"}:
        raise ValueError(f"unsupported role {role!r} for {account_id}")
    suffix = f"{numeric_id:02d}"
    environment = str(row.get("environment") or row.get("secret_scope") or f"kaggle-{suffix}").strip()
    shard_index = (numeric_id - 1) // capacity
    shard_id = f"s{shard_index:02d}"
    cloudflare_secret = "CGP_KAGGLE_MASTER_TOKEN" if role == "master" else f"CGP_KAGGLE_KG{suffix}_TOKEN"
    return {
        "account_id": account_id,
        "numeric_id": numeric_id,
        "owner_slug": owner,
        "role": role,
        "environment": environment,
        "github_secret_name": "KAGGLE_API_TOKEN",
        "cloudflare_secret_name": cloudflare_secret,
        "shard_id": shard_id,
        "shard_worker": f"chatgpt-kaggle-gateway-{shard_id}",
        "route": "cloudflare-oidc",
    }

def merge_accounts(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in [*existing, *incoming]:
        account_id = str(row.get("account_id", "")).strip()
        if not account_id:
            raise ValueError("account_id is required")
        if account_id in merged:
            prior = merged[account_id]
            old_owner = str(prior.get("owner_slug", "")).strip()
            new_owner = str(row.get("owner_slug", "")).strip()
            if old_owner and new_owner and old_owner.lower() != new_owner.lower():
                raise ValueError(f"owner conflict for {account_id}: {old_owner!r} vs {new_owner!r}")
            merged[account_id] = {**prior, **{k: v for k, v in row.items() if v not in (None, "")}}
        else:
            merged[account_id] = dict(row)
    return list(merged.values())


def build_plan(rows: list[dict[str, Any]], capacity: int, safe_single: int) -> dict[str, Any]:
    normalized = [normalize(row, capacity) for row in rows]
    normalized.sort(key=lambda row: row["numeric_id"])
    ids = [row["account_id"] for row in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate account IDs after normalization")
    secret_names = [row["cloudflare_secret_name"] for row in normalized]
    if len(secret_names) != len(set(secret_names)):
        raise ValueError("duplicate Cloudflare secret names")
    shard_map: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        shard_map.setdefault(row["shard_id"], []).append(row)
    shards = []
    for shard_id, members in sorted(shard_map.items()):
        if len(members) > capacity:
            raise ValueError(f"{shard_id} exceeds capacity {capacity}")
        shards.append({
            "shard_id": shard_id,
            "worker_name": members[0]["shard_worker"],
            "account_count": len(members),
            "account_ids": [m["account_id"] for m in members],
            "bulk_secret_names": [m["cloudflare_secret_name"] for m in members],
        })
    mode = "single-worker" if len(normalized) <= safe_single else "sharded"
    stable = {
        "schema_version": 1,
        "mode": mode,
        "shard_capacity": capacity,
        "safe_single_worker_accounts": safe_single,
        "account_count": len(normalized),
        "accounts": normalized,
        "shards": shards,
    }
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        **stable,
        "plan_sha256": digest,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a secret-free deterministic Kaggle fleet plan")
    parser.add_argument("--new", required=True, type=Path, help="JSON metadata for new accounts")
    parser.add_argument("--existing", type=Path, help="Existing accounts.json or prior metadata")
    parser.add_argument("--output", type=Path, help="Write plan JSON here; stdout if omitted")
    parser.add_argument("--shard-capacity", type=int, default=48)
    parser.add_argument("--safe-single", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.shard_capacity <= 100:
        parser.error("--shard-capacity must be between 1 and 100")
    if not 1 <= args.safe_single <= args.shard_capacity:
        parser.error("--safe-single must be between 1 and shard capacity")
    incoming = account_list(load_json(args.new))
    existing = account_list(load_json(args.existing)) if args.existing else []
    plan = build_plan(merge_accounts(existing, incoming), args.shard_capacity, args.safe_single)
    encoded = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"PLAN_ERROR: {exc}") from exc
