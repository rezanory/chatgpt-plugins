from __future__ import annotations

import argparse
import json
import re

from ..config import load_accounts
from .common import load_github_event, write_github_output

_SAFE_SEARCH = re.compile(r"^[A-Za-z0-9._ -]{0,80}$")
_PREFIX = "[KAGGLE-INVENTORY]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-file", required=True)
    args = parser.parse_args(argv)

    event = load_github_event()
    title = str((event.get("issue") or {}).get("title") or "")
    if not title.startswith(_PREFIX):
        raise ValueError("inventory issue title must start with [KAGGLE-INVENTORY]")
    search = title[len(_PREFIX) :].strip()
    if not _SAFE_SEARCH.fullmatch(search):
        raise ValueError("inventory search contains unsupported characters or is too long")

    include = []
    for account in load_accounts(args.accounts_file):
        descriptor = account.descriptor
        if not descriptor.enabled:
            continue
        include.append(
            {
                "account_id": descriptor.account_id,
                "account_environment": descriptor.secret_scope,
                "kaggle_owner": account.owner_slug,
                "search": search,
            }
        )

    write_github_output("matrix", json.dumps({"include": include}, separators=(",", ":")))
    write_github_output("count", str(len(include)))
    write_github_output("search", search)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
