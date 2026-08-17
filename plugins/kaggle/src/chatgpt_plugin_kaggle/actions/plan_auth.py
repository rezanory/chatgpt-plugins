from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..config import load_accounts


def _write_output(name: str, value: str) -> None:
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-file", required=True)
    args = parser.parse_args(argv)

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
            }
        )

    matrix = json.dumps({"include": include}, separators=(",", ":"))
    _write_output("matrix", matrix)
    _write_output("count", str(len(include)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
