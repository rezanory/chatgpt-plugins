from __future__ import annotations

import argparse
from pathlib import Path

from ..kaggle_cli import KaggleCliError, run_kaggle
from ..sanitize import sanitize_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--comment-file", required=True)
    args = parser.parse_args(argv)

    try:
        run_kaggle(["datasets", "list", "-m", "-p", "1"], timeout=120)
        body = (
            f"Kaggle auth check: account=`{args.account_id}` owner=`{args.owner}` "
            "status=`AUTH_OK`.\n"
        )
    except KaggleCliError as exc:
        safe = sanitize_text(str(exc), max_chars=1500).strip()
        body = (
            f"Kaggle auth check: account=`{args.account_id}` owner=`{args.owner}` "
            "status=`AUTH_FAILED`.\n\n"
            "```text\n"
            f"{safe}\n"
            "```\n"
        )

    Path(args.comment_file).write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
