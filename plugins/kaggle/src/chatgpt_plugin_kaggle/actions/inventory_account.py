from __future__ import annotations

import argparse
from pathlib import Path

from ..kaggle_cli import KaggleCliError, run_kaggle
from ..sanitize import sanitize_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--search", default="")
    parser.add_argument("--comment-file", required=True)
    args = parser.parse_args(argv)

    cli_args = [
        "kernels",
        "list",
        "-m",
        "--page-size",
        "50",
        "-v",
        "--sort-by",
        "dateRun",
    ]
    if args.search:
        cli_args += ["-s", args.search]

    try:
        output = run_kaggle(cli_args, timeout=120)
        safe = sanitize_text(output, max_chars=12_000).strip()
        search_suffix = f" search=`{args.search}`" if args.search else ""
        body = (
            f"Kaggle inventory: account=`{args.account_id}` owner=`{args.owner}` "
            f"status=`OK`{search_suffix}.\n\n"
            "```csv\n"
            f"{safe or '(no matching kernels)'}\n"
            "```\n"
        )
    except KaggleCliError as exc:
        safe = sanitize_text(str(exc), max_chars=2_000).strip()
        body = (
            f"Kaggle inventory: account=`{args.account_id}` owner=`{args.owner}` "
            "status=`AUTH_OR_API_FAILED`.\n\n"
            "```text\n"
            f"{safe}\n"
            "```\n"
        )

    Path(args.comment_file).write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
