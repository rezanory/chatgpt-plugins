from __future__ import annotations

import argparse
import os
from pathlib import Path

from chatgpt_plugins_github_bridge import GitHubIssueClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--file", required=True)
    args = parser.parse_args(argv)
    body = Path(args.file).read_text(encoding="utf-8")
    client = GitHubIssueClient(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
    comment_id = client.create_comment(args.issue, body)
    print(f"created issue comment {comment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
