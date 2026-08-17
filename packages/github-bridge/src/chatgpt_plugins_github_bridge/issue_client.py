from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class GitHubIssueApiError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IssueComment:
    id: int
    body: str
    user_login: str


class GitHubIssueClient:
    """Minimal GitHub Issues REST client for Actions-side control-plane operations."""

    def __init__(
        self, repository: str, token: str, *, api_base: str = "https://api.github.com"
    ) -> None:
        if "/" not in repository:
            raise ValueError("repository must be owner/name")
        if not token:
            raise ValueError("GitHub token is required")
        self.repository = repository
        self.token = token
        self.api_base = api_base.rstrip("/")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.api_base}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise GitHubIssueApiError(f"GitHub API HTTP {exc.code}: {detail}") from exc
        if not raw:
            return None
        return json.loads(raw)

    def list_comments(self, issue_number: int, *, max_pages: int = 10) -> list[IssueComment]:
        comments: list[IssueComment] = []
        for page in range(1, max_pages + 1):
            path = (
                f"/repos/{self.repository}/issues/{issue_number}/comments?per_page=100&page={page}"
            )
            data = self._request("GET", path)
            if not isinstance(data, list):
                raise GitHubIssueApiError("unexpected comments response")
            for item in data:
                comments.append(
                    IssueComment(
                        id=int(item["id"]),
                        body=str(item.get("body") or ""),
                        user_login=str((item.get("user") or {}).get("login") or ""),
                    )
                )
            if len(data) < 100:
                break
        return comments

    def create_comment(self, issue_number: int, body: str) -> int:
        if not body or len(body) > 60_000:
            raise ValueError("comment body must be between 1 and 60000 characters")
        data = self._request(
            "POST",
            f"/repos/{self.repository}/issues/{issue_number}/comments",
            {"body": body},
        )
        return int(data["id"])
