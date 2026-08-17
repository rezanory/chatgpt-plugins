from __future__ import annotations

import fnmatch


class RepositoryNotAllowed(PermissionError):
    pass


class RepoPolicy:
    def __init__(self, patterns: list[str] | tuple[str, ...], *, allow_any: bool = False) -> None:
        self.patterns = tuple(p.strip() for p in patterns if p.strip())
        self.allow_any = allow_any

    def allows(self, repository: str) -> bool:
        if self.allow_any:
            return True
        return any(fnmatch.fnmatchcase(repository, pattern) for pattern in self.patterns)

    def require(self, repository: str) -> None:
        if not self.allows(repository):
            raise RepositoryNotAllowed(f"repository {repository!r} is not allowlisted")
