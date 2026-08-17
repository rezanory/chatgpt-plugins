from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from chatgpt_plugins_core import FailureReport, SourceRef


@dataclass(frozen=True, slots=True)
class RepairProposal:
    summary: str
    unified_diff: str
    rationale: str
    backend: str


class RepairBackend(Protocol):
    """Model/vendor-neutral source repair contract for V0.2 implementations."""

    def propose(self, source: SourceRef, failure: FailureReport) -> RepairProposal: ...
