from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import ComputeJobSpec


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    provider: str
    features: frozenset[str]


@dataclass(frozen=True, slots=True)
class TransportReceipt:
    transport: str
    external_id: str
    metadata: dict[str, Any]


class TransportAdapter(Protocol):
    """How an already validated provider operation is carried to the external control plane."""

    def submit(self, job: ComputeJobSpec) -> TransportReceipt: ...


class ComputeProvider(Protocol):
    """Provider semantics independent from GitHub Issue, MCP, or any future transport."""

    def capabilities(self) -> ProviderCapabilities: ...
