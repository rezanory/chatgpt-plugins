from __future__ import annotations

from dataclasses import dataclass

from .capability_registry import SafetyClass, preferred_transports, supports_safety_class


@dataclass(frozen=True, slots=True)
class RouteDecision:
    provider: str
    family: str
    safety: SafetyClass
    transport: str
    checked_transports: tuple[str, ...]
    authorization_required: bool


class CapabilityRouteError(RuntimeError):
    pass


def route_capability(
    *,
    provider: str,
    family: str,
    safety: SafetyClass,
    available_transports: set[str] | frozenset[str],
) -> RouteDecision:
    if not supports_safety_class(provider, family, safety):
        raise CapabilityRouteError(
            f"{provider}:{family} does not declare safety class {safety.value}"
        )

    preferred = preferred_transports(provider, family)
    checked: list[str] = []
    for transport in preferred:
        checked.append(transport)
        if transport in available_transports:
            return RouteDecision(
                provider=provider,
                family=family,
                safety=safety,
                transport=transport,
                checked_transports=tuple(checked),
                authorization_required=safety is not SafetyClass.READ,
            )

    raise CapabilityRouteError(
        "no available transport for "
        f"{provider}:{family}; checked={','.join(checked) or '<none>'}"
    )
