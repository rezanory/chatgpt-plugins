import pytest
from chatgpt_plugins_core.capability_registry import SafetyClass
from chatgpt_plugins_core.capability_router import CapabilityRouteError, route_capability


def test_github_prefers_native_connector() -> None:
    result = route_capability(
        provider="github",
        family="pull_requests_reviews_merge_queue",
        safety=SafetyClass.READ,
        available_transports={"native_connector", "gh", "rest"},
    )
    assert result.transport == "native_connector"
    assert result.authorization_required is False


def test_github_falls_back_to_gh_when_connector_missing() -> None:
    result = route_capability(
        provider="github",
        family="pull_requests_reviews_merge_queue",
        safety=SafetyClass.WRITE,
        available_transports={"gh", "rest"},
    )
    assert result.transport == "gh"
    assert result.authorization_required is True


def test_kaggle_raw_service_fallback_is_available() -> None:
    result = route_capability(
        provider="kaggle",
        family="raw_services",
        safety=SafetyClass.READ,
        available_transports={"rest"},
    )
    assert result.transport == "rest"


def test_missing_transport_is_explicit() -> None:
    with pytest.raises(CapabilityRouteError, match="no available transport"):
        route_capability(
            provider="cloudflare",
            family="workers",
            safety=SafetyClass.READ,
            available_transports=set(),
        )
