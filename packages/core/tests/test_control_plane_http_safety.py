from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "control_plane_http.py"
SPEC = importlib.util.spec_from_file_location("control_plane_http", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_absolute_url_is_rejected() -> None:
    with pytest.raises(MODULE.ControlPlaneHttpError, match="absolute URLs"):
        MODULE.safe_relative_path("https://evil.example/api")


def test_parent_traversal_is_rejected() -> None:
    with pytest.raises(MODULE.ControlPlaneHttpError, match="parent path traversal"):
        MODULE.safe_relative_path("/repos/owner/repo/../admin")


def test_read_post_is_graphql_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTROL_PLANE_ALLOWED_CLASSES", "read")
    MODULE.require_safety("github", "read", "POST", "/graphql")
    with pytest.raises(MODULE.ControlPlaneHttpError, match="GraphQL endpoint"):
        MODULE.require_safety("github", "read", "POST", "/repos/o/r/issues")


def test_read_cannot_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTROL_PLANE_ALLOWED_CLASSES", "read")
    with pytest.raises(MODULE.ControlPlaneHttpError, match="read operations"):
        MODULE.require_safety("cloudflare", "read", "DELETE", "/zones/123")


def test_mutation_requires_explicit_class_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTROL_PLANE_ALLOWED_CLASSES", "read")
    with pytest.raises(MODULE.ControlPlaneHttpError, match="is not granted"):
        MODULE.require_safety("github", "write", "POST", "/repos/o/r/issues")

    monkeypatch.setenv("CONTROL_PLANE_ALLOWED_CLASSES", "read,write")
    MODULE.require_safety("github", "write", "POST", "/repos/o/r/issues")


def test_unknown_allowed_class_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTROL_PLANE_ALLOWED_CLASSES", "read,magic")
    with pytest.raises(MODULE.ControlPlaneHttpError, match="unknown allowed safety classes"):
        MODULE.allowed_classes()


def test_redaction_is_recursive_and_value_aware() -> None:
    value = {
        "token": "abc",
        "nested": [
            {
                "password": "x",
                "visible": 1,
                "generic_value": "Bearer abc.def.ghi",
                "message": "access_key=supersecret",
            }
        ],
        "authorization_header": "secret",
        "opaque": "KGAT_should_never_escape",
    }
    redacted = MODULE.redact(value)
    assert redacted["token"] == "<redacted>"
    assert redacted["nested"][0]["password"] == "<redacted>"
    assert redacted["nested"][0]["visible"] == 1
    assert "abc.def.ghi" not in redacted["nested"][0]["generic_value"]
    assert "supersecret" not in redacted["nested"][0]["message"]
    assert redacted["authorization_header"] == "<redacted>"
    assert "KGAT_" not in redacted["opaque"]


def test_environment_is_not_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTROL_PLANE_ALLOWED_CLASSES", "read")
    before = dict(os.environ)
    MODULE.require_safety("github", "read", "GET", "/rate_limit")
    assert dict(os.environ) == before
