from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from chatgpt_plugins_core import CliTransportError, SafetyClass
from chatgpt_plugins_core.cli_transport import run_provider_cli


def test_read_transport_rejects_mutation_verb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/gh")
    with pytest.raises(CliTransportError, match="mutation-like verb"):
        run_provider_cli(
            "github",
            ["issue", "create"],
            safety=SafetyClass.READ,
            env={"CONTROL_PLANE_ALLOWED_CLASSES": "read"},
        )


def test_unknown_read_command_requires_non_read_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/wrangler")
    with pytest.raises(CliTransportError, match="read-only classification is not proven"):
        run_provider_cli(
            "cloudflare",
            ["experimental-future-command"],
            safety=SafetyClass.READ,
            env={"CONTROL_PLANE_ALLOWED_CLASSES": "read"},
        )


def test_new_non_read_command_remains_usable_with_explicit_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/wrangler")

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    receipt = run_provider_cli(
        "cloudflare",
        ["experimental-future-command", "--flag"],
        safety=SafetyClass.WRITE,
        env={"CONTROL_PLANE_ALLOWED_CLASSES": "read,write"},
    )
    assert receipt.provider == "cloudflare"
    assert receipt.safety is SafetyClass.WRITE
    assert receipt.output == "ok"


def test_non_read_command_fails_without_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/kaggle")
    with pytest.raises(CliTransportError, match="is not granted"):
        run_provider_cli(
            "kaggle",
            ["models", "create"],
            safety=SafetyClass.WRITE,
            env={"CONTROL_PLANE_ALLOWED_CLASSES": "read"},
        )


def test_provider_executable_is_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/safe/{name}")

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["shell"] = kwargs["shell"]
        return SimpleNamespace(returncode=0, stdout="version")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_provider_cli(
        "github",
        ["--version"],
        safety=SafetyClass.READ,
        env={"CONTROL_PLANE_ALLOWED_CLASSES": "read"},
    )
    assert captured["command"][0] == "/safe/gh"
    assert captured["shell"] is False


def test_cli_receipt_redacts_secret_output_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/kaggle")

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="token=KGAT_super_secret_value\nAuthorization: Bearer abc.def.ghi",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    receipt = run_provider_cli(
        "kaggle",
        ["auth", "print-access-token", "--token", "KGAT_argument_secret"],
        safety=SafetyClass.PRIVILEGED,
        env={"CONTROL_PLANE_ALLOWED_CLASSES": "privileged"},
    )
    assert "KGAT_" not in receipt.output
    assert "abc.def.ghi" not in receipt.output
    assert "KGAT_argument_secret" not in " ".join(receipt.argv)
    assert "<redacted>" in receipt.output
    assert "<redacted>" in receipt.argv
