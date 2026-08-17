from __future__ import annotations

import hashlib
import hmac
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from chatgpt_plugin_kaggle_gateway.control import (
    CONTROL_MARKER,
    CONTROL_SCHEMA,
    ControlCommand,
    rerun_existing,
    verify_github_signature,
)


def _body(**overrides) -> str:
    payload = {
        "schema": CONTROL_SCHEMA,
        "action": "rerun_existing",
        "job_id": "job-001",
        "account_id": "kg-01",
        "kernel_ref": "owner-one/kernel-a",
    }
    payload.update(overrides)
    return f"{CONTROL_MARKER}\n```json\n{json.dumps(payload)}\n```"


def test_control_command_parses_strict_issue_payload():
    command = ControlCommand.from_issue("[KAGGLE-RUN] rerun shard", _body())
    assert command.job_id == "job-001"
    assert command.account_id == "kg-01"
    assert command.kernel_ref == "owner-one/kernel-a"


def test_control_command_rejects_write_actions_outside_v01():
    with pytest.raises(ValueError, match="only supports rerun_existing"):
        ControlCommand.from_issue("[KAGGLE-RUN] bad", _body(action="delete_kernel"))


def test_github_signature_verification():
    secret = "webhook-secret"
    body = b'{"action":"opened"}'
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_github_signature(secret, body, f"sha256={digest}") is True
    assert verify_github_signature(secret, body, "sha256=deadbeef") is False


class _FakeApi:
    def __init__(self, kernel_ref: str, *, wrong_identity: bool = False) -> None:
        self.kernel_ref = kernel_ref
        self.wrong_identity = wrong_identity
        self.events: list[tuple] = []

    def kernels_pull(self, kernel_ref, path, metadata=True, quiet=True):
        self.events.append(("pull", kernel_ref, metadata, quiet))
        identity = "other-owner/other-kernel" if self.wrong_identity else kernel_ref
        Path(path, "kernel-metadata.json").write_text(
            json.dumps({"id": identity, "code_file": "main.py"}), encoding="utf-8"
        )
        Path(path, "main.py").write_text("print('ok')\n", encoding="utf-8")

    def kernels_push(self, folder):
        self.events.append(("push", Path(folder).name))
        return SimpleNamespace(url="https://example.invalid/kernel", versionNumber=2)


class _FakePool:
    def __init__(self, *, wrong_identity: bool = False) -> None:
        api = _FakeApi("owner-one/kernel-a", wrong_identity=wrong_identity)
        self.api = api
        self.slot = SimpleNamespace(api=api, lock=threading.RLock())

    def _validate_kernel_owner(self, account_id, kernel_ref):
        assert account_id == "kg-01"
        if not kernel_ref.startswith("owner-one/"):
            raise ValueError("wrong owner")

    def _slot(self, account_id):
        assert account_id == "kg-01"
        return self.slot

    def safe_error(self, account_id, exc):
        return str(exc)


def test_rerun_existing_uses_direct_pull_then_push():
    pool = _FakePool()
    command = ControlCommand.from_issue("[KAGGLE-RUN] rerun", _body())
    result = rerun_existing(pool, command)
    assert pool.api.events[0] == ("pull", "owner-one/kernel-a", True, True)
    assert pool.api.events[1][0] == "push"
    assert result["url"] == "https://example.invalid/kernel"


def test_rerun_existing_rejects_pulled_identity_mismatch():
    pool = _FakePool(wrong_identity=True)
    command = ControlCommand.from_issue("[KAGGLE-RUN] rerun", _body())
    with pytest.raises(RuntimeError, match="identity does not match"):
        rerun_existing(pool, command)
