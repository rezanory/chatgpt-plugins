from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from chatgpt_plugin_kaggle_gateway import api_pool
from chatgpt_plugin_kaggle_gateway.config import GatewayAccount, GatewayRegistry


class FakeKaggleApi:
    CONFIG_NAME_USER = "username"
    CONFIG_NAME_KEY = "key"

    created: list[FakeKaggleApi] = []
    created_lock = threading.Lock()
    barrier: threading.Barrier | None = None

    def __init__(self):
        self.config_dir = ""
        self.config_file = "kaggle.json"
        self.config = ""
        self.config_values = {}
        self._authenticated = False
        self.events: list[tuple] = []
        with self.created_lock:
            self.created.append(self)

    def set_config_value(self, name, value, quiet=False):
        self.events.append(("set", name, value, quiet))
        self.config_values[name] = value
        path = Path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.config_values), encoding="utf-8")

    def authenticate(self):
        self.events.append(("authenticate",))
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        self._authenticated = True

    def kernels_list(self, page_size=20, **kwargs):
        self.events.append(("kernels_list", page_size, kwargs))
        if not self._authenticated:
            raise RuntimeError("not authenticated")
        return [{"ref": f"{self.config_values['username']}/kernel"}]

    def kernels_status(self, kernel_ref):
        self.events.append(("kernels_status", kernel_ref))
        return {"ref": kernel_ref, "status": "complete"}

    def kernels_logs(self, kernel_ref):
        self.events.append(("kernels_logs", kernel_ref))
        token = self.config_values["key"]
        return f"normal log\ncredential={token}\n"


def _registry() -> GatewayRegistry:
    return GatewayRegistry(
        (
            GatewayAccount(
                account_id="kg-01",
                owner_slug="owner-one",
                username_env="CGP_KAGGLE_KG01_USERNAME",
                token_env="CGP_KAGGLE_KG01_TOKEN",
            ),
            GatewayAccount(
                account_id="kg-02",
                owner_slug="owner-two",
                username_env="CGP_KAGGLE_KG02_USERNAME",
                token_env="CGP_KAGGLE_KG02_TOKEN",
            ),
        )
    )


def _credentials(monkeypatch):
    monkeypatch.setenv("CGP_KAGGLE_KG01_USERNAME", "owner-one")
    monkeypatch.setenv("CGP_KAGGLE_KG01_TOKEN", "token-one")
    monkeypatch.setenv("CGP_KAGGLE_KG02_USERNAME", "owner-two")
    monkeypatch.setenv("CGP_KAGGLE_KG02_TOKEN", "token-two")
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)


def _use_fake(monkeypatch, cls=FakeKaggleApi):
    FakeKaggleApi.created.clear()
    FakeKaggleApi.barrier = None
    monkeypatch.setattr(api_pool, "_kaggle_api_class", lambda: cls)


def test_direct_auth_uses_operator_proven_sequence_and_isolated_config(monkeypatch):
    _credentials(monkeypatch)
    _use_fake(monkeypatch)

    pool = api_pool.KaggleApiPool(_registry())
    try:
        result = pool.auth_check("kg-01")
        assert result["auth_ok"] is True
        fake = FakeKaggleApi.created[0]
        assert fake.events[:4] == [
            ("set", "username", "owner-one", True),
            ("set", "key", "token-one", True),
            ("authenticate",),
            ("kernels_list", 1, {}),
        ]
        assert Path(fake.config).parent.name.startswith("cgp-kaggle-kg-01-")
        payload = json.loads(Path(fake.config).read_text(encoding="utf-8"))
        assert payload == {"username": "owner-one", "key": "token-one"}
    finally:
        config_dirs = [created.config_dir for created in FakeKaggleApi.created]
        pool.close()
        assert all(not Path(path).exists() for path in config_dirs)


def test_two_accounts_authenticate_in_parallel_without_config_collision(monkeypatch):
    _credentials(monkeypatch)
    _use_fake(monkeypatch)
    FakeKaggleApi.barrier = threading.Barrier(2)

    pool = api_pool.KaggleApiPool(_registry())
    started = time.monotonic()
    try:
        results = pool.auth_check_all(max_workers=2)
        elapsed = time.monotonic() - started
        assert elapsed < 2
        assert [result["auth_ok"] for result in results] == [True, True]
        assert len(FakeKaggleApi.created) == 2
        configs = {Path(fake.config) for fake in FakeKaggleApi.created}
        assert len(configs) == 2
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in configs]
        assert {item["username"] for item in payloads} == {"owner-one", "owner-two"}
        assert {item["key"] for item in payloads} == {"token-one", "token-two"}
    finally:
        FakeKaggleApi.barrier = None
        pool.close()


def test_global_kaggle_auth_env_is_rejected(monkeypatch):
    _credentials(monkeypatch)
    monkeypatch.setenv("KAGGLE_KEY", "must-not-override-account-config")
    with pytest.raises(RuntimeError, match="global Kaggle authentication variables are forbidden"):
        api_pool.KaggleApiPool(_registry())


def test_kernel_owner_must_match_selected_account(monkeypatch):
    _credentials(monkeypatch)
    _use_fake(monkeypatch)

    with (
        api_pool.KaggleApiPool(_registry()) as pool,
        pytest.raises(ValueError, match="does not match account"),
    ):
        pool.kernels_status("kg-01", "owner-two/foreign-kernel")


def test_auth_error_redacts_token(monkeypatch):
    _credentials(monkeypatch)

    class FailingApi(FakeKaggleApi):
        def authenticate(self):
            token = self.config_values["key"]
            raise RuntimeError(f"authentication failed for secret={token}")

    _use_fake(monkeypatch, FailingApi)
    with api_pool.KaggleApiPool(_registry()) as pool:
        result = pool.auth_check_all(max_workers=2)[0]
    assert result["auth_ok"] is False
    assert "token-one" not in result["error"]
    assert "[REDACTED]" in result["error"]


def test_system_exit_from_one_account_does_not_kill_other_accounts(monkeypatch):
    _credentials(monkeypatch)

    class ExitOneApi(FakeKaggleApi):
        def authenticate(self):
            if self.config_values["username"] == "owner-one":
                raise SystemExit(1)
            self._authenticated = True

    _use_fake(monkeypatch, ExitOneApi)
    with api_pool.KaggleApiPool(_registry()) as pool:
        results = pool.auth_check_all(max_workers=2)

    assert results[0]["account_id"] == "kg-01"
    assert results[0]["auth_ok"] is False
    assert results[0]["error_type"] == "RuntimeError"
    assert results[1]["account_id"] == "kg-02"
    assert results[1]["auth_ok"] is True


def test_kernel_logs_redact_all_gateway_credentials(monkeypatch):
    _credentials(monkeypatch)
    _use_fake(monkeypatch)

    with api_pool.KaggleApiPool(_registry()) as pool:
        log = pool.kernels_logs("kg-01", "owner-one/kernel")

    assert "token-one" not in log
    assert "owner-one" not in log
    assert "[REDACTED]" in log
