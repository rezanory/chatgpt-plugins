from __future__ import annotations

from chatgpt_plugin_kaggle_gateway import api_pool
from chatgpt_plugin_kaggle_gateway.config import GatewayAccount, GatewayRegistry


class MinimalApi:
    CONFIG_NAME_USER = "username"
    CONFIG_NAME_KEY = "key"

    def __init__(self):
        self.config_dir = ""
        self.config_file = "kaggle.json"
        self.config = ""
        self.config_values = {}
        self._authenticated = False

    def set_config_value(self, name, value, quiet=False):
        self.config_values[name] = value

    def authenticate(self):
        self._authenticated = True

    def kernels_list(self, page_size=20, **kwargs):
        assert self._authenticated
        return []


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


def test_kaggle_api_class_is_loaded_once_before_parallel_auth(monkeypatch):
    monkeypatch.setenv("CGP_KAGGLE_KG01_USERNAME", "owner-one")
    monkeypatch.setenv("CGP_KAGGLE_KG01_TOKEN", "token-one")
    monkeypatch.setenv("CGP_KAGGLE_KG02_USERNAME", "owner-two")
    monkeypatch.setenv("CGP_KAGGLE_KG02_TOKEN", "token-two")
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(name, raising=False)

    load_count = 0

    def load_class():
        nonlocal load_count
        load_count += 1
        return MinimalApi

    monkeypatch.setattr(api_pool, "_kaggle_api_class", load_class)
    with api_pool.KaggleApiPool(_registry()) as pool:
        results = pool.auth_check_all(max_workers=2)

    assert load_count == 1
    assert [result["auth_ok"] for result in results] == [True, True]
