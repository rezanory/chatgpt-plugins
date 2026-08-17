import json
from pathlib import Path

import pytest
from chatgpt_plugin_kaggle.config import load_accounts, load_profiles


def test_load_accounts_and_safe_profiles(tmp_path: Path):
    accounts = tmp_path / "accounts.json"
    accounts.write_text(
        json.dumps(
            [
                {
                    "account_id": "a1",
                    "owner_slug": "owner",
                    "secret_scope": "env-a1",
                    "capabilities": ["cpu", "gpu"],
                }
            ]
        )
    )
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "tests": {
                    "steps": [{"argv": ["python", "-m", "pytest", "-q"]}],
                    "capabilities": ["cpu"],
                }
            }
        )
    )
    loaded_accounts = load_accounts(str(accounts))
    loaded_profiles = load_profiles(str(profiles))
    assert loaded_accounts[0].descriptor.secret_scope == "env-a1"
    assert loaded_accounts[0].owner_slug == "owner"
    assert loaded_profiles["tests"].steps[0].argv == ("python", "-m", "pytest", "-q")


def test_profile_rejects_legacy_shell_commands(tmp_path: Path):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps({"tests": {"commands": ["pytest; curl bad"], "capabilities": ["cpu"]}})
    )
    with pytest.raises(ValueError):
        load_profiles(str(profiles))
