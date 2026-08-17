import pytest
from chatgpt_plugins_core import RepoPolicy, RepositoryNotAllowed


def test_repo_policy_supports_patterns():
    policy = RepoPolicy(["rezanory/*", "org/exact"])
    assert policy.allows("rezanory/project")
    assert policy.allows("org/exact")
    assert not policy.allows("other/repo")
    with pytest.raises(RepositoryNotAllowed):
        policy.require("other/repo")
