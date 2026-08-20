from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OIDC = ROOT / "deploy/cloudflare-worker-free/src/control-plane-v3-oidc.ts"
INDEX = ROOT / "deploy/cloudflare-worker-free/src/control-plane-v3-index.ts"
QUERY = ROOT / "scripts/control_plane_query.py"


def test_oidc_broker_is_bound_to_stable_repository_and_actor_ids() -> None:
    source = OIDC.read_text(encoding="utf-8")
    assert 'TRUSTED_REPOSITORY_ID = "1337215097"' in source
    assert 'TRUSTED_OWNER_ID = "62356000"' in source
    assert 'TRUSTED_ACTOR_ID = "62356000"' in source
    assert 'CONTROL_PLANE_AUDIENCE = "cgp-control-plane-v3"' in source
    assert "crypto.subtle.verify" in source
    assert 'header.alg !== "RS256"' in source
    assert 'eventName !== "issue_comment"' in source


def test_worker_read_broker_is_read_only_and_falls_through_to_canonical_worker() -> None:
    source = INDEX.read_text(encoding="utf-8")
    assert '"/control-plane/v3/read/kaggle"' in source
    assert "verifyGitHubReadBrokerOidc" in source
    assert "kaggleReadCall" in source
    assert "kaggleScopedCall" not in source
    assert "canonicalWorker.fetch(request, env, ctx)" in source
    assert 'read_only: true' in source


def test_query_broker_uses_fixed_hosts_and_rejects_graphql_mutations() -> None:
    source = QUERY.read_text(encoding="utf-8")
    assert 'https://api.github.com' in source
    assert 'https://api.cloudflare.com' in source
    assert 'https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev' in source
    assert 'GraphQL read broker accepts query operations only' in source
    assert 'Cloudflare GraphQL broker accepts query operations only' in source
    assert 'provider must be github, cloudflare, or kaggle' in source
    assert 'subprocess.run(' in source
    assert 'shell=True' not in source
