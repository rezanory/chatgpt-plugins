from chatgpt_plugins_core.capability_registry import (
    SafetyClass,
    capability_families,
    preferred_transports,
    providers,
    supports_safety_class,
)


def test_all_three_control_plane_providers_are_registered() -> None:
    assert providers() == ("kaggle", "cloudflare", "github")


def test_kaggle_registry_covers_non_kernel_surfaces() -> None:
    names = {item.name for item in capability_families("kaggle")}
    assert {
        "competitions",
        "datasets",
        "kernels",
        "models",
        "model_variations",
        "model_variation_versions",
        "files_inbox",
        "forums_discussions",
        "benchmarks",
        "auth_config",
        "accelerator_quota",
        "search",
        "raw_services",
    } <= names


def test_cloudflare_registry_covers_resource_families() -> None:
    names = {item.name for item in capability_families("cloudflare")}
    assert {
        "workers",
        "pages",
        "kv",
        "d1",
        "r2",
        "queues",
        "workflows",
        "pipelines",
        "hyperdrive",
        "vectorize",
        "ai_ai_search",
        "containers",
        "vpc_network_tunnel",
        "secrets",
        "dns_zones_rules_waf",
        "zero_trust",
        "observability_analytics",
        "raw_api",
    } <= names


def test_github_registry_has_native_connector_and_api_fallback() -> None:
    assert preferred_transports("github", "pull_requests_reviews_merge_queue")[0] == "native_connector"
    assert "gh_api" in preferred_transports("github", "raw_api")
    assert supports_safety_class("github", "actions_workflows_runs_jobs_artifacts_caches_runners", SafetyClass.COMPUTE)


def test_read_only_search_is_not_misclassified_as_compute() -> None:
    assert supports_safety_class("kaggle", "search", SafetyClass.READ)
    assert not supports_safety_class("kaggle", "search", SafetyClass.COMPUTE)
