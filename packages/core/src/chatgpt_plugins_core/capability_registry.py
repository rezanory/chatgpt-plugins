from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SafetyClass(StrEnum):
    READ = "read"
    WRITE = "write"
    COMPUTE = "compute"
    DESTRUCTIVE = "destructive"
    PRIVILEGED = "privileged"


@dataclass(frozen=True, slots=True)
class CapabilityFamily:
    provider: str
    name: str
    transports: tuple[str, ...]
    safety: tuple[SafetyClass, ...]
    runtime_discovery: bool = True


R = (SafetyClass.READ,)
RWD = (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE)
RWCD = (
    SafetyClass.READ,
    SafetyClass.WRITE,
    SafetyClass.COMPUTE,
    SafetyClass.DESTRUCTIVE,
)
RWP = (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.PRIVILEGED)
RWDP = (
    SafetyClass.READ,
    SafetyClass.WRITE,
    SafetyClass.DESTRUCTIVE,
    SafetyClass.PRIVILEGED,
)
RWCP = (
    SafetyClass.READ,
    SafetyClass.WRITE,
    SafetyClass.COMPUTE,
    SafetyClass.PRIVILEGED,
)
RWCDP = (
    SafetyClass.READ,
    SafetyClass.WRITE,
    SafetyClass.COMPUTE,
    SafetyClass.DESTRUCTIVE,
    SafetyClass.PRIVILEGED,
)


def _family(
    provider: str,
    name: str,
    transports: tuple[str, ...],
    safety: tuple[SafetyClass, ...],
) -> CapabilityFamily:
    return CapabilityFamily(provider, name, transports, safety)


def _k(name: str, transports: tuple[str, ...], safety: tuple[SafetyClass, ...]) -> CapabilityFamily:
    return _family("kaggle", name, transports, safety)


def _c(name: str, transports: tuple[str, ...], safety: tuple[SafetyClass, ...]) -> CapabilityFamily:
    return _family("cloudflare", name, transports, safety)


def _g(name: str, transports: tuple[str, ...], safety: tuple[SafetyClass, ...]) -> CapabilityFamily:
    return _family("github", name, transports, safety)


REGISTRY: dict[str, tuple[CapabilityFamily, ...]] = {
    "kaggle": (
        _k("auth_config", ("kaggle_cli", "rest"), RWD),
        _k("search", ("kaggle_cli", "rest"), R),
        _k("accelerator_quota", ("kaggle_cli", "rest"), R),
        _k("competitions", ("kaggle_cli", "rest"), RWCDP),
        _k("datasets", ("kaggle_cli", "kagglehub", "rest"), RWCD),
        _k("kernels", ("kaggle_cli", "rest"), RWCD),
        _k("models", ("kaggle_cli", "kagglehub", "rest"), RWCD),
        _k("model_variations", ("kaggle_cli", "kagglehub", "rest"), RWCD),
        _k("model_variation_versions", ("kaggle_cli", "kagglehub", "rest"), RWCD),
        _k("files_inbox", ("kaggle_cli", "rest"), RWCD),
        _k("forums_discussions", ("kaggle_cli", "rest"), RWP),
        _k("benchmarks", ("kaggle_cli", "rest"), RWCDP),
        _k("raw_services", ("rest",), RWCDP),
    ),
    "cloudflare": (
        _c("workers", ("cloudflare_plugin", "wrangler", "rest"), RWCD),
        _c("pages", ("cloudflare_plugin", "wrangler", "rest"), RWCD),
        _c("kv", ("wrangler", "rest"), RWD),
        _c("d1", ("wrangler", "rest"), RWCD),
        _c("r2", ("wrangler", "rest", "s3"), RWCD),
        _c("queues", ("wrangler", "rest"), RWCD),
        _c("workflows", ("wrangler", "rest"), RWCD),
        _c("pipelines", ("wrangler", "rest"), RWCD),
        _c("hyperdrive", ("wrangler", "rest"), RWD),
        _c("vectorize", ("wrangler", "rest"), RWCD),
        _c("ai_ai_search", ("cloudflare_plugin", "wrangler", "rest"), RWCD),
        _c("browser_rendering", ("cloudflare_plugin", "wrangler", "rest"), RWCD),
        _c("containers", ("wrangler", "rest"), RWCD),
        _c("durable_objects", ("wrangler", "rest", "worker_rpc"), RWCD),
        _c("workers_for_platforms", ("wrangler", "rest"), RWCDP),
        _c("vpc_network_tunnel", ("cloudflare_plugin", "wrangler", "rest"), RWCDP),
        _c("secrets", ("wrangler", "rest"), RWDP),
        _c("dns_zones_rules_waf", ("cloudflare_plugin", "rest"), RWCDP),
        _c("zero_trust", ("cloudflare_plugin", "rest"), RWCDP),
        _c(
            "observability_analytics",
            ("cloudflare_plugin", "wrangler", "graphql", "rest"),
            RWCP,
        ),
        _c("certificates_domains", ("wrangler", "rest"), RWCDP),
        _c("account_billing", ("rest",), RWP),
        _c("raw_api", ("rest", "graphql"), RWCDP),
    ),
    "github": (
        _g(
            "repositories_contents_git",
            ("native_connector", "gh", "rest", "graphql"),
            RWCDP,
        ),
        _g(
            "branches_tags_commits_status_checks",
            ("native_connector", "gh", "rest"),
            RWCD,
        ),
        _g(
            "pull_requests_reviews_merge_queue",
            ("native_connector", "gh", "rest", "graphql"),
            RWCD,
        ),
        _g("issues_comments_reactions", ("native_connector", "gh", "rest"), RWD),
        _g(
            "actions_workflows_runs_jobs_artifacts_caches_runners",
            ("native_connector", "gh", "rest"),
            RWCDP,
        ),
        _g("environments_deployments", ("gh", "rest"), RWCDP),
        _g("releases_packages_pages", ("gh", "rest"), RWCD),
        _g("projects", ("gh", "graphql", "rest"), RWD),
        _g(
            "organizations_teams_members",
            ("native_connector", "gh", "rest", "graphql"),
            RWDP,
        ),
        _g("codespaces", ("gh", "rest"), RWCD),
        _g("discussions_gists", ("gh", "rest", "graphql"), RWD),
        _g("webhooks_apps_installations", ("rest",), RWCDP),
        _g(
            "security_dependabot_scanning_advisories",
            ("native_connector", "gh", "rest"),
            RWCDP,
        ),
        _g("attestations_artifact_metadata", ("gh", "rest"), RWCP),
        _g("rulesets_branch_protection", ("gh", "rest"), RWDP),
        _g("search", ("native_connector", "gh", "rest"), R),
        _g("billing_usage_copilot", ("gh", "rest"), RWP),
        _g("raw_api", ("gh_api", "rest", "graphql"), RWCDP),
    ),
}


def providers() -> tuple[str, ...]:
    return tuple(REGISTRY)


def capability_families(provider: str) -> tuple[CapabilityFamily, ...]:
    try:
        return REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(f"unknown provider: {provider}") from exc


def preferred_transports(provider: str, family: str) -> tuple[str, ...]:
    for capability in capability_families(provider):
        if capability.name == family:
            return capability.transports
    raise ValueError(f"unknown capability family for {provider}: {family}")


def supports_safety_class(provider: str, family: str, safety: SafetyClass) -> bool:
    for capability in capability_families(provider):
        if capability.name == family:
            return safety in capability.safety
    return False
