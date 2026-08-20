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


REGISTRY: dict[str, tuple[CapabilityFamily, ...]] = {
    "kaggle": (
        CapabilityFamily("kaggle", "auth_config", ("kaggle_cli", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("kaggle", "search", ("kaggle_cli", "rest"), (SafetyClass.READ,)),
        CapabilityFamily("kaggle", "accelerator_quota", ("kaggle_cli", "rest"), (SafetyClass.READ,)),
        CapabilityFamily("kaggle", "competitions", ("kaggle_cli", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("kaggle", "datasets", ("kaggle_cli", "kagglehub", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("kaggle", "kernels", ("kaggle_cli", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("kaggle", "models", ("kaggle_cli", "kagglehub", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("kaggle", "model_variations", ("kaggle_cli", "kagglehub", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("kaggle", "model_variation_versions", ("kaggle_cli", "kagglehub", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("kaggle", "files_inbox", ("kaggle_cli", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("kaggle", "forums_discussions", ("kaggle_cli", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("kaggle", "benchmarks", ("kaggle_cli", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("kaggle", "raw_services", ("rest",), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
    ),
    "cloudflare": (
        CapabilityFamily("cloudflare", "workers", ("cloudflare_plugin", "wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "pages", ("cloudflare_plugin", "wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "kv", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "d1", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "r2", ("wrangler", "rest", "s3"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "queues", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "workflows", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "pipelines", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "hyperdrive", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "vectorize", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "ai_ai_search", ("cloudflare_plugin", "wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "browser_rendering", ("cloudflare_plugin", "wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "containers", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "durable_objects", ("wrangler", "rest", "worker_rpc"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("cloudflare", "workers_for_platforms", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("cloudflare", "vpc_network_tunnel", ("cloudflare_plugin", "wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("cloudflare", "secrets", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("cloudflare", "dns_zones_rules_waf", ("cloudflare_plugin", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("cloudflare", "zero_trust", ("cloudflare_plugin", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("cloudflare", "observability_analytics", ("cloudflare_plugin", "wrangler", "graphql", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("cloudflare", "certificates_domains", ("wrangler", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("cloudflare", "account_billing", ("rest",), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("cloudflare", "raw_api", ("rest", "graphql"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
    ),
    "github": (
        CapabilityFamily("github", "repositories_contents_git", ("native_connector", "gh", "rest", "graphql"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("github", "branches_tags_commits_status_checks", ("native_connector", "gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("github", "pull_requests_reviews_merge_queue", ("native_connector", "gh", "rest", "graphql"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("github", "issues_comments_reactions", ("native_connector", "gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("github", "actions_workflows_runs_jobs_artifacts_caches_runners", ("native_connector", "gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("github", "environments_deployments", ("gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("github", "releases_packages_pages", ("gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("github", "projects", ("gh", "graphql", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("github", "organizations_teams_members", ("native_connector", "gh", "rest", "graphql"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("github", "codespaces", ("gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("github", "discussions_gists", ("gh", "rest", "graphql"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE)),
        CapabilityFamily("github", "webhooks_apps_installations", ("rest",), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("github", "security_dependabot_scanning_advisories", ("native_connector", "gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("github", "attestations_artifact_metadata", ("gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE)),
        CapabilityFamily("github", "rulesets_branch_protection", ("gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("github", "search", ("native_connector", "gh", "rest"), (SafetyClass.READ,)),
        CapabilityFamily("github", "billing_usage_copilot", ("gh", "rest"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.PRIVILEGED)),
        CapabilityFamily("github", "raw_api", ("gh_api", "rest", "graphql"), (SafetyClass.READ, SafetyClass.WRITE, SafetyClass.COMPUTE, SafetyClass.DESTRUCTIVE, SafetyClass.PRIVILEGED)),
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
