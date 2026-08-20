from pathlib import Path


WORKER = (
    Path(__file__).resolve().parents[3]
    / "deploy"
    / "cloudflare-worker-free"
    / "src"
    / "control-plane-v3-kaggle.ts"
)


def test_raw_kaggle_transport_is_fixed_host_and_scoped() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert 'const API_ROOT = "https://api.kaggle.com/v1"' in source
    assert "CGP_PROJECT_CONTROL_TOKEN" in source
    assert "CGP_CONTROL_SCOPES" in source
    assert "CGP_CONTROL_EXPIRES_AT" in source
    assert "controlPlaneCapabilityAuthorized" in source


def test_read_methods_fail_closed() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "const READ_METHOD" in source
    assert 'spec.operationClass === "read" && !READ_METHOD.test(spec.method)' in source
    assert "not read-like" in source


def test_global_scope_requires_separate_opt_in() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "CGP_CONTROL_ALLOW_GLOBAL_SCOPE" in source
    assert 'candidate === "*" && env.CGP_CONTROL_ALLOW_GLOBAL_SCOPE === "1"' in source


def test_non_read_capability_must_be_enabled_and_not_expired() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "CGP_CONTROL_V3_MUTATION_ENABLED" in source
    assert 'env.CGP_CONTROL_V3_MUTATION_ENABLED !== "1"' in source
    assert "function notExpired" in source
    assert "if (!notExpired(env)) return false" in source


def test_live_phase_probe_uses_bounded_sanitized_log() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "export async function kaggleLiveLog" in source
    assert "export async function kagglePhaseProbe" in source
    assert "ListKernelSessionOutput" in source
    assert "CGP_PHASE:" in source
    assert "sanitizeLog" in source
    assert "maxChars > 200_000" in source


def test_phase_probe_does_not_require_labels_or_metrics() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "labels_or_metrics_required: false" in source
