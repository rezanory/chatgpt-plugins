from pathlib import Path

WORKFLOW = Path(".github/workflows/pneumonia-v17-m07-continuation-20260914.yml")


def test_runtime_read_broker_has_bounded_transient_retry() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "m07-runtime-admission/1.2" in text
    assert "m07-runtime-terminal/2.2" in text
    assert "M07_RUNTIME_READ_RETRY scope=admission" in text
    assert "M07_RUNTIME_READ_RETRY scope=terminal" in text
    assert text.count("for attempt in range(1,5):") >= 4
    assert "exc.code not in {408,429} and not 500 <= exc.code <= 599" in text
    assert "Runtime admission read broker transient retry exhausted" in text
    assert "Runtime terminal read broker transient retry exhausted" in text


def test_oidc_expiry_refresh_path_is_preserved() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("exc.code==403 and 'GitHub OIDC JWT expired' in detail") >= 2
    assert text.count("value=broker_request(refresh_oidc())") >= 2
