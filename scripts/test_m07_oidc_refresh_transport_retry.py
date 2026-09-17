from pathlib import Path


def test_m07_oidc_refresh_has_bounded_transport_retry():
    text = Path(".github/workflows/pneumonia-v17-m07-continuation-20260914.yml").read_text(encoding="utf-8")
    assert text.count("for attempt in range(1,5):") >= 2
    assert text.count("M07_RUNTIME_OIDC_REFRESH_RETRY") == 2
    assert text.count("except (urllib.error.URLError, TimeoutError, OSError) as exc:") == 2
    assert text.count("exc.code in {408,429} or 500 <= exc.code <= 599") == 2
    assert "Runtime admission OIDC refresh exhausted transient retries" in text
    assert "Runtime terminal OIDC refresh exhausted transient retries" in text


def test_m07_oidc_retry_does_not_weaken_broker_guards():
    text = Path(".github/workflows/pneumonia-v17-m07-continuation-20260914.yml").read_text(encoding="utf-8")
    assert "audience='+urllib.parse.quote('cgp-control-plane-v3',safe='')" in text
    assert "if value.get('ok') is not True or value.get('read_only') is not True:" in text
    assert "startsWith(github.event.inputs.recovery, 'M07_RUNTIME_M07_R')" in text
