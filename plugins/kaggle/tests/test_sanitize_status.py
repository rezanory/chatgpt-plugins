from chatgpt_plugin_kaggle.sanitize import sanitize_text
from chatgpt_plugin_kaggle.status import normalize_kernel_status


def test_sanitizer_redacts_tokens_and_ansi():
    text = "\x1b[31mKAGGLE_API_TOKEN=KGAT_supersecretvalue\x1b[0m"
    cleaned = sanitize_text(text)
    assert "supersecretvalue" not in cleaned
    assert "\\x1b" not in repr(cleaned)
    assert "[REDACTED]" in cleaned


def test_status_normalization_is_conservative():
    assert normalize_kernel_status("Status: running").state == "running"
    assert normalize_kernel_status("Status: complete").state == "succeeded"
    assert normalize_kernel_status("Status: error").state == "failed"
    assert normalize_kernel_status("some new status").state == "unknown"
