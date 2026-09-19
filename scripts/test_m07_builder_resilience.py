from pathlib import Path

WORKFLOW = Path(".github/workflows/pneumonia-v17-m07-continuation-20260914.yml")

def test_builder_read_retry():
    text=WORKFLOW.read_text(encoding="utf-8")
    assert "M07_BUILDER_GITHUB_READ_RETRY" in text
    assert "M07 builder GitHub evidence read exhausted transient retries" in text
    assert "exc.code not in {408, 429} and not 500 <= exc.code <= 599" in text
    assert "except (urllib.error.URLError, TimeoutError, OSError) as exc:" in text

def test_builder_native_failure_capture():
    text=WORKFLOW.read_text(encoding="utf-8")
    assert "$previousErrorActionPreference=$ErrorActionPreference" in text
    assert "$ErrorActionPreference='Continue'" in text
    assert "$ErrorActionPreference=$previousErrorActionPreference" in text
    assert "M07_CONTINUATION_BUILDER_FAILURE=" in text

def test_launch_comment_failure_non_fatal():
    text=WORKFLOW.read_text(encoding="utf-8")
    start=text.index("- name: Publish continuation launch receipt")
    assert "continue-on-error: true" in text[start:start+400]
