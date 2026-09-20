from pathlib import Path

W = Path(".github/workflows/control-plane-v3-query.yml")


def test_query_result_is_durable_before_best_effort_artifact_transport():
    text = W.read_text(encoding="utf-8")
    execute = text.index("- name: Execute provider read query")
    marker = text.index("CGP_QUERY_RESULT_BEGIN", execute)
    upload = text.index("- name: Upload durable query result artifact", marker)
    assert execute < marker < upload
    block = text[upload:upload + 320]
    assert "if: always()" in block
    assert "continue-on-error: true" in block
    assert "timeout-minutes: 2" in block
    assert "uses: actions/upload-artifact@v4" in block


def test_provider_query_exit_is_not_redefined_by_artifact_transport():
    text = W.read_text(encoding="utf-8")
    execute = text.index("- name: Execute provider read query")
    upload = text.index("- name: Upload durable query result artifact", execute)
    execute_block = text[execute:upload]
    assert "CGP_QUERY_EXIT=$code" in execute_block
    assert "CGP_QUERY_RESULT_BEGIN" in execute_block
    assert "exit 0" in execute_block
