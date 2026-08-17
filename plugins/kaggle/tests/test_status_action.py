import json
from pathlib import Path

from chatgpt_plugins_github_bridge import parse_status_comment
from chatgpt_plugin_kaggle.actions import status_task


def test_failed_status_generates_sanitized_hashed_evidence(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(status_task, "kernel_status", lambda ref: "Status: failed")

    def fake_output(ref, path, file_pattern=None):
        target = Path(path) / "chatgpt-plugin"
        target.mkdir(parents=True)
        (target / "job.log").write_text(
            "KAGGLE_API_TOKEN=KGAT_supersecretvalue\\nModuleNotFoundError: No module named 'x'"
        )
        (target / "result.json").write_text(json.dumps({
            "status": "failed",
            "summary": "ModuleNotFoundError: No module named 'x'",
        }))
        return "ok"

    monkeypatch.setattr(status_task, "kernel_output", fake_output)
    output = tmp_path / "evidence"
    comment = tmp_path / "status.md"
    rc = status_task.main([
        "--job-id", "job-1",
        "--task-id", "task-1",
        "--account-id", "kg-01",
        "--kernel-ref", "owner/kernel",
        "--github-run-id", "456",
        "--artifact-name", "artifact-1",
        "--output-dir", str(output),
        "--comment-file", str(comment),
    ])
    assert rc == 0
    record = parse_status_comment(comment.read_text())
    assert record.state == "failed"
    assert record.failure_category == "import_error"
    assert record.failure_fingerprint
    assert "supersecretvalue" not in (output / "failure.json").read_text()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["files"]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
