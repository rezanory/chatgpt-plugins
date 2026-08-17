import json
import subprocess
from pathlib import Path

from chatgpt_plugin_kaggle.actions import dispatch_task
from chatgpt_plugins_github_bridge import parse_run_comment


def test_dispatch_task_builds_run_record_without_network(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("print('ok')\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "app.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "init"], check=True)
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()

    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "tests": {
                    "capabilities": ["cpu"],
                    "steps": [{"argv": ["python", "app.py"]}],
                    "internet": False,
                }
            }
        )
    )
    monkeypatch.setattr(dispatch_task, "create_private_dataset", lambda path: "created")
    monkeypatch.setattr(dispatch_task, "wait_dataset_ready", lambda ref: "ready")
    monkeypatch.setattr(dispatch_task, "push_kernel", lambda path, accelerator=None: "submitted")

    comment = tmp_path / "comment.md"
    diagnostic = tmp_path / "diagnostic.md"
    rc = dispatch_task.main(
        [
            "--job-id",
            "job-1",
            "--task-id",
            "task-1",
            "--account-id",
            "kg-01",
            "--account-environment",
            "kaggle-01",
            "--owner",
            "owner",
            "--source",
            str(source),
            "--source-repository",
            "owner/repo",
            "--source-commit",
            sha,
            "--profile",
            "tests",
            "--profiles-file",
            str(profiles),
            "--parameters-json",
            '{"seed":1}',
            "--github-run-id",
            "123",
            "--job-integrity-hash",
            "f" * 64,
            "--comment-file",
            str(comment),
            "--diagnostic-file",
            str(diagnostic),
        ]
    )
    assert rc == 0
    assert not diagnostic.exists()
    record = parse_run_comment(comment.read_text())
    assert record.source_commit == sha
    assert record.account_environment == "kaggle-01"
    assert record.kernel_ref.startswith("owner/cgp-run-")
