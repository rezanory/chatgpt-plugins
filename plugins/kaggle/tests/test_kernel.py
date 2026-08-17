import json
from pathlib import Path

from chatgpt_plugin_kaggle.config import ExecutionProfile, ExecutionStep
from chatgpt_plugin_kaggle.kernel import build_bootstrap, write_kernel_package


def test_kernel_metadata_is_private_and_shell_free(tmp_path: Path):
    profile = ExecutionProfile(
        "tests",
        "",
        frozenset({"cpu"}),
        (ExecutionStep(("python", "-m", "pytest", "-q")),),
        (),
        internet=False,
    )
    bootstrap = build_bootstrap(
        job_id="j1",
        task_id="t1",
        dataset_slug="src-j1",
        dataset_ref="o/src-j1",
        kernel_ref="o/run-j1",
        source_ref="a" * 40,
        profile=profile,
        parameters={"seed": 1, "amp": True},
    )
    write_kernel_package(
        tmp_path,
        owner="o",
        slug="run-j1",
        title="CGP Run j1",
        dataset_ref="o/src-j1",
        bootstrap=bootstrap,
        profile=profile,
    )
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["o/src-j1"]
    assert metadata["enable_internet"] is False
    assert "shell=False" in bootstrap
    assert "callback" not in bootstrap.lower()
    compile(bootstrap, "bootstrap.py", "exec")
