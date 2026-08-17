from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExecutionProfile


def build_bootstrap(
    *,
    job_id: str,
    task_id: str,
    dataset_slug: str,
    dataset_ref: str,
    kernel_ref: str,
    source_ref: str,
    profile: ExecutionProfile,
    parameters: dict[str, str | int | float | bool] | None = None,
) -> str:
    """Create the code run by Kaggle.

    Crucially, commands are emitted as argv arrays from the trusted profile registry and executed
    with ``shell=False``. Issue/job fields never become shell fragments.
    """

    steps_json = json.dumps([{"argv": list(step.argv), "cwd": step.cwd} for step in profile.steps])
    globs_json = json.dumps(list(profile.artifact_globs))
    parameters_json = json.dumps(parameters or {}, sort_keys=True)
    return f"""\
from __future__ import annotations
import glob, json, os, pathlib, subprocess, traceback, zipfile

JOB_ID = {job_id!r}
TASK_ID = {task_id!r}
DATASET_REF = {dataset_ref!r}
KERNEL_REF = {kernel_ref!r}
SOURCE_REF = {source_ref!r}
STEPS = json.loads({steps_json!r})
ARTIFACT_GLOBS = json.loads({globs_json!r})
PARAMETERS = json.loads({parameters_json!r})
WORKING = pathlib.Path('/kaggle/working/chatgpt-plugin')
SOURCE = WORKING / 'source'
LOG = WORKING / 'job.log'
RESULT = WORKING / 'result.json'
PARAMS = WORKING / 'job-parameters.json'
WORKING.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)


def safe_extract(zf, destination):
    base = destination.resolve()
    for info in zf.infolist():
        candidate = (destination / info.filename).resolve()
        if candidate != base and base not in candidate.parents:
            raise RuntimeError('unsafe path in source archive')
    zf.extractall(destination)


def run():
    source_zip = pathlib.Path('/kaggle/input/{dataset_slug}/source.zip')
    if not source_zip.exists():
        raise FileNotFoundError(f'source archive not mounted: {{source_zip}}')
    with zipfile.ZipFile(source_zip) as zf:
        safe_extract(zf, SOURCE)
    PARAMS.write_text(json.dumps(PARAMETERS, indent=2, sort_keys=True), encoding='utf-8')
    env = os.environ.copy()
    env['CHATGPT_PLUGINS_JOB_ID'] = JOB_ID
    env['CHATGPT_PLUGINS_TASK_ID'] = TASK_ID
    env['CHATGPT_PLUGINS_SOURCE_COMMIT'] = SOURCE_REF
    env['CHATGPT_PLUGINS_PARAMETERS'] = str(PARAMS)
    with LOG.open('w', encoding='utf-8') as log:
        for step in STEPS:
            argv = step['argv']
            cwd = (SOURCE / step.get('cwd', '.')).resolve()
            if cwd != SOURCE.resolve() and SOURCE.resolve() not in cwd.parents:
                raise RuntimeError('profile cwd escaped source root')
            log.write('$ ' + json.dumps(argv) + '\\n')
            log.flush()
            proc = subprocess.run(
                argv,
                cwd=cwd,
                shell=False,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(f'step failed with exit code {{proc.returncode}}: {{argv[0]}}')
    artifacts = []
    for pattern in ARTIFACT_GLOBS:
        for matched in glob.glob(str(WORKING / pattern), recursive=True):
            p = pathlib.Path(matched).resolve()
            if p.is_file() and (p == WORKING.resolve() or WORKING.resolve() in p.parents):
                artifacts.append(str(p.relative_to(pathlib.Path('/kaggle/working'))))
    return sorted(set(artifacts))[:200]


try:
    artifacts = run()
    payload = {{
        'schema': 'chatgpt.compute.result/v1',
        'job_id': JOB_ID,
        'task_id': TASK_ID,
        'status': 'succeeded',
        'summary': 'profile completed',
        'source_commit': SOURCE_REF,
        'dataset_ref': DATASET_REF,
        'kernel_ref': KERNEL_REF,
        'artifacts': artifacts,
    }}
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
except Exception as exc:
    trace = traceback.format_exc()
    previous = LOG.read_text(encoding='utf-8', errors='replace') if LOG.exists() else ''
    excerpt = (previous + '\\n' + trace)[-16000:]
    payload = {{
        'schema': 'chatgpt.compute.result/v1',
        'job_id': JOB_ID,
        'task_id': TASK_ID,
        'status': 'failed',
        'summary': str(exc)[:1000],
        'source_commit': SOURCE_REF,
        'dataset_ref': DATASET_REF,
        'kernel_ref': KERNEL_REF,
        'log_excerpt': excerpt,
        'artifacts': [],
    }}
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    raise
"""


def write_kernel_package(
    path: Path,
    *,
    owner: str,
    slug: str,
    title: str,
    dataset_ref: str,
    bootstrap: str,
    profile: ExecutionProfile,
    accelerator: str | None = None,
) -> str:
    path.mkdir(parents=True, exist_ok=True)
    code_file = "bootstrap.py"
    (path / code_file).write_text(bootstrap, encoding="utf-8")
    kernel_ref = f"{owner}/{slug}"
    use_gpu = "gpu" in profile.capabilities or bool(accelerator)
    metadata: dict[str, Any] = {
        "id": kernel_ref,
        "title": title[:50],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": use_gpu,
        "enable_tpu": False,
        "enable_internet": bool(profile.internet),
        "dataset_sources": [dataset_ref],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (path / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return kernel_ref
