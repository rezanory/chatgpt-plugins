from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from chatgpt_plugins_github_bridge import RunRecord, render_run_comment

from ..config import load_profiles
from ..kaggle_cli import create_private_dataset, push_kernel, wait_dataset_ready
from ..kernel import build_bootstrap, write_kernel_package
from ..package_source import package_source, write_dataset_metadata
from ..sanitize import sanitize_text


def resolve_commit(source_dir: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def _slug(prefix: str, job_id: str, task_id: str) -> str:
    raw = f"{prefix}-{job_id}-{task_id}".lower()
    safe = "".join(c if c.isalnum() else "-" for c in raw)
    while "--" in safe:
        safe = safe.replace("--", "-")
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{safe.strip('-')[:38]}-{digest}"[:49]


def _write_diagnostic(path: str, *, stage: str, exc: Exception) -> None:
    safe = sanitize_text(str(exc), max_chars=4_000).strip() or exc.__class__.__name__
    body = (
        f"Kaggle submit diagnostic: substage=`{stage}`; error_type=`{exc.__class__.__name__}`.\n\n"
        "```text\n"
        f"{safe}\n"
        "```\n"
    )
    Path(path).write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--account-environment", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profiles-file", required=True)
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--accelerator", default="")
    parser.add_argument("--github-run-id", required=True)
    parser.add_argument("--job-integrity-hash", required=True)
    parser.add_argument("--comment-file", required=True)
    parser.add_argument("--diagnostic-file", required=True)
    args = parser.parse_args(argv)

    stage = "resolve-source"
    try:
        source_dir = Path(args.source).resolve()
        resolved_sha = resolve_commit(source_dir)
        if resolved_sha.lower() != args.source_commit.lower():
            raise RuntimeError("checked-out commit does not match immutable job contract")

        stage = "load-profile"
        profiles = load_profiles(args.profiles_file)
        profile = profiles.get(args.profile)
        if profile is None:
            raise RuntimeError(f"unknown profile: {args.profile}")
        parameters = json.loads(args.parameters_json)
        if not isinstance(parameters, dict):
            raise RuntimeError("parameters-json must be an object")

        dataset_slug = _slug("cgp-src", args.job_id, args.task_id)
        kernel_slug = _slug("cgp-run", args.job_id, args.task_id)
        with tempfile.TemporaryDirectory(prefix=f"cgp-{args.job_id}-{args.task_id}-") as temp:
            temp_dir = Path(temp)
            dataset_dir = temp_dir / "dataset"
            kernel_dir = temp_dir / "kernel"
            dataset_dir.mkdir()
            kernel_dir.mkdir()

            stage = "package-source"
            package_source(source_dir, dataset_dir / "source.zip")
            dataset_ref = write_dataset_metadata(
                dataset_dir,
                owner=args.owner,
                slug=dataset_slug,
                job_id=f"{args.job_id}-{args.task_id}",
            )

            stage = "create-dataset"
            create_private_dataset(dataset_dir)

            stage = "wait-dataset"
            wait_dataset_ready(dataset_ref)

            stage = "build-kernel"
            kernel_ref = f"{args.owner}/{kernel_slug}"
            bootstrap = build_bootstrap(
                job_id=args.job_id,
                task_id=args.task_id,
                dataset_slug=dataset_slug,
                dataset_ref=dataset_ref,
                kernel_ref=kernel_ref,
                source_ref=resolved_sha,
                profile=profile,
                parameters=parameters,
            )
            write_kernel_package(
                kernel_dir,
                owner=args.owner,
                slug=kernel_slug,
                title=f"CGP {args.job_id} {args.task_id}",
                dataset_ref=dataset_ref,
                bootstrap=bootstrap,
                profile=profile,
                accelerator=args.accelerator or None,
            )

            stage = "push-kernel"
            push_kernel(kernel_dir, accelerator=args.accelerator or None)

        stage = "render-run-record"
        record = RunRecord(
            schema="chatgpt.compute.run/v1",
            job_id=args.job_id,
            task_id=args.task_id,
            account_id=args.account_id,
            account_environment=args.account_environment,
            kernel_ref=kernel_ref,
            dataset_ref=dataset_ref,
            source_repository=args.source_repository,
            source_commit=resolved_sha,
            profile=args.profile,
            state="submitted",
            github_run_id=args.github_run_id,
            integrity_hash=args.job_integrity_hash,
        )
        Path(args.comment_file).write_text(render_run_comment(record), encoding="utf-8")
        print(
            json.dumps(
                {"kernel_ref": kernel_ref, "dataset_ref": dataset_ref, "task_id": args.task_id}
            )
        )
        return 0
    except Exception as exc:
        _write_diagnostic(args.diagnostic_file, stage=stage, exc=exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
