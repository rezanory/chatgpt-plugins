from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from chatgpt_plugins_github_bridge import StatusRecord, render_status_comment

from ..kaggle_cli import kernel_output, kernel_status
from ..sanitize import sanitize_text
from ..status import classify_result_failure, load_result_files, normalize_kernel_status
from .common import write_github_output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--github-run-id", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comment-file", required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    normalized = normalize_kernel_status(kernel_status(args.kernel_ref))
    result: dict = {}
    log = ""
    failure_category = None
    failure_fingerprint = None
    summary = normalized.raw[-4000:]

    if normalized.state in {"succeeded", "failed"}:
        try:
            kernel_output(
                args.kernel_ref,
                output_dir,
                file_pattern=r".*(result\.json|job\.log)$",
            )
        except Exception as exc:
            # Status itself is still useful even if output collection is temporarily unavailable.
            (output_dir / "output-download-error.txt").write_text(
                sanitize_text(str(exc), max_chars=8000), encoding="utf-8"
            )
        result, log = load_result_files(output_dir)
        if result.get("status") in {"succeeded", "failed"}:
            normalized = normalize_kernel_status(str(result.get("status")))
        summary = sanitize_text(str(result.get("summary") or summary), max_chars=4000)
        if normalized.state == "failed":
            failure = classify_result_failure(result, log)
            failure_category = failure.category.value
            failure_fingerprint = failure.fingerprint
            (output_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "category": failure.category.value,
                        "confidence": failure.confidence,
                        "retryable": failure.retryable,
                        "fingerprint": failure.fingerprint,
                        "summary": failure.summary,
                        "log_excerpt": failure.log_excerpt,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    manifest = {
        "schema": "chatgpt.compute.artifacts/v1",
        "job_id": args.job_id,
        "task_id": args.task_id,
        "account_id": args.account_id,
        "kernel_ref": args.kernel_ref,
        "state": normalized.state,
        "files": [],
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"].append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    record = StatusRecord(
        schema="chatgpt.compute.status/v1",
        job_id=args.job_id,
        task_id=args.task_id,
        account_id=args.account_id,
        kernel_ref=args.kernel_ref,
        state=normalized.state,
        failure_category=failure_category,
        failure_fingerprint=failure_fingerprint,
        summary=summary,
        artifact_name=args.artifact_name if normalized.state in {"succeeded", "failed"} else None,
        github_run_id=args.github_run_id,
    )
    Path(args.comment_file).write_text(render_status_comment(record), encoding="utf-8")
    write_github_output("state", normalized.state)
    write_github_output("terminal", "true" if normalized.state in {"succeeded", "failed"} else "false")
    print(json.dumps({"state": normalized.state, "artifact_name": record.artifact_name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
