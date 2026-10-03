from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import zipfile

import m07_rsna_pediatric_external_dispatch_v1 as legacy
import phase2_final_evidence_extract_v1 as evidence
import phase2_finalist_rsna_external_dispatch_v1 as dispatch


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_kernel_ref(contract: dict, kernel_ref: str) -> int:
    raw = str(kernel_ref or "").strip().strip("/")
    if raw.count("/") != 1:
        raise RuntimeError("HANDOFF_KERNEL_REF_INVALID")
    owner, slug = raw.split("/", 1)
    if owner.casefold() != str(contract["owner"]).casefold():
        raise RuntimeError("HANDOFF_KERNEL_OWNER_MISMATCH")
    model_id = str(contract["model_id"]).lower()
    resolution = int(contract["resolution"])
    prefix = f"phase2-finalist-rsna-{model_id}-r{resolution}-v1-"
    if not slug.casefold().startswith(prefix):
        raise RuntimeError("HANDOFF_KERNEL_SLUG_MISMATCH")
    suffix = slug[len(prefix):]
    if not suffix.isdigit():
        raise RuntimeError("HANDOFF_KERNEL_RUN_ID_INVALID")
    return int(suffix)


def choose_complete_kernel(
    candidates: list[tuple[str, str]],
) -> tuple[str, int]:
    completed = [
        (run_id, ref)
        for ref, status in candidates
        if status == "COMPLETE"
        for run_id in [int(ref.rsplit("-", 1)[1])]
    ]
    if not completed:
        states = {ref: status for ref, status in candidates}
        raise RuntimeError(
            "HANDOFF_COMPLETE_KERNEL_NOT_FOUND="
            + json.dumps(states, sort_keys=True)
        )
    run_id, ref = max(completed)
    return ref, run_id


def verify_zip_artifacts(
    zip_path: pathlib.Path,
    receipt: dict,
    model_id: str,
    resolution: int,
    selection_freeze_sha256: str,
) -> dict:
    manifest = receipt.get("artifact_sha256")
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("HANDOFF_RECEIPT_ARTIFACT_MANIFEST_MISSING")

    report_name = f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_REPORT.json"
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        by_normalized = {}
        for name in names:
            normalized = pathlib.PurePosixPath(name).as_posix().lstrip("./")
            by_normalized.setdefault(normalized, []).append(name)

        for relative_name, expected_sha in manifest.items():
            normalized = pathlib.PurePosixPath(str(relative_name)).as_posix().lstrip("./")
            matches = by_normalized.get(normalized) or []
            if len(matches) != 1:
                raise RuntimeError(
                    "HANDOFF_ZIP_MANIFEST_MEMBER_COUNT_INVALID:"
                    + normalized
                    + ":"
                    + str(len(matches))
                )
            actual_sha = sha256_bytes(archive.read(matches[0]))
            if actual_sha != str(expected_sha):
                raise RuntimeError(
                    "HANDOFF_ZIP_ARTIFACT_SHA_MISMATCH:" + normalized
                )

        report_matches = [
            name for name in names
            if pathlib.PurePosixPath(name).name == report_name
        ]
        if len(report_matches) != 1:
            raise RuntimeError(
                f"HANDOFF_REPORT_MEMBER_COUNT_INVALID:{len(report_matches)}"
            )
        report = json.loads(archive.read(report_matches[0]).decode("utf-8"))

    expected = {
        "schema": "pneumonia.phase2.finalist.external.rsna_pediatric.resolution.v1",
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "model_id": model_id,
        "resolution": resolution,
        "selection_freeze_sha256": selection_freeze_sha256,
        "scientific_classification": "POST_FREEZE_REPORT_ONLY_EXTERNAL_VALIDATION",
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
        "external_calibration_fitting": False,
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "external_may_reselect": False,
    }
    bad = {
        key: {"expected": wanted, "actual": report.get(key)}
        for key, wanted in expected.items()
        if report.get(key) != wanted
    }
    if bad:
        raise RuntimeError(
            "HANDOFF_EXTERNAL_REPORT_POLICY_INVALID="
            + json.dumps(bad, sort_keys=True)
        )
    for cohort in ("primary_pediatric_lt10", "expanded_pediatric_le18"):
        metrics = ((report.get(cohort) or {}).get("metrics"))
        if not isinstance(metrics, dict):
            raise RuntimeError("HANDOFF_EXTERNAL_METRICS_MISSING:" + cohort)
        for metric in ("accuracy", "macro_precision", "macro_recall"):
            if metric not in metrics:
                raise RuntimeError(
                    "HANDOFF_EXTERNAL_METRIC_MISSING:"
                    + cohort
                    + ":"
                    + metric
                )
    return report


def read_output_items(
    read_token: str,
    contract: dict,
    kernel_ref: str,
) -> list[dict]:
    account_id = str(contract["account_id"])
    owner, slug = kernel_ref.split("/", 1)
    items: list[dict] = []
    for page in range(1, 31):
        listing = legacy.post_json(
            legacy.READ_ENDPOINT,
            read_token,
            {
                "action": "raw_read",
                "account_id": account_id,
                "service": "kernels.KernelsApiService",
                "method": "ListKernelSessionOutput",
                "body": {
                    "userName": owner,
                    "kernelSlug": slug,
                    "page": page,
                    "pageSize": 100,
                },
            },
            timeout=120,
        )
        if not listing.get("ok"):
            raise RuntimeError(
                "HANDOFF_OUTPUT_LIST_FAILED:"
                + json.dumps(listing, sort_keys=True)[:2000]
            )
        result = evidence._read_payload(listing)
        files = result.get("files") or []
        items.extend(files)
        if len(files) < 100:
            break
    return items


def select_complete_output_candidate(
    candidates: list[tuple[str, int, list[dict]]],
    receipt_suffix: str,
    zip_suffix: str,
) -> tuple[str, int, dict, dict]:
    complete = []
    for kernel_ref, run_id, items in candidates:
        receipt_item = next(
            (
                item for item in items
                if str(item.get("fileName") or "").endswith(receipt_suffix)
                and item.get("url")
            ),
            None,
        )
        zip_item = next(
            (
                item for item in items
                if str(item.get("fileName") or "").endswith(zip_suffix)
                and item.get("url")
            ),
            None,
        )
        if receipt_item and zip_item:
            complete.append(
                (int(run_id), kernel_ref, receipt_item, zip_item)
            )
    if not complete:
        raise RuntimeError("HANDOFF_COMPLETE_OUTPUT_NOT_FOUND")
    run_id, kernel_ref, receipt_item, zip_item = max(complete)
    return kernel_ref, run_id, receipt_item, zip_item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--kernel-ref", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.token not in dispatch.CONTRACTS:
        raise SystemExit("HANDOFF_TOKEN_INVALID")
    contract = dispatch.CONTRACTS[args.token]
    model_id = str(contract["model_id"])
    resolution = int(contract["resolution"])
    root = pathlib.Path(__file__).resolve().parents[1]
    _, freeze_sha = dispatch.load_and_verify_selection_freeze(root)

    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(read_token) < 100:
        raise SystemExit("HANDOFF_READ_OIDC_TOKEN_INVALID")

    receipt_suffix = (
        f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_TERMINAL_RECEIPT.json"
    )
    zip_suffix = (
        f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_R{resolution}_V1_COMPLETE.zip"
    )

    candidate_outputs = []
    read_errors = {}
    ranked_refs = []
    for kernel_ref in args.kernel_ref:
        run_id = validate_kernel_ref(contract, kernel_ref)
        ranked_refs.append((run_id, kernel_ref))
    for run_id, kernel_ref in sorted(ranked_refs, reverse=True):
        try:
            items = read_output_items(read_token, contract, kernel_ref)
        except Exception as exc:
            read_errors[kernel_ref] = f"{type(exc).__name__}:{exc}"
            continue
        candidate_outputs.append((kernel_ref, run_id, items))

    try:
        kernel_ref, source_run_id, receipt_item, zip_item = (
            select_complete_output_candidate(
                candidate_outputs,
                receipt_suffix,
                zip_suffix,
            )
        )
    except RuntimeError as exc:
        raise RuntimeError(
            str(exc)
            + ":"
            + json.dumps(
                {
                    "candidates": [
                        {
                            "kernel_ref": ref,
                            "run_id": run_id,
                            "file_count": len(items),
                        }
                        for ref, run_id, items in candidate_outputs
                    ],
                    "read_errors": read_errors,
                },
                sort_keys=True,
            )
        ) from exc

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / receipt_suffix
    zip_path = out / zip_suffix
    legacy.safe_download(str(receipt_item["url"]), receipt_path)
    legacy.safe_download(str(zip_item["url"]), zip_path)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    dispatch.verify_receipt(
        receipt,
        contract,
        selection_freeze_sha256=freeze_sha,
    )
    report = verify_zip_artifacts(
        zip_path,
        receipt,
        model_id,
        resolution,
        freeze_sha,
    )

    handoff = {
        "schema": "pneumonia.phase2.finalist.external.github_handoff.v1",
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "model_id": model_id,
        "resolution": resolution,
        "account_id": str(contract["account_id"]),
        "owner": str(contract["owner"]),
        "kernel_ref": kernel_ref,
        "source_kernel_run_id": str(source_run_id),
        "state_handle": str(contract["state_handle"]),
        "compute_mode": "RECOVERED_EXISTING_COMPLETE",
        "external_dataset": dispatch.DATASET_REF,
        "external_manifest_sha256": dispatch.EXPECTED_MANIFEST_SHA256,
        "selection_freeze_sha256": freeze_sha,
        "terminal_receipt_sha256": receipt["receipt_sha256"],
        "complete_zip_sha256": sha256_file(zip_path),
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
        "external_calibration_fitting": False,
        "locked_test_used_for_selection": False,
        "external_used_for_selection": False,
        "external_may_reselect": False,
    }
    handoff_path = out / f"{model_id}_R{resolution}_RSNA_HANDOFF.json"
    handoff_path.write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = out / f"{model_id}_RSNA_PEDIATRIC_EXTERNAL_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "PHASE2_FINALIST_RSNA_HANDOFF_PASS",
        json.dumps(
            {
                "model_id": model_id,
                "resolution": resolution,
                "kernel_ref": kernel_ref,
                "source_kernel_run_id": source_run_id,
                "terminal_receipt_sha256": receipt["receipt_sha256"],
                "complete_zip_sha256": handoff["complete_zip_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
