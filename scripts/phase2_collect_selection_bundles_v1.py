from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile

READ_ENDPOINT = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control-plane/v3/read/kaggle"
ALLOWED_OUTPUT_HOST = "www.kaggleusercontent.com"
MAX_BUNDLE_BYTES = 150_000_000
MAX_MEMBERS_PER_BUNDLE = 100
PREDICTION_MEMBER = re.compile(
    r"^(M(?:0[1-9]|1[0-2]))/R(224|320|384)/(OOF_PREDICTIONS\.csv|LOCKED_TEST_PREDICTIONS\.csv|FINAL_REPORT\.json)$"
)
REQUIRED_MODELS = tuple(f"M{i:02d}" for i in range(1, 13))
REQUIRED_RESOLUTIONS = (224, 320, 384)


class SelectionBundleError(RuntimeError):
    pass


def post_json(payload: dict, token: str, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        READ_ENDPOINT,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "phase2-selection-bundle-collector/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "http_status": exc.code,
            "error": exc.read(12000).decode("utf-8", "replace"),
        }


def find_output_url(token: str, account_id: str, kernel_ref: str, file_name: str) -> str:
    owner, slug = kernel_ref.split("/", 1)
    matches = []
    page_token = ""
    seen_tokens = set()
    for _ in range(20):
        body = {"userName": owner, "kernelSlug": slug, "pageSize": 200}
        if page_token:
            body["pageToken"] = page_token
        response = post_json(
            {
                "action": "raw_read",
                "account_id": account_id,
                "service": "kernels.KernelsApiService",
                "method": "ListKernelSessionOutput",
                "body": body,
            },
            token,
        )
        if not response.get("ok"):
            raise SelectionBundleError(
                f"OUTPUT_LIST_FAILED:{account_id}:{kernel_ref}:{response}"
            )
        result = response.get("result") or {}
        for item in result.get("files") or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("fileName") or "")
            if source.split("/")[-1] == file_name:
                matches.append(item)
        next_token = str(result.get("nextPageToken") or "").strip()
        if not next_token:
            break
        if next_token in seen_tokens:
            raise SelectionBundleError(f"OUTPUT_PAGINATION_TOKEN_REPEATED:{kernel_ref}")
        seen_tokens.add(next_token)
        page_token = next_token
    if len(matches) != 1:
        raise SelectionBundleError(
            f"OUTPUT_BUNDLE_MATCH_COUNT_INVALID:{account_id}:{kernel_ref}:{file_name}:{len(matches)}"
        )
    raw_url = str(matches[0].get("url") or "")
    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_OUTPUT_HOST:
        raise SelectionBundleError(
            f"OUTPUT_BUNDLE_URL_HOST_INVALID:{account_id}:{kernel_ref}:{file_name}"
        )
    return raw_url


def download_bundle(url: str, expected_bytes: int, expected_sha256: str) -> bytes:
    if expected_bytes <= 0 or expected_bytes > MAX_BUNDLE_BYTES:
        raise SelectionBundleError(f"BUNDLE_EXPECTED_SIZE_INVALID:{expected_bytes}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise SelectionBundleError("BUNDLE_EXPECTED_SHA_INVALID")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_OUTPUT_HOST:
        raise SelectionBundleError("BUNDLE_DOWNLOAD_HOST_INVALID")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "phase2-selection-bundle-collector/1.0"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        data = response.read(MAX_BUNDLE_BYTES + 1)
    if len(data) > MAX_BUNDLE_BYTES:
        raise SelectionBundleError("BUNDLE_DOWNLOAD_TOO_LARGE")
    if len(data) != expected_bytes:
        raise SelectionBundleError(
            f"BUNDLE_SIZE_MISMATCH:{len(data)}:{expected_bytes}"
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise SelectionBundleError(
            f"BUNDLE_SHA_MISMATCH:{digest}:{expected_sha256}"
        )
    return data


def _safe_zip_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = pathlib.PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or normalized.startswith("/")
        or ":" in path.parts[0]
    ):
        raise SelectionBundleError(f"BUNDLE_MEMBER_PATH_UNSAFE:{name}")
    return path.as_posix()


def validate_and_extract_bundle_bytes(
    data: bytes,
    account_id: str,
    output_root: pathlib.Path,
) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) < 3 or len(members) > MAX_MEMBERS_PER_BUNDLE:
            raise SelectionBundleError(
                f"BUNDLE_MEMBER_COUNT_INVALID:{account_id}:{len(members)}"
            )
        names = [_safe_zip_name(info.filename) for info in members]
        if len(names) != len(set(names)):
            raise SelectionBundleError(f"BUNDLE_DUPLICATE_MEMBER:{account_id}")
        manifest_names = [name for name in names if name == "BUNDLE_MANIFEST.json"]
        if len(manifest_names) != 1:
            raise SelectionBundleError(
                f"BUNDLE_MANIFEST_COUNT_INVALID:{account_id}:{len(manifest_names)}"
            )
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json").decode("utf-8"))
        if (
            manifest.get("schema")
            != "pneumonia.phase2.selection_prediction_bundle.v1"
            or manifest.get("status") != "PASS"
            or manifest.get("account_id") != account_id
            or manifest.get("training_performed") is not False
            or manifest.get("hpo_performed") is not False
            or manifest.get("locked_test_executed") is not False
            or manifest.get("external_validation_executed") is not False
        ):
            raise SelectionBundleError(f"BUNDLE_MANIFEST_POLICY_INVALID:{account_id}")

        manifest_rows = manifest.get("files") or []
        by_path = {}
        for row in manifest_rows:
            if not isinstance(row, dict):
                raise SelectionBundleError(f"BUNDLE_MANIFEST_ROW_INVALID:{account_id}")
            path = _safe_zip_name(str(row.get("path") or ""))
            if path in by_path:
                raise SelectionBundleError(
                    f"BUNDLE_MANIFEST_DUPLICATE_PATH:{account_id}:{path}"
                )
            by_path[path] = row

        prediction_names = [name for name in names if name != "BUNDLE_MANIFEST.json"]
        if set(prediction_names) != set(by_path):
            raise SelectionBundleError(f"BUNDLE_MANIFEST_INVENTORY_MISMATCH:{account_id}")

        extracted = []
        for name in sorted(prediction_names):
            match = PREDICTION_MEMBER.fullmatch(name)
            if not match:
                raise SelectionBundleError(
                    f"BUNDLE_MEMBER_NOT_CANONICAL_SELECTION_ARTIFACT:{account_id}:{name}"
                )
            info = archive.getinfo(name)
            if info.file_size > 25_000_000:
                raise SelectionBundleError(
                    f"BUNDLE_MEMBER_UNCOMPRESSED_TOO_LARGE:{account_id}:{name}:{info.file_size}"
                )
            payload = archive.read(name)
            row = by_path[name]
            expected_bytes = row.get("bytes")
            expected_sha = str(row.get("sha256") or "")
            actual_sha = hashlib.sha256(payload).hexdigest()
            if expected_bytes != len(payload) or expected_sha != actual_sha:
                raise SelectionBundleError(
                    f"BUNDLE_MEMBER_INTEGRITY_MISMATCH:{account_id}:{name}"
                )
            target = output_root.joinpath(*pathlib.PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                existing_sha = hashlib.sha256(target.read_bytes()).hexdigest()
                if existing_sha != actual_sha:
                    raise SelectionBundleError(f"BUNDLE_EXTRACTION_COLLISION:{name}")
            else:
                target.write_bytes(payload)
            extracted.append(
                {
                    "account_id": account_id,
                    "path": name,
                    "bytes": len(payload),
                    "sha256": actual_sha,
                }
            )
        return extracted


def validate_prediction_coverage(output_root: pathlib.Path) -> dict:
    expected = {
        f"{model}/R{resolution}/{kind}_PREDICTIONS.csv"
        for model in REQUIRED_MODELS
        for resolution in REQUIRED_RESOLUTIONS
        for kind in ("OOF", "LOCKED_TEST")
    }
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*_PREDICTIONS.csv")
        if path.is_file()
    }
    if actual != expected:
        raise SelectionBundleError(
            "PREDICTION_COVERAGE_INVALID="
            + json.dumps(
                {
                    "count": len(actual),
                    "missing": sorted(expected - actual),
                    "extra": sorted(actual - expected),
                },
                sort_keys=True,
            )
        )
    return {
        "models": len(REQUIRED_MODELS),
        "resolutions_per_model": len(REQUIRED_RESOLUTIONS),
        "prediction_csv_count": len(actual),
        "expected_prediction_csv_count": len(expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()

    token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(token) < 100:
        raise SystemExit("READ_OIDC_TOKEN_INVALID")
    matrix = json.loads(pathlib.Path(args.matrix).read_text(encoding="utf-8"))
    if matrix.get("schema") != "pneumonia.phase2.final_evidence.matrix.v1" or matrix.get("status") != "PASS":
        raise SelectionBundleError("FINAL_EVIDENCE_MATRIX_INVALID")
    bundle_rows = matrix.get("selection_prediction_bundles") or []
    if len(bundle_rows) != 11:
        raise SelectionBundleError(
            f"SELECTION_BUNDLE_ACCOUNT_COUNT_INVALID:{len(bundle_rows)}"
        )

    output_root = pathlib.Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    extracted = []
    accounts = set()
    for row in sorted(bundle_rows, key=lambda item: str(item.get("account_id") or "")):
        account_id = str(row.get("account_id") or "")
        kernel_ref = str(row.get("kernel_ref") or "")
        file_name = str(row.get("file_name") or "")
        expected_sha = str(row.get("sha256") or "")
        expected_bytes = row.get("bytes")
        if (
            not account_id
            or account_id in accounts
            or kernel_ref.count("/") != 1
            or not file_name
            or not isinstance(expected_bytes, int)
        ):
            raise SelectionBundleError(f"SELECTION_BUNDLE_ROW_INVALID:{account_id}")
        accounts.add(account_id)
        url = find_output_url(token, account_id, kernel_ref, file_name)
        data = download_bundle(url, expected_bytes, expected_sha)
        extracted.extend(
            validate_and_extract_bundle_bytes(data, account_id, output_root)
        )

    coverage = validate_prediction_coverage(output_root)
    manifest = {
        "schema": "pneumonia.phase2.selection_prediction_collection.v1",
        "status": "PASS",
        "accounts": len(accounts),
        **coverage,
        "training_performed": False,
        "hpo_performed": False,
        "locked_test_executed": False,
        "external_validation_executed": False,
        "files": sorted(extracted, key=lambda item: item["path"]),
    }
    target = pathlib.Path(args.manifest_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "PHASE2_SELECTION_PREDICTION_COLLECTION_PASS="
        + json.dumps(
            {
                "accounts": manifest["accounts"],
                "prediction_csv_count": manifest["prediction_csv_count"],
                "files": len(extracted),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
