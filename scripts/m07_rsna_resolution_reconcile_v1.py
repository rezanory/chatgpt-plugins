from __future__ import annotations

import argparse
import json
import os
import pathlib
import urllib.parse
import urllib.request

from m07_rsna_age_audit_v1 import post_json

RUNS = {
    224: {
        "account_id": "kg-03",
        "kernel_ref": "rezanory/m07-rsna-pediatric-external-r224-v1-36572524789",
    },
    320: {
        "account_id": "kg-05",
        "kernel_ref": "trickermark/m07-rsna-pediatric-external-r320-v1-36587612802",
    },
    384: {
        "account_id": "kg-05",
        "kernel_ref": "trickermark/m07-rsna-pediatric-external-r384-v1-36588025918",
    },
}
REPORT = "M07_RSNA_PEDIATRIC_EXTERNAL_REPORT.json"
RECEIPT = "M07_RSNA_PEDIATRIC_EXTERNAL_TERMINAL_RECEIPT.json"


def output_inventory(account_id: str, kernel_ref: str, token: str) -> list[dict]:
    owner, slug = kernel_ref.split("/", 1)
    result = post_json(
        {
            "action": "raw_read",
            "account_id": account_id,
            "service": "kernels.KernelsApiService",
            "method": "ListKernelSessionOutput",
            "body": {
                "userName": owner,
                "kernelSlug": slug,
                "page": 1,
                "pageSize": 100,
            },
        },
        token,
    )
    files = result.get("files")
    if not isinstance(files, list):
        raise RuntimeError("kernel output inventory missing files")
    return [item for item in files if isinstance(item, dict)]


def safe_output_url(files: list[dict], suffix: str) -> str:
    hits = [
        item
        for item in files
        if str(item.get("fileName") or "").endswith(suffix) and item.get("url")
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one {suffix}, found {len(hits)}")
    url = str(hits[0]["url"])
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed = (
        host == "www.kaggleusercontent.com"
        or host.endswith(".kaggleusercontent.com")
        or host.endswith(".googleusercontent.com")
        or host == "storage.googleapis.com"
        or host in {"api.kaggle.com", "www.kaggle.com"}
    )
    if parsed.scheme != "https" or not allowed:
        raise RuntimeError("output URL host not allowlisted")
    return url


def download_json(url: str, max_bytes: int = 2_000_000) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "m07-rsna-resolution-reconcile-v1/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise RuntimeError("JSON output exceeded bounded size")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON output must be an object")
    return value


def extract(resolution: int, receipt: dict, report: dict) -> dict:
    if receipt.get("status") != "SCIENTIFIC_RECEIPT_PASS":
        raise RuntimeError(f"R{resolution} receipt status is not PASS")
    if int(receipt.get("resolution", -1)) != resolution:
        raise RuntimeError(f"R{resolution} receipt resolution mismatch")
    if report.get("status") != "SCIENTIFIC_RECEIPT_PASS":
        raise RuntimeError(f"R{resolution} report status is not PASS")
    if int(report.get("resolution", -1)) != resolution:
        raise RuntimeError(f"R{resolution} report resolution mismatch")
    for key in (
        "training_performed",
        "hpo_performed",
        "external_threshold_tuning",
        "external_adaptation",
        "external_calibration_fitting",
    ):
        if receipt.get(key) is not False or report.get(key) is not False:
            raise RuntimeError(f"R{resolution} scientific purity violation: {key}")
    if int(receipt.get("expanded_n", -1)) != 1099:
        raise RuntimeError(f"R{resolution} expanded_n mismatch")
    if int(receipt.get("primary_n", -1)) != 282:
        raise RuntimeError(f"R{resolution} primary_n mismatch")
    expanded = ((report.get("expanded_pediatric_le18") or {}).get("metrics") or {})
    primary = ((report.get("primary_pediatric_lt10") or {}).get("metrics") or {})
    required = (
        "accuracy",
        "balanced_accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "auroc",
        "auprc_positive",
        "tn",
        "fp",
        "fn",
        "tp",
    )
    for name, metrics in (("expanded", expanded), ("primary", primary)):
        missing = [key for key in required if key not in metrics]
        if missing:
            raise RuntimeError(f"R{resolution} {name} metrics missing {missing}")
    return {
        "resolution": resolution,
        "status": "SCIENTIFIC_RECEIPT_PASS",
        "receipt_sha256": receipt.get("receipt_sha256"),
        "source_state": receipt.get("source_state"),
        "expanded": {key: expanded[key] for key in required},
        "primary": {key: primary[key] for key in required},
    }


def reconcile(token: str) -> dict:
    rows = {}
    for resolution, spec in RUNS.items():
        files = output_inventory(spec["account_id"], spec["kernel_ref"], token)
        receipt = download_json(safe_output_url(files, RECEIPT))
        report = download_json(safe_output_url(files, REPORT))
        rows[str(resolution)] = extract(resolution, receipt, report)
    return {
        "schema": "m07.external.rsna.resolution_reconcile.v1",
        "status": "PASS_ALL_RESOLUTIONS",
        "pure_external": True,
        "training_performed": False,
        "external_tuning_performed": False,
        "resolutions": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    token = os.environ.get("CGP_READ_OIDC_TOKEN", "").strip()
    if len(token) < 100:
        raise SystemExit("CGP_READ_OIDC_TOKEN unavailable")
    result = reconcile(token)
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("M07_RSNA_RESOLUTION_RECONCILE=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
