from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev"
TOKEN = os.environ["CGP_READ_OIDC_TOKEN"]
ASSIGNMENTS = {
    "M01": ("master", "azadka"),
    "M02": ("kg-02", "radlinaradlina"),
    "M03": ("kg-03", "rezanory"),
    "M04": ("kg-04", "reyhanehazad"),
    "M05": ("kg-05", "trickermark"),
    "M06": ("kg-06", "msdenis"),
    "M08": ("kg-07", "nisabulutmark"),
    "M09": ("kg-08", "azadkk"),
    "M10": ("kg-09", "mylovevpn1"),
    "M11": ("kg-10", "computstu1"),
    "M12": ("kg-11", "jobreza1"),
}
RESOLUTIONS = (224, 320, 384)


def broker(payload: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        BASE + "/control-plane/v3/read/kaggle",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + TOKEN,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "phase2-historical-matrix-audit-v2/1.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            out = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(4000).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    if not out.get("ok"):
        raise RuntimeError(str(out.get("error") or "broker not ok"))
    result = out.get("result")
    return result if isinstance(result, dict) else {}


def inspect(model_id: str, resolution: int) -> dict:
    account_id, owner = ASSIGNMENTS[model_id]
    slug = f"pneumonia-{model_id.lower()}-r{resolution}-state-v1-7"
    handle = f"{owner}/{slug}"
    row = {
        "model_id": model_id,
        "resolution": resolution,
        "account_id": account_id,
        "owner": owner,
        "dataset_ref": handle,
        "exists": False,
        "marker_status": None,
        "completed_folds": [],
        "fold_count": 0,
        "has_final_evidence": False,
        "resolution_complete": False,
    }
    listed = broker(
        {
            "action": "raw_read",
            "account_id": account_id,
            "service": "datasets.DatasetApiService",
            "method": "ListDatasets",
            "body": {"group": "MY", "search": slug, "page": 1, "pageSize": 100},
        },
        60,
    )
    datasets = [x for x in (listed.get("datasets") or []) if isinstance(x, dict)]
    exact = [x for x in datasets if str(x.get("ref") or "").lower() == handle.lower()]
    if not exact:
        return row
    if len(exact) != 1:
        raise RuntimeError(f"{handle}: exact dataset count={len(exact)}")
    version = exact[0].get("currentVersionNumber", exact[0].get("current_version_number"))
    if type(version) is not int or version < 1:
        raise RuntimeError(f"{handle}: invalid current version")
    row.update(
        exists=True,
        dataset_version_number=version,
        last_updated=exact[0].get("lastUpdated"),
    )
    marker = broker(
        {
            "action": "dataset_json_files",
            "account_id": account_id,
            "dataset_ref": handle,
            "dataset_version_number": version,
            "file_names": ["CAMPAIGN_STATE.json"],
            "max_bytes_per_file": 262144,
        },
        120,
    )
    files = marker.get("files") or []
    if len(files) != 1 or not isinstance(files[0], dict) or not isinstance(files[0].get("json"), dict):
        raise RuntimeError(f"{handle}: CAMPAIGN_STATE missing")
    payload = files[0]["json"]
    artifacts = payload.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise RuntimeError(f"{handle}: artifact_sha256 missing")
    folds = []
    for name in artifacts:
        match = re.fullmatch(r"FOLD_([1-5])_RECOVERY\.(?:zip|cgpzip)", str(name))
        if match:
            folds.append(int(match.group(1)))
    folds = sorted(set(folds))
    has_final = "FINAL_EVIDENCE.cgpzip" in artifacts or "FINAL_EVIDENCE.zip" in artifacts
    status = payload.get("status")
    row.update(
        marker_status=status,
        completed_folds=folds,
        fold_count=len(folds),
        has_final_evidence=has_final,
        resolution_complete=(folds == [1, 2, 3, 4, 5] and has_final and status == "COMPLETE"),
        marker_updated_utc=payload.get("updated_utc"),
        marker_sha256=files[0].get("sha256"),
    )
    return row


def main() -> int:
    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            pool.submit(inspect, model_id, resolution): (model_id, resolution)
            for model_id in ASSIGNMENTS
            for resolution in RESOLUTIONS
        }
        for future in as_completed(futures):
            model_id, resolution = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "model_id": model_id,
                        "resolution": resolution,
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                )
    rows.sort(key=lambda r: (r["model_id"], r["resolution"]))

    models = []
    for model_id in ASSIGNMENTS:
        subset = [r for r in rows if r["model_id"] == model_id]
        by_resolution = {str(r["resolution"]): r for r in subset}
        fold_total = sum(int(r.get("fold_count") or 0) for r in subset)
        missing = []
        for resolution in RESOLUTIONS:
            row = by_resolution.get(str(resolution))
            present = set((row or {}).get("completed_folds") or [])
            for fold in range(1, 6):
                if fold not in present:
                    missing.append(f"R{resolution}/F{fold}")
        models.append(
            {
                "model_id": model_id,
                "folds_complete": fold_total,
                "folds_total": 15,
                "percent": round(fold_total / 15 * 100, 4),
                "all_resolutions_complete": all(
                    by_resolution.get(str(resolution), {}).get("resolution_complete") is True
                    for resolution in RESOLUTIONS
                ),
                "missing_slots": missing,
                "resolutions": {
                    str(resolution): {
                        "folds": by_resolution.get(str(resolution), {}).get("completed_folds", []),
                        "fold_count": by_resolution.get(str(resolution), {}).get("fold_count", 0),
                        "complete": by_resolution.get(str(resolution), {}).get("resolution_complete", False),
                        "exists": by_resolution.get(str(resolution), {}).get("exists", False),
                        "version": by_resolution.get(str(resolution), {}).get("dataset_version_number"),
                        "marker_status": by_resolution.get(str(resolution), {}).get("marker_status"),
                        "has_final_evidence": by_resolution.get(str(resolution), {}).get("has_final_evidence", False),
                        "marker_updated_utc": by_resolution.get(str(resolution), {}).get("marker_updated_utc"),
                    }
                    for resolution in RESOLUTIONS
                },
            }
        )

    folds_complete = sum(model["folds_complete"] for model in models)
    result = {
        "schema": "pneumonia.phase2.historical_matrix.audit.v2",
        "status": "PASS" if not errors else "PARTIAL_QUERY_ERROR",
        "models": models,
        "raw_states": rows,
        "errors": errors,
        "folds_complete": folds_complete,
        "folds_total": 165,
        "folds_remaining": 165 - folds_complete,
        "percent": round(folds_complete / 165 * 100, 4),
        "models_15_of_15": sum(1 for model in models if model["folds_complete"] == 15),
        "models_total": len(models),
    }
    output = Path(os.environ["RUNNER_TEMP"]) / "PHASE2_HISTORICAL_MATRIX_AUDIT_V2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("PHASE2_HISTORICAL_MATRIX=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    if errors:
        raise SystemExit("historical matrix contains query errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
