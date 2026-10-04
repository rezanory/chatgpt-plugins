from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

BASE = "https://chatgpt-kaggle-gateway.rezanory-chatgpt-plugins.workers.dev/control/phase2/master-full-source"
PROJECT = "PNEUMONIA Phase-2"
PURPOSE = "master_full_kernel_source_recovery"

KERNELS = [
    ("M01",224,"master","azadka/pneumonia-v1-7-phase2-m01-r224-a18-35698388883"),
    ("M01",320,"master","azadka/pneumonia-v1-7-phase2-m01-r320-a05-35814596649"),
    ("M01",384,"master","azadka/pneumonia-v1-7-phase2-m01-r384-a05-36351862662"),
    ("M02",224,"kg-02","radlinaradlina/pneumonia-v1-7-phase2-m02-r224-a19-35698392981"),
    ("M02",320,"kg-02","radlinaradlina/pneumonia-v1-7-phase2-m02-r320-a05-35814599480"),
    ("M02",384,"kg-02","radlinaradlina/pneumonia-v1-7-phase2-m02-r384-a05-35927816711"),
    ("M03",224,"kg-03","rezanory/pneumonia-v1-7-phase2-m03-r224-a19-35698396139"),
    ("M03",320,"kg-03","rezanory/pneumonia-v1-7-phase2-m03-r320-a05-35814602879"),
    ("M03",384,"kg-03","rezanory/pneumonia-v1-7-phase2-m03-r384-a05-36362420209"),
    ("M04",224,"kg-04","reyhanehazad/pneumonia-v1-7-phase2-m04-r224-a19-35698399363"),
    ("M04",320,"kg-04","reyhanehazad/pneumonia-v1-7-phase2-m04-r320-a05-35814939405"),
    ("M04",384,"kg-04","reyhanehazad/pneumonia-v1-7-phase2-m04-r384-a05-36337420014"),
    ("M05",224,"kg-05","trickermark/pneumonia-v1-7-phase2-m05-r224-a12-36269524803"),
    ("M05",320,"kg-05","trickermark/pneumonia-v1-7-phase2-m05-r320-a04-36370588557"),
    ("M05",384,"kg-05","trickermark/pneumonia-v1-7-phase2-m05-r384-a05-36485988648"),
    ("M06",224,"kg-06","msdenis/pneumonia-v1-7-phase2-m06-r224-a18-35698405831"),
    ("M06",320,"kg-06","msdenis/pneumonia-v1-7-phase2-m06-r320-a05-35814605997"),
    ("M06",384,"kg-06","msdenis/pneumonia-v1-7-phase2-m06-r384-a05-36371466038"),
    ("M07",224,"kg-03","rezanory/m07-final-5fold-fix2-20260914"),
    ("M07",320,"kg-05","trickermark/m07-runtime-r320-a13-20260917-35265801216"),
    ("M07",384,"kg-05","trickermark/m07-runtime-r384-a12-20260920-35505225105"),
    ("M08",224,"kg-07","nisabulutmark/pneumonia-v1-7-phase2-m08-r224-a18-35698409656"),
    ("M08",320,"kg-07","nisabulutmark/pneumonia-v1-7-phase2-m08-r320-a05-35814941709"),
    ("M08",384,"kg-07","nisabulutmark/pneumonia-v1-7-phase2-m08-r384-a05-36338577091"),
    ("M09",224,"kg-08","azadkk/pneumonia-v1-7-phase2-m09-r224-a18-35698412479"),
    ("M09",320,"kg-08","azadkk/pneumonia-v1-7-phase2-m09-r320-a05-35904860843"),
    ("M09",384,"kg-08","azadkk/pneumonia-v1-7-phase2-m09-r384-a03-36398484836"),
    ("M10",224,"kg-09","mylovevpn1/pneumonia-v1-7-phase2-m10-r224-a18-35698415361"),
    ("M10",320,"kg-09","mylovevpn1/pneumonia-v1-7-phase2-m10-r320-a05-35814608153"),
    ("M10",384,"kg-09","mylovevpn1/pneumonia-v1-7-phase2-m10-r384-a05-36354744478"),
    ("M11",224,"kg-10","computstu1/pneumonia-v1-7-phase2-m11-r224-a11-36347200480"),
    ("M11",320,"kg-10","computstu1/pneumonia-v1-7-phase2-m11-r320-a05-36394459369"),
    ("M11",384,"kg-10","computstu1/pneumonia-v1-7-phase2-m11-r384-a05-36597162497"),
    ("M12",224,"kg-11","jobreza1/pneumonia-v1-7-phase2-m12-r224-a18-35698425105"),
    ("M12",320,"kg-11","jobreza1/pneumonia-v1-7-phase2-m12-r320-a05-35826043198"),
    ("M12",384,"kg-11","jobreza1/pneumonia-v1-7-phase2-m12-r384-a03-36369326430"),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(token: str, kernel_ref: str) -> dict:
    payload = json.dumps({"kernel_ref": kernel_ref}).encode("utf-8")
    last = None
    for attempt in range(1, 8):
        req = urllib.request.Request(
            BASE,
            data=payload,
            method="POST",
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "phase2-master-full-source-collector/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                value = json.loads(response.read().decode("utf-8"))
            if value.get("ok") is False:
                raise RuntimeError(str(value.get("error")))
            return value
        except Exception as exc:
            last = exc
            print(f"FULL_SOURCE_RETRY {attempt}/7 {kernel_ref}: {exc}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"FULL_SOURCE_FETCH_FAILED:{kernel_ref}:{last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    token = os.environ["PHASE2_MASTER_FULL_SOURCE_TOKEN"].strip()
    if len(token) < 32:
        raise SystemExit("FULL_SOURCE_TOKEN_INVALID")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_cells = 0
    for index, (model, resolution, account_id, kernel_ref) in enumerate(KERNELS, start=1):
        value = fetch(token, kernel_ref)
        if value.get("project") != PROJECT or value.get("purpose") != PURPOSE:
            raise RuntimeError("FULL_SOURCE_BRIDGE_IDENTITY_DRIFT")
        if value.get("account_id") != account_id or value.get("kernel_ref") != kernel_ref:
            raise RuntimeError(f"FULL_SOURCE_KERNEL_IDENTITY_DRIFT:{kernel_ref}")
        blob = value.get("blob") or {}
        source = blob.get("source")
        if not isinstance(source, str) or not source:
            raise RuntimeError(f"FULL_SOURCE_MISSING:{kernel_ref}")
        if "<truncated>" in source or "\\<truncated>" in source:
            raise RuntimeError(f"FULL_SOURCE_TRUNCATED:{kernel_ref}")
        raw = source.encode("utf-8")
        actual_sha = sha(raw)
        if actual_sha != value.get("source_sha256"):
            raise RuntimeError(f"FULL_SOURCE_SHA_MISMATCH:{kernel_ref}")

        fmt = "script"
        cell_count = 1
        suffix = ".py"
        try:
            notebook = json.loads(source)
            if isinstance(notebook, dict) and isinstance(notebook.get("cells"), list):
                fmt = "notebook"
                cell_count = len(notebook["cells"])
                suffix = ".ipynb"
        except json.JSONDecodeError:
            pass

        name = f"{index:02d}_{model}_R{resolution}{suffix}"
        (args.output_dir / name).write_bytes(raw)
        total_cells += cell_count
        row = {
            "index": index,
            "model": model,
            "resolution": resolution,
            "account_id": account_id,
            "kernel_ref": kernel_ref,
            "source_file": name,
            "source_sha256": actual_sha,
            "source_bytes": len(raw),
            "format": fmt,
            "cell_count": cell_count,
            "metadata": value.get("metadata") or {},
        }
        rows.append(row)
        print(
            "FULL_SOURCE_PASS "
            + json.dumps(
                {
                    "index": index,
                    "model": model,
                    "resolution": resolution,
                    "bytes": len(raw),
                    "cells": cell_count,
                    "sha256": actual_sha,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(rows) != 36:
        raise RuntimeError(f"FULL_SOURCE_COUNT_INVALID:{len(rows)}")
    if total_cells < 100:
        raise RuntimeError(f"FULL_SOURCE_CELL_COUNT_TOO_SMALL:{total_cells}")

    manifest = {
        "schema": "pneumonia.phase2.master.full-kaggle-source.manifest.v1",
        "status": "PASS",
        "kernel_count": 36,
        "total_original_cells": total_cells,
        "kernels": rows,
    }
    (args.output_dir / "PHASE2_MASTER_FULL_SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "PHASE2_MASTER_FULL_SOURCE_RECOVERY_PASS "
        + json.dumps({"kernels": 36, "total_original_cells": total_cells}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
