from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import pathlib
import time
import urllib.request
import zlib

import phase2_final_evidence_extract_v1 as broker

OWNER = "azadka"
FILE_NAME = "PNEUMONIA_PHASE2_MASTER_ACTUAL_KAGGLE_SOURCES_V2.ipynb"
EXPECTED_SHA256 = "72cec18366b769a3753d70be879d5d458c39cc09bdbf106204f53ccf51d79daf"
EXPECTED_BYTES = 4455079
EXPECTED_CELLS = 162
EXPECTED_CODE_CELLS = 98
PART_COUNT = 5
PART_REFS = [f"{OWNER}/pneumonia-phase2-master-part-{i:02d}-v1" for i in range(PART_COUNT)]
FINAL_REF = f"{OWNER}/pneumonia-phase2-master-actual-v3"
FINAL_RECEIPT = "PNEUMONIA_PHASE2_MASTER_ACTUAL_KAGGLE_SOURCES_V2_REPLAY_RECEIPT.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recursive_status(value) -> str:
    statuses = []
    stack = [value]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        for key in ("status", "state", "kernel_status", "statusName", "status_name"):
            item = cur.get(key)
            if isinstance(item, str) and item.strip():
                statuses.append(item.strip().upper())
        stack.extend(v for v in cur.values() if isinstance(v, dict))
    for terminal in ("COMPLETE", "ERROR", "CANCELLED"):
        if terminal in statuses:
            return terminal
    for running in ("RUNNING", "QUEUED", "PENDING"):
        if running in statuses:
            return running
    return statuses[-1] if statuses else "UNKNOWN"


def action_save(token: str, request_id: str, ref: str, title: str, source: str, *, kernel_sources=None) -> dict:
    payload = {
        "request_id": request_id,
        "provider": "kaggle",
        "operation_class": "compute",
        "account_id": "master",
        "purpose": (
            "Publish one lossless Phase-2 master payload part or the final assembler on the master Kaggle account. "
            "No model training, tuning, or inference."
        ),
        "service": "kernels.KernelsApiService",
        "method": "SaveKernel",
        "body": {
            "slug": ref,
            "newTitle": title,
            "text": source,
            "language": "PYTHON",
            "kernelType": "SCRIPT",
            "kernelExecutionType": "SAVE_AND_RUN_ALL",
            "isPrivate": True,
            "enableGpu": False,
            "enableTpu": False,
            "enableInternet": False,
            "datasetDataSources": [],
            "kernelDataSources": list(kernel_sources or []),
            "competitionDataSources": [],
            "modelDataSources": [],
        },
    }
    response = broker.post_json(broker.ACTION_ENDPOINT, token, payload, timeout=300)
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    if not response.get("ok") or response.get("error") or result.get("error"):
        raise RuntimeError("SAVE_KERNEL_REJECTED:" + ref + ":" + json.dumps(response)[:10000])
    return response


def read_status(token: str, ref: str) -> dict:
    return broker.post_json(
        broker.READ_ENDPOINT,
        token,
        {"action": "resolved_kernel_status", "account_id": "master", "kernel_ref": ref},
        timeout=180,
    )


def wait_complete(token: str, ref: str, *, max_attempts=180, sleep_seconds=5) -> None:
    last = {}
    for attempt in range(1, max_attempts + 1):
        row = read_status(token, ref)
        last = row
        status = recursive_status(row)
        print("MASTER_PART_STATUS", ref, attempt, status, flush=True)
        if status == "COMPLETE":
            return
        if status in {"ERROR", "CANCELLED"}:
            raise RuntimeError("KERNEL_TERMINAL_" + status + ":" + ref)
        time.sleep(sleep_seconds)
    raise RuntimeError("KERNEL_STATUS_TIMEOUT:" + ref + ":" + json.dumps(last)[:5000])


def list_output(token: str, ref: str) -> list[dict]:
    owner, slug = ref.split("/", 1)
    response = broker.post_json(
        broker.READ_ENDPOINT,
        token,
        {
            "action": "raw_read",
            "account_id": "master",
            "service": "kernels.KernelsApiService",
            "method": "ListKernelSessionOutput",
            "body": {"userName": owner, "kernelSlug": slug, "page": 1, "pageSize": 100},
        },
        timeout=180,
    )
    if not response.get("ok"):
        return []
    payload = broker._read_payload(response)
    files = payload.get("files") or []
    return [x for x in files if isinstance(x, dict)]


def wait_output_file(token: str, ref: str, file_name: str, *, attempts=60) -> dict:
    for attempt in range(1, attempts + 1):
        for item in list_output(token, ref):
            name = str(item.get("fileName") or item.get("name") or "")
            if name.replace("\\", "/").endswith(file_name):
                print("MASTER_OUTPUT_FILE_READY", ref, file_name, attempt, flush=True)
                return item
        time.sleep(3)
    raise RuntimeError("OUTPUT_FILE_NOT_READY:" + ref + ":" + file_name)


def part_source(index: int, raw_part: bytes) -> str:
    b64 = base64.b64encode(raw_part).decode("ascii")
    expected = sha256_bytes(raw_part)
    return (
        "import base64,hashlib,json,pathlib\n"
        f"INDEX={index}\n"
        f"EXPECTED={expected!r}\n"
        f"DATA={b64!r}\n"
        "raw=base64.b64decode(DATA)\n"
        "assert hashlib.sha256(raw).hexdigest()==EXPECTED\n"
        "out=pathlib.Path('/kaggle/working')/f'phase2_master_part_{INDEX:02d}.bin'\n"
        "out.write_bytes(raw)\n"
        "receipt={'status':'PASS','part':INDEX,'bytes':len(raw),'sha256':EXPECTED}\n"
        "(pathlib.Path('/kaggle/working')/f'phase2_master_part_{INDEX:02d}.json').write_text(json.dumps(receipt,sort_keys=True))\n"
        "print('PHASE2_MASTER_PART_PASS '+json.dumps(receipt,sort_keys=True))\n"
    )


def assembler_source(part_refs: list[str], compressed_sha: str, compressed_bytes: int) -> str:
    return f"""import hashlib,json,pathlib,zlib
EXPECTED_SHA={EXPECTED_SHA256!r}
EXPECTED_BYTES={EXPECTED_BYTES}
EXPECTED_CELLS={EXPECTED_CELLS}
EXPECTED_CODE_CELLS={EXPECTED_CODE_CELLS}
EXPECTED_COMPRESSED_SHA={compressed_sha!r}
EXPECTED_COMPRESSED_BYTES={compressed_bytes}
PART_REFS={part_refs!r}
root=pathlib.Path('/kaggle/input')
parts=sorted(root.rglob('phase2_master_part_*.bin'),key=lambda p:p.name)
if len(parts)!={PART_COUNT}:
    raise RuntimeError(f'PART_COUNT_INVALID:{{len(parts)}}:{{[str(p) for p in parts]}}')
compressed=b''.join(p.read_bytes() for p in parts)
if len(compressed)!=EXPECTED_COMPRESSED_BYTES:
    raise RuntimeError(f'COMPRESSED_BYTES_INVALID:{{len(compressed)}}')
if hashlib.sha256(compressed).hexdigest()!=EXPECTED_COMPRESSED_SHA:
    raise RuntimeError('COMPRESSED_SHA_INVALID')
raw=zlib.decompress(compressed)
if len(raw)!=EXPECTED_BYTES:
    raise RuntimeError(f'RAW_BYTES_INVALID:{{len(raw)}}')
if hashlib.sha256(raw).hexdigest()!=EXPECTED_SHA:
    raise RuntimeError('RAW_SHA_INVALID')
nb=json.loads(raw)
cells=nb.get('cells') or []
code_cells=sum(c.get('cell_type')=='code' for c in cells)
if len(cells)!=EXPECTED_CELLS or code_cells!=EXPECTED_CODE_CELLS:
    raise RuntimeError(f'CELL_COUNT_INVALID:{{len(cells)}}:{{code_cells}}')
out=pathlib.Path('/kaggle/working')/{FILE_NAME!r}
out.write_bytes(raw)
receipt={{
    'schema':'pneumonia.phase2.master.kaggle-parts-reassembly.v1',
    'status':'PASS',
    'mode':'REASSEMBLED_FROM_MASTER_PART_KERNELS',
    'file_name':out.name,
    'bytes':len(raw),
    'sha256':EXPECTED_SHA,
    'cells':len(cells),
    'code_cells':code_cells,
    'part_count':len(parts),
    'part_refs':PART_REFS,
    'training_performed':False,
    'inference_performed':False,
}}
(pathlib.Path('/kaggle/working')/{FINAL_RECEIPT!r}).write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\\n')
print('PHASE2_MASTER_REASSEMBLY_PASS '+json.dumps(receipt,sort_keys=True))
"""


def fetch_receipt(token: str, ref: str) -> dict:
    response = broker.post_json(
        broker.READ_ENDPOINT,
        token,
        {
            "action": "output_json_files",
            "account_id": "master",
            "kernel_ref": ref,
            "file_names": [FINAL_RECEIPT],
            "max_bytes_per_file": 262144,
        },
        timeout=180,
    )
    if not response.get("ok"):
        raise RuntimeError("FINAL_RECEIPT_QUERY_FAILED=" + json.dumps(response)[:6000])
    payload = broker._read_payload(response)
    files = payload.get("files") or []
    if len(files) != 1 or not isinstance(files[0].get("json"), dict):
        raise RuntimeError("FINAL_RECEIPT_MISSING")
    return files[0]["json"]


def download_output(item: dict) -> bytes:
    url = str(item.get("url") or "")
    if not url.startswith("https://"):
        raise RuntimeError("FINAL_OUTPUT_URL_MISSING")
    request = urllib.request.Request(url, headers={"User-Agent": "phase2-master-kaggle-parts-verify/1.0"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    action_token = os.environ["CGP_ACTION_OIDC_TOKEN"].strip()
    read_token = os.environ["CGP_READ_OIDC_TOKEN"].strip()
    if len(action_token) < 100 or len(read_token) < 100:
        raise SystemExit("OIDC_TOKEN_INVALID")

    raw = args.notebook.read_bytes()
    if len(raw) != EXPECTED_BYTES or sha256_bytes(raw) != EXPECTED_SHA256:
        raise RuntimeError("INPUT_NOTEBOOK_IDENTITY_INVALID")
    parsed = json.loads(raw)
    if len(parsed.get("cells") or []) != EXPECTED_CELLS:
        raise RuntimeError("INPUT_CELL_COUNT_INVALID")
    if sum(c.get("cell_type") == "code" for c in parsed["cells"]) != EXPECTED_CODE_CELLS:
        raise RuntimeError("INPUT_CODE_CELL_COUNT_INVALID")

    compressed = zlib.compress(raw, 9)
    compressed_sha = sha256_bytes(compressed)
    step = math.ceil(len(compressed) / PART_COUNT)
    parts = [compressed[i * step : (i + 1) * step] for i in range(PART_COUNT)]
    if any(not part for part in parts):
        raise RuntimeError("EMPTY_PART")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_meta = []
    for index, (ref, part) in enumerate(zip(PART_REFS, parts)):
        source = part_source(index, part)
        source_bytes = len(source.encode("utf-8"))
        if source_bytes >= 900_000:
            raise RuntimeError(f"PART_SOURCE_TOO_LARGE:{index}:{source_bytes}")
        title = f"Pneumonia Phase2 Master Part {index:02d} V1"
        launch = action_save(
            action_token,
            f"phase2-master-part-{index:02d}-{os.environ['GITHUB_RUN_ID']}",
            ref,
            title,
            source,
        )
        wait_complete(read_token, ref)
        part_meta.append(
            {
                "index": index,
                "ref": ref,
                "compressed_bytes": len(part),
                "compressed_sha256": sha256_bytes(part),
                "source_bytes": source_bytes,
                "launch": launch,
            }
        )

    print("MASTER_PARTS_ALL_COMPLETE", json.dumps(PART_REFS), flush=True)
    time.sleep(20)
    final_source = assembler_source(PART_REFS, compressed_sha, len(compressed))
    if len(final_source.encode("utf-8")) >= 900_000:
        raise RuntimeError("ASSEMBLER_SOURCE_TOO_LARGE")
    final_launch = action_save(
        action_token,
        f"phase2-master-assembler-{os.environ['GITHUB_RUN_ID']}",
        FINAL_REF,
        "Pneumonia Phase2 Master Actual V3",
        final_source,
        kernel_sources=PART_REFS,
    )
    wait_complete(read_token, FINAL_REF)
    output_item = wait_output_file(read_token, FINAL_REF, FILE_NAME)
    wait_output_file(read_token, FINAL_REF, FINAL_RECEIPT)
    receipt = fetch_receipt(read_token, FINAL_REF)

    if receipt.get("status") != "PASS" or receipt.get("sha256") != EXPECTED_SHA256:
        raise RuntimeError("FINAL_REASSEMBLY_RECEIPT_INVALID=" + json.dumps(receipt))
    if int(receipt.get("cells") or 0) != EXPECTED_CELLS or int(receipt.get("code_cells") or 0) != EXPECTED_CODE_CELLS:
        raise RuntimeError("FINAL_REASSEMBLY_CELL_COUNTS_INVALID")

    downloaded = download_output(output_item)
    downloaded_sha = sha256_bytes(downloaded)
    if len(downloaded) != EXPECTED_BYTES or downloaded_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"FINAL_KAGGLE_DOWNLOAD_IDENTITY_INVALID:{len(downloaded)}:{downloaded_sha}"
        )

    readback = args.output_dir / FILE_NAME
    readback.write_bytes(downloaded)
    publish_receipt = {
        "schema": "pneumonia.phase2.master.kaggle-parts-publish.v1",
        "status": "PASS",
        "kernel_ref": FINAL_REF,
        "file_name": FILE_NAME,
        "bytes": len(downloaded),
        "sha256": downloaded_sha,
        "cells": EXPECTED_CELLS,
        "code_cells": EXPECTED_CODE_CELLS,
        "part_count": PART_COUNT,
        "part_refs": PART_REFS,
        "compressed_bytes": len(compressed),
        "compressed_sha256": compressed_sha,
        "final_receipt": receipt,
        "training_performed": False,
        "inference_performed": False,
    }
    (args.output_dir / "KAGGLE_MASTER_PARTS_PUBLISH_RECEIPT.json").write_text(
        json.dumps(publish_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "KAGGLE_MASTER_PARTS_META.json").write_text(
        json.dumps({"parts": part_meta, "final_launch": final_launch}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PHASE2_MASTER_KAGGLE_PARTS_PUBLISH_PASS " + json.dumps(publish_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
