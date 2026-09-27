from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

CANONICAL_R384_STATE_HANDLE = "trickermark/m07-gate-r384-state-v1-7"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    state_root = Path(args.state_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()

    if not data_root.is_dir():
        raise SystemExit("AUTHORIZED_DATA_ROOT_NOT_FOUND")
    if not state_root.is_dir():
        raise SystemExit("FROZEN_STATE_ROOT_NOT_FOUND")
    if args.batch_size < 1:
        raise SystemExit("BATCH_SIZE_INVALID")

    work_root.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    preflight = here / "m07_vindr_pcxr_authorized_preflight.py"
    external = here / "m07_vindr_pcxr_external_v2.py"
    if not preflight.is_file() or not external.is_file():
        raise SystemExit("M07_VINDR_RUNTIME_SCRIPT_MISSING")

    preflight_receipt = work_root / "M07_VINDR_AUTHORIZED_PREFLIGHT.json"
    cmd = [
        sys.executable,
        str(preflight),
        "--data-root",
        str(data_root),
        "--output",
        str(preflight_receipt),
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit("M07_VINDR_AUTHORIZED_PREFLIGHT_FAILED")
    receipt = json.loads(preflight_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_AUTHORIZED_DATA_PREFLIGHT":
        raise SystemExit("M07_VINDR_AUTHORIZED_PREFLIGHT_NOT_PASS")

    os.environ["M07_EXTERNAL_RESOLUTION"] = "384"
    os.environ["M07_EXTERNAL_STATE_HANDLE"] = CANONICAL_R384_STATE_HANDLE
    os.environ["M07_EXTERNAL_BATCH_SIZE"] = str(args.batch_size)
    os.environ["M07_EXTERNAL_DATA_ROOT"] = str(data_root)
    os.environ["M07_EXTERNAL_STATE_ROOT"] = str(state_root)
    os.environ["M07_EXTERNAL_WORK_ROOT"] = str(work_root)

    handoff = {
        "schema": "m07.external.vindr_pcxr.authorized_local_handoff.v1",
        "status": "PREFLIGHT_PASS_READY_FOR_FROZEN_INFERENCE",
        "resolution": 384,
        "state_handle": CANONICAL_R384_STATE_HANDLE,
        "data_root": str(data_root),
        "state_root": str(state_root),
        "work_root": str(work_root),
        "batch_size": args.batch_size,
        "training_performed": False,
        "hpo_performed": False,
        "external_threshold_tuning": False,
        "external_adaptation": False,
        "preflight_receipt": str(preflight_receipt),
    }
    (work_root / "M07_VINDR_AUTHORIZED_LOCAL_HANDOFF.json").write_text(
        json.dumps(handoff, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(handoff, sort_keys=True), flush=True)

    runpy.run_path(str(external), run_name="__main__")

if __name__ == "__main__":
    main()
