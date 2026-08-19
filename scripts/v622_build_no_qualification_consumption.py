#!/usr/bin/env python3
"""Build the V6.2.2 declaration that MODEL_QUALIFICATION was not consumed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT = "PNEUMONIA V6.2.2"
STATEMENT = (
    "MODEL_QUALIFICATION was not executed or used for model, ensemble, "
    "threshold, calibration, or policy selection."
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_record() -> dict[str, Any]:
    evidence = {
        "project": PROJECT,
        "statement": STATEMENT,
        "selection_data": "effective validation only",
    }
    record: dict[str, Any] = {
        "schema_version": 2,
        "record_kind": "DECLARATION_OF_NON_CONSUMPTION",
        "dataset_identity": "NO_MODEL_QUALIFICATION_COHORT_CONSUMED",
        "qualification_performed": False,
        "qualification_manifest_sha256": None,
        "evidence_sha256": sha256_json(evidence),
        "used_for_selection": False,
        "still_eligible_as_final_lockbox": False,
        "training_performed": False,
        "selection_data": "effective validation only",
        "locked_test_accessed": False,
        "external_data_accessed": False,
    }
    record["consumption_sha256"] = sha256_json(record)
    return record


def validate(record: dict[str, Any]) -> None:
    declared = str(record.get("consumption_sha256", ""))
    check = dict(record)
    check.pop("consumption_sha256", None)
    actual = sha256_json(check)
    if declared != actual:
        raise SystemExit("qualification non-consumption record hash mismatch")
    if record.get("used_for_selection") is not False:
        raise SystemExit("qualification non-consumption record must not be used for selection")
    if record.get("qualification_performed") is not False:
        raise SystemExit("qualification non-consumption record must declare no qualification execution")
    if record.get("training_performed") is not False:
        raise SystemExit("qualification non-consumption record must declare no training")
    if record.get("locked_test_accessed") is not False:
        raise SystemExit("qualification non-consumption record must declare no locked-test access")
    if record.get("external_data_accessed") is not False:
        raise SystemExit("qualification non-consumption record must declare no external-data access")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    record = build_record()
    validate(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "status": "PASS",
        "project": PROJECT,
        "record_kind": record["record_kind"],
        "used_for_selection": False,
        "qualification_performed": False,
        "consumption_sha256": record["consumption_sha256"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
