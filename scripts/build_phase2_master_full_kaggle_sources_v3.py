from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import zlib
from collections import Counter, defaultdict

CLOSURE_SHA = "83481411e937ba1da52c5e95841e68da22f1522ef70b622d624d6a14d65f87f5"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cell_text(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def source_lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def md(value: str, metadata: dict | None = None) -> dict:
    return {"cell_type": "markdown", "metadata": metadata or {}, "source": source_lines(value)}


def code(value: str, metadata: dict | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata or {},
        "outputs": [],
        "source": source_lines(value),
    }


def cell_sha(cell: dict) -> str:
    return sha((str(cell.get("cell_type")) + "\0" + cell_text(cell)).encode("utf-8"))


def one(root: pathlib.Path, name: str) -> pathlib.Path:
    rows = [p for p in root.rglob(name) if p.is_file()]
    if len(rows) != 1:
        raise RuntimeError(f"FILE_COUNT_INVALID:{name}:{len(rows)}")
    return rows[0]


def normalize_cells(raw: bytes, row: dict) -> list[dict]:
    if sha(raw) != row["source_sha256"]:
        raise RuntimeError(f"SOURCE_SHA_MISMATCH:{row['kernel_ref']}")
    text = raw.decode("utf-8")
    if "<truncated>" in text or "\\<truncated>" in text:
        raise RuntimeError(f"SOURCE_TRUNCATED:{row['kernel_ref']}")

    if row["format"] == "notebook":
        notebook = json.loads(text)
        original = notebook.get("cells")
        if not isinstance(original, list):
            raise RuntimeError(f"NOTEBOOK_CELLS_MISSING:{row['kernel_ref']}")
        if len(original) != int(row["cell_count"]):
            raise RuntimeError(f"NOTEBOOK_CELL_COUNT_DRIFT:{row['kernel_ref']}")
        cells = []
        for index, source_cell in enumerate(original):
            if not isinstance(source_cell, dict):
                continue
            ctype = source_cell.get("cell_type")
            if ctype not in {"code", "markdown", "raw"}:
                continue
            cell = {
                "cell_type": ctype,
                "metadata": dict(source_cell.get("metadata") or {}),
                "source": source_lines(cell_text(source_cell)),
            }
            cell["metadata"].update(
                {
                    "phase2_original_kernel_ref": row["kernel_ref"],
                    "phase2_original_cell_index": index,
                    "phase2_model": row["model"],
                    "phase2_resolution": row["resolution"],
                }
            )
            if ctype == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            cells.append(cell)
        return cells

    return [
        code(
            text,
            {
                "phase2_original_kernel_ref": row["kernel_ref"],
                "phase2_original_cell_index": 0,
                "phase2_model": row["model"],
                "phase2_resolution": row["resolution"],
            },
        )
    ]


def metric_block(unit: dict, wanted: str) -> dict:
    for source in unit.get("json_sources") or []:
        for block in source.get("metric_blocks") or []:
            if block.get("path") == wanted:
                return block.get("values") or {}
    return {}


def pct(value) -> str:
    try:
        return f"{100.0 * float(value):.4f}%"
    except Exception:
        return "—"


def result_table(model: str, units: list[dict]) -> str:
    rows = [
        f"### {model} — نتایج واقعی نهایی",
        "",
        "| R | OOF Accuracy | OOF Precision | OOF Recall | AUROC | Locked Accuracy | Locked Precision | Locked Recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for unit in sorted(units, key=lambda x: int(x["resolution"])):
        oof = metric_block(unit, "oof")
        locked = (((unit.get("locked_test_recovery") or {}).get("primary") or {}).get("metrics") or {})
        rows.append(
            "| {r} | {oa} | {op} | {orr} | {auc} | {la} | {lp} | {lr} |".format(
                r=unit["resolution"],
                oa=pct(oof.get("accuracy")),
                op=pct(oof.get("macro_precision")),
                orr=pct(oof.get("macro_recall", oof.get("balanced_accuracy"))),
                auc=pct(oof.get("auroc")),
                la=pct(locked.get("accuracy")),
                lp=pct(locked.get("macro_precision")),
                lr=pct(locked.get("macro_recall", locked.get("balanced_accuracy"))),
            )
        )
    return "\n".join(rows) + "\n"


def fold_table(model: str, units: list[dict]) -> str:
    rows = [
        f"#### {model} — نتایج Foldها",
        "",
        "| R | Fold | Train Acc | Train Precision | Train Recall | Val Acc | Val Precision | Val Recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for unit in sorted(units, key=lambda x: int(x["resolution"])):
        for fold in sorted(unit.get("recovered_folds") or [], key=lambda x: int(x.get("fold_id") or 0)):
            train = fold.get("train_metrics") if isinstance(fold.get("train_metrics"), dict) else {}
            val = fold.get("validation_metrics") if isinstance(fold.get("validation_metrics"), dict) else {}
            rows.append(
                "| {r} | {f} | {ta} | {tp} | {tr} | {va} | {vp} | {vr} |".format(
                    r=unit["resolution"],
                    f=fold.get("fold_id"),
                    ta=pct(train.get("accuracy")),
                    tp=pct(train.get("macro_precision")),
                    tr=pct(train.get("balanced_accuracy", train.get("macro_recall"))),
                    va=pct(val.get("accuracy")),
                    vp=pct(val.get("macro_precision")),
                    vr=pct(val.get("balanced_accuracy", val.get("macro_recall"))),
                )
            )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--full-results", type=pathlib.Path, required=True)
    parser.add_argument("--closure", type=pathlib.Path, required=True)
    parser.add_argument("--dual", type=pathlib.Path, required=True)
    parser.add_argument("--paired", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()

    manifest_path = args.source_root / "PHASE2_MASTER_FULL_SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("kernel_count") != 36:
        raise RuntimeError("SOURCE_MANIFEST_INVALID")
    if int(manifest.get("total_original_cells") or 0) < 100:
        raise RuntimeError("SOURCE_MANIFEST_CELL_COUNT_TOO_SMALL")

    full_path = one(args.full_results, "PHASE2_FULL_RESULTS_MATRIX_V1.json")
    closure_path = one(args.closure, "PHASE2_FINAL_SCIENTIFIC_CLOSURE_V1.json")
    full = json.loads(full_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if full.get("status") != "PASS" or int(full.get("folds_represented") or 0) != 180:
        raise RuntimeError("FULL_RESULTS_INVALID")
    if closure.get("status") != "CLOSED_PASS" or closure.get("closure_sha256") != CLOSURE_SHA:
        raise RuntimeError("CLOSURE_INVALID")

    units = {(str(row["model_id"]), int(row["resolution"])): row for row in full.get("units") or []}
    expected = {(f"M{i:02d}", resolution) for i in range(1, 13) for resolution in (224, 320, 384)}
    if set(units) != expected:
        raise RuntimeError("FULL_RESULTS_COVERAGE_INVALID")

    store = {}
    occurrences = defaultdict(list)
    sequences = {}
    kernel_rows = []

    for row in manifest["kernels"]:
        raw = (args.source_root / row["source_file"]).read_bytes()
        cells = normalize_cells(raw, row)
        sequence = []
        for index, cell in enumerate(cells):
            digest = cell_sha(cell)
            sequence.append(digest)
            store.setdefault(digest, cell)
            occurrences[digest].append(
                {
                    "kernel_ref": row["kernel_ref"],
                    "model": row["model"],
                    "resolution": row["resolution"],
                    "index": index,
                }
            )
        sequences[row["kernel_ref"]] = sequence
        kernel_rows.append(dict(row))

    original_occurrences = sum(len(value) for value in sequences.values())
    if original_occurrences != int(manifest["total_original_cells"]):
        raise RuntimeError(
            f"ORIGINAL_CELL_OCCURRENCE_DRIFT:{original_occurrences}:{manifest['total_original_cells']}"
        )

    counts = Counter({digest: len(rows) for digest, rows in occurrences.items()})
    shared = {digest for digest, count in counts.items() if count >= 2}

    final_manifest = {
        "schema": "pneumonia.phase2.master.full-source-cell-manifest.v1",
        "kernel_count": 36,
        "original_cell_occurrences": original_occurrences,
        "unique_cell_count": len(store),
        "shared_unique_cell_count": len(shared),
        "kernels": kernel_rows,
        "cell_sequences": sequences,
    }

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "phase2_full_source_manifest": final_manifest,
        },
        "cells": [],
    }
    notebook["cells"].append(
        md(
            "# PNEUMONIA Phase-2 Master — کد کامل واقعی ۳۶ Kernel از ۱۱ حساب Kaggle\n\n"
            "این فایل از full source خود Kaggle ساخته شده است؛ نه template، نه نسخه CORRECTED و نه پاسخ truncate شده broker.\n\n"
            "فقط سلول‌هایی که byte-for-byte یکسان بوده‌اند یک بار در Shared Exact Cells آمده‌اند. ترتیب و hash تمام سلول‌های اصلی هر Kernel در metadata محفوظ است.\n"
        )
    )

    notebook["cells"].append(md("## Shared Exact Cells\n"))
    for digest in sorted(shared):
        cell = dict(store[digest])
        cell["metadata"] = dict(cell.get("metadata") or {})
        cell["metadata"].update(
            {
                "phase2_cell_sha256": digest,
                "phase2_occurrence_count": counts[digest],
                "phase2_source_refs": sorted({x["kernel_ref"] for x in occurrences[digest]}),
            }
        )
        notebook["cells"].append(cell)

    for model in [f"M{i:02d}" for i in range(1, 13)]:
        notebook["cells"].append(md(f"# {model}\n"))
        model_rows = [row for row in kernel_rows if row["model"] == model]
        for row in sorted(model_rows, key=lambda x: int(x["resolution"])):
            ref = row["kernel_ref"]
            notebook["cells"].append(
                md(
                    f"## {model} — R{row['resolution']}\n\n"
                    f"Account: {row['account_id']}  \n"
                    f"Kernel: {ref}  \n"
                    f"Full source SHA256: {row['source_sha256']}  \n"
                    f"Original cells: {row['cell_count']}\n"
                )
            )
            for digest in sequences[ref]:
                if digest in shared:
                    continue
                cell = dict(store[digest])
                cell["metadata"] = dict(cell.get("metadata") or {})
                cell["metadata"].update(
                    {
                        "phase2_cell_sha256": digest,
                        "phase2_source_ref": ref,
                        "phase2_model": model,
                        "phase2_resolution": row["resolution"],
                    }
                )
                notebook["cells"].append(cell)
        model_units = [units[(model, resolution)] for resolution in (224, 320, 384)]
        notebook["cells"].append(md(result_table(model, model_units)))
        notebook["cells"].append(md(fold_table(model, model_units)))

    full_raw = full_path.read_bytes()
    closure_raw = closure_path.read_bytes()
    full_b64 = base64.b64encode(zlib.compress(full_raw, 9)).decode("ascii")
    closure_b64 = base64.b64encode(zlib.compress(closure_raw, 9)).decode("ascii")
    notebook["cells"].append(md("# Final Results / Scientific Closure\n"))
    notebook["cells"].append(
        code(
            "import base64, json, zlib\n"
            + f"PHASE2_FULL_RESULTS_B64={full_b64!r}\n"
            + f"PHASE2_FINAL_CLOSURE_B64={closure_b64!r}\n"
            + "PHASE2_FULL_RESULTS=json.loads(zlib.decompress(base64.b64decode(PHASE2_FULL_RESULTS_B64)).decode('utf-8'))\n"
            + "PHASE2_FINAL_CLOSURE=json.loads(zlib.decompress(base64.b64decode(PHASE2_FINAL_CLOSURE_B64)).decode('utf-8'))\n"
            + f"assert PHASE2_FINAL_CLOSURE['closure_sha256']=={CLOSURE_SHA!r}\n"
            + "print(PHASE2_FINAL_CLOSURE['status'], PHASE2_FINAL_CLOSURE['remaining_gate'], PHASE2_FINAL_CLOSURE['training_folds'])\n",
            {"tags": ["final-evidence"]},
        )
    )

    extras = {}
    for root, prefix in ((args.dual, "dual"), (args.paired, "paired")):
        for file in sorted(root.rglob("*")):
            if file.is_file() and file.suffix.lower() in {".json", ".csv", ".md"}:
                extras[f"{prefix}/{file.name}"] = base64.b64encode(
                    zlib.compress(file.read_bytes(), 9)
                ).decode("ascii")
    notebook["cells"].append(
        code(
            "PHASE2_EXTRA_STATS="
            + repr(json.dumps(extras, sort_keys=True))
            + "\nprint(sorted(json.loads(PHASE2_EXTRA_STATS)))\n",
            {"tags": ["paired-bootstrap-holm"]},
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    receipt = {
        "schema": "pneumonia.phase2.master.full-kaggle-source.build.v3",
        "status": "PASS",
        "output_sha256": sha(args.output.read_bytes()),
        "bytes": args.output.stat().st_size,
        "cells": len(notebook["cells"]),
        "code_cells": sum(c.get("cell_type") == "code" for c in notebook["cells"]),
        "real_kernel_count": 36,
        "original_cell_occurrences": original_occurrences,
        "unique_source_cells": len(store),
        "shared_unique_source_cells": len(shared),
        "closure_sha256": CLOSURE_SHA,
        "source_manifest_sha256": sha(manifest_path.read_bytes()),
    }
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE2_MASTER_FULL_SOURCE_BUILD_PASS " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
