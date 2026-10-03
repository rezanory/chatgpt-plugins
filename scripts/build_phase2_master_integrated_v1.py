#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import lzma
import pathlib
import re
import zipfile

BASE_SHA = "219d97dd03c6d8de87b4e40f9d25be618c4d7d3bbda475b2ee3c2e68ad6a3a8d"
CLOSURE_SHA = "83481411e937ba1da52c5e95841e68da22f1522ef70b622d624d6a14d65f87f5"

FIXES = (
    ("a6825ac", "storage-safe finalization and restore headroom"),
    ("8b1d590", "transient Kaggle state reads"),
    ("8d2209a", "XAI helper topology"),
    ("c7c749f", "decision curve helper"),
    ("a146ed9", "risk coverage helper"),
    ("e90e329", "reliability correctness"),
    ("06537e", "reliability plot helper"),
    ("8a464c", "locked-test logit helper"),
    ("5ae71d", "validation history figures"),
    ("0852b", "confusion helper"),
    ("4f551b", "state snapshot classification"),
    ("cbb3d7", "state lineage"),
    ("75bb68", "token-only owner reconciliation"),
    ("9a05b8", "state restore file semantics"),
    ("d387fe", "A03 runtime recovery"),
    ("0b060f", "frozen M07 recipe binding"),
    ("d43e2a", "legacy M07 gate isolation"),
    ("f399fa", "Phase-2 validation hardening"),
    ("dcc53ac", "distributed comparator campaign"),
    ("ef9d2bd", "external archive materialization"),
    ("f66a594", "cgpzip external restore"),
    ("8a75191", "GPU capacity CPU fallback"),
    ("d7727fc", "corrected final closure"),
    ("331aa97", "verified-output finalist handoff"),
    ("199bd74", "final closure artifact path fix"),
)

SUPPORT_SOURCES = (
    "scripts/pneumonia_phase2_unit.py",
    "scripts/phase2_dual_pr_oof_stats_v1.py",
    "scripts/phase2_paired_oof_stats_v1.py",
    "scripts/phase2_finalist_rsna_external_dispatch_v1.py",
    "scripts/m07_rsna_pediatric_external_v1.py",
    "scripts/phase2_final_scientific_closure_v1.py",
)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def cell_source(cell):
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value)

def make_code(text, tags=()):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": list(tags)},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

def make_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def find_one(root, name):
    rows = [p for p in pathlib.Path(root).rglob(name) if p.is_file()]
    if len(rows) != 1:
        raise RuntimeError("FILE_COUNT_INVALID:%s:%s" % (name, len(rows)))
    return rows[0]

def patch_runtime(nb):
    if len(nb.get("cells", [])) != 16:
        raise RuntimeError("BASE_TOPOLOGY_INVALID")
    joined = "\n".join(cell_source(c) for c in nb["cells"])
    required = (
        "PHASE2_UNIT_FROZEN_M07_RECIPE_BOUND",
        "PHASE2_UNIT_LEGACY_M07_PERSISTENCE_SKIPPED",
        "def phase2_restore(",
        "_PHASE2_TRANSIENT_HTTP_STATUSES",
        "_phase2_require_restore_headroom",
        "def run_phase2_model_resolution(",
        "PHASE2_UNIT_NEW_FOLD_BOUND_EXCEEDED",
        ".cgpzip",
    )
    missing = [m for m in required if m not in joined]
    if missing:
        raise RuntimeError("RUNTIME_MARKERS_MISSING=" + ",".join(missing))

    config = cell_source(nb["cells"][1]).replace(
        "PHASE2_RUN_EXTERNAL = True", "PHASE2_RUN_EXTERNAL = False"
    )
    pattern = re.compile(
        r"# Exact Phase-2 unit identity injected by pneumonia_phase2_unit\.py\.\n"
        r"PHASE2_UNIT_TOKEN = .*?\n"
        r"PHASE2_UNIT_MODEL_ID = .*?\n"
        r"PHASE2_UNIT_RESOLUTION = .*?\n"
        r"PHASE2_UNIT_ATTEMPT = .*?\n"
        r"PHASE2_UNIT_CONTRACT_SHA256 = .*?\n"
        r"PHASE2_PERSIST_OWNER = .*?\n"
        r"UNLOCK_REMAINING_MODELS = True\n"
        r"CGP_PHASE2_MAX_NEW_FOLDS = .*?\n"
        r"CGP_PHASE2_EXPECTED_RESTORED_FOLDS = .*?\n",
        re.S,
    )
    replacement = """# Master-integrated identity.
PHASE2_UNIT_TOKEN = "MASTER_INTEGRATED_V1"
PHASE2_UNIT_MODEL_ID = "MASTER"
PHASE2_UNIT_RESOLUTION = 0
PHASE2_UNIT_ATTEMPT = 1
PHASE2_UNIT_CONTRACT_SHA256 = "MASTER_INTEGRATED_RUNTIME"
PHASE2_PERSIST_OWNER = "azadka"
UNLOCK_REMAINING_MODELS = True
CGP_PHASE2_MAX_NEW_FOLDS = 5
CGP_PHASE2_EXPECTED_RESTORED_FOLDS = []
"""
    config, count = pattern.subn(replacement, config, count=1)
    if count != 1:
        raise RuntimeError("MASTER_IDENTITY_PATCH_FAILED")

    persistence = cell_source(nb["cells"][13])
    old_slug = '''def phase2_slug(model_id, resolution):
    if str(model_id) == "M07":
        return f"m07-gate-r{int(resolution)}-state-v1-7"
    return f"pneumonia-{model_id.lower()}-r{int(resolution)}-state-v1-7"
'''
    new_slug = '''def phase2_slug(model_id, resolution):
    if str(model_id) == "M07":
        return f"m07-gate-r{int(resolution)}-state-v1-7-master-integrated-v1"
    return f"pneumonia-{model_id.lower()}-r{int(resolution)}-state-v1-7-master-integrated-v1"
'''
    if persistence.count(old_slug) != 1:
        raise RuntimeError("MASTER_STATE_NAMESPACE_PATCH_FAILED")
    persistence = persistence.replace(old_slug, new_slug, 1)

    run = cell_source(nb["cells"][14])
    old_lineage = '''    if restored_folds != list(CGP_PHASE2_EXPECTED_RESTORED_FOLDS):
        raise RuntimeError(
            "BLOCKED_EXACT_EVIDENCE_LINEAGE_MISMATCH: "
            f"expected restored folds {list(CGP_PHASE2_EXPECTED_RESTORED_FOLDS)}, "
            f"observed {restored_folds}"
        )
'''
    new_lineage = '''    if restored_folds != list(range(1, len(restored_folds) + 1)):
        raise RuntimeError("MASTER_RESTORE_LINEAGE_NONCONTIGUOUS: " + str(restored_folds))
    globals()["CGP_PHASE2_EXPECTED_RESTORED_FOLDS"] = list(restored_folds)
'''
    if run.count(old_lineage) != 1:
        raise RuntimeError("MASTER_LINEAGE_PATCH_FAILED")
    run = run.replace(old_lineage, new_lineage, 1)

    sources = []
    for index in range(1, 15):
        if index == 1:
            value = config
        elif index == 13:
            value = persistence
        elif index == 14:
            value = run
        else:
            value = cell_source(nb["cells"][index])
        sources.append("# ===== SOURCE CELL %d =====\n%s" % (index, value.rstrip()))
    runtime = "\n\n".join(sources) + "\n"
    compile(runtime, "<corrected-master-runtime>", "exec")
    return runtime

def metric_block(unit, path):
    for source in unit.get("json_sources") or []:
        for block in source.get("metric_blocks") or []:
            if block.get("path") == path:
                return block.get("values") or {}
    return {}

def compact_results(full):
    units = []
    for unit in full.get("units") or []:
        folds = []
        for fold in unit.get("recovered_folds") or []:
            folds.append({
                "fold_id": fold.get("fold_id"),
                "receipt_sha256": fold.get("receipt_sha256"),
                "run_fingerprint": fold.get("run_fingerprint"),
                "train_metrics": fold.get("train_metrics"),
                "validation_metrics": fold.get("validation_metrics"),
                "training_evidence": fold.get("training_evidence"),
            })
        locked = unit.get("locked_test_recovery") or {}
        units.append({
            "model_id": unit.get("model_id"),
            "resolution": unit.get("resolution"),
            "oof_metrics": metric_block(unit, "oof"),
            "folds": folds,
            "locked_test_recovery": locked,
        })
    return {
        "schema": "pneumonia.phase2.master.compact_results.v1",
        "status": "PASS",
        "summary": {
            key: full.get(key)
            for key in (
                "accounts_complete",
                "model_resolution_units",
                "folds_represented",
                "train_metric_folds",
                "legacy_training_history_folds",
                "training_evidence_folds",
                "validation_metric_folds",
                "locked_test_units_recovered",
                "locked_test_units_not_recovered",
            )
        },
        "legacy_training_metric_limitation": full.get("legacy_training_metric_limitation"),
        "units": units,
    }

def build_evidence(args, repo):
    full_path = find_one(args.full_results, "PHASE2_FULL_RESULTS_MATRIX_V1.json")
    full = json.loads(full_path.read_text(encoding="utf-8"))
    compact = compact_results(full)
    if len(compact["units"]) != 36:
        raise RuntimeError("COMPACT_UNIT_COUNT_INVALID")

    buf = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_LZMA) as archive:
        def put(name, data):
            archive.writestr(name, data)
            manifest[name] = {"sha256": sha(data), "bytes": len(data)}
        put(
            "results/PHASE2_COMPACT_RESULTS.json",
            json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )
        for source_root, prefix in ((args.closure, "closure"), (args.dual, "dual"), (args.paired, "paired")):
            for path in sorted(pathlib.Path(source_root).rglob("*")):
                if not path.is_file() or path.suffix.lower() not in (".json", ".csv", ".md"):
                    continue
                if path.name.startswith("PHASE2_OOF_VECTOR_"):
                    continue
                put(prefix + "/" + path.name, path.read_bytes())
        for source_root, prefix in ((args.external_m09, "external/M09"), (args.external_m10, "external/M10")):
            for path in sorted(pathlib.Path(source_root).rglob("*.json")):
                put(prefix + "/" + path.name, path.read_bytes())
        freeze = repo / "evidence/phase2_final_closure_v1/SELECTION_FREEZE.json"
        put("closure/SELECTION_FREEZE.json", freeze.read_bytes())
        for rel in SUPPORT_SOURCES:
            path = repo / rel
            if path.is_file():
                put("source/" + rel, path.read_bytes())
        put("MANIFEST.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))
    return buf.getvalue(), manifest

def runtime_loader(runtime_b64, runtime_sha):
    return """# Corrected executable runtime, embedded from the exact successful unit source.
import base64, hashlib, lzma, pathlib
MASTER_RUNTIME_LZMA_SHA256=%r
MASTER_RUNTIME_LZMA_B64=%r
_runtime_blob=base64.b64decode(MASTER_RUNTIME_LZMA_B64)
if hashlib.sha256(_runtime_blob).hexdigest()!=MASTER_RUNTIME_LZMA_SHA256:
    raise RuntimeError("MASTER_RUNTIME_BLOB_SHA_MISMATCH")
MASTER_CORRECTED_RUNTIME_SOURCE=lzma.decompress(_runtime_blob).decode("utf-8")
_runtime_path=(pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path.cwd())/"PNEUMONIA_PHASE2_MASTER_CORRECTED_RUNTIME.py"
_runtime_path.write_text(MASTER_CORRECTED_RUNTIME_SOURCE,encoding="utf-8")
compile(MASTER_CORRECTED_RUNTIME_SOURCE,"<PNEUMONIA_PHASE2_MASTER_CORRECTED_RUNTIME>","exec")
if MASTER_RUN_FULL:
    exec(compile(MASTER_CORRECTED_RUNTIME_SOURCE,"<PNEUMONIA_PHASE2_MASTER_CORRECTED_RUNTIME>","exec"),globals())
    print("MASTER_CORRECTED_RUNTIME_LOADED")
else:
    print("MASTER_REPLAY_MODE: corrected runtime syntax verified; training code not executed")
""" % (runtime_sha, runtime_b64)

def master_driver():
    return '''# Single-account resumable full re-execution.
MASTER_MODEL_ORDER=tuple("M%02d"%i for i in range(1,13))
MASTER_RESOLUTIONS=(224,320,384)
MASTER_MAX_UNITS_PER_SESSION=int(os.environ.get("PNEUMONIA_MASTER_MAX_UNITS_PER_SESSION","1"))
if MASTER_RUN_FULL:
    if PHASE2_PERSIST_OWNER!="azadka":
        raise RuntimeError("MASTER_OWNER_DRIFT")
    if MASTER_MAX_UNITS_PER_SESSION<1 or MASTER_MAX_UNITS_PER_SESSION>36:
        raise RuntimeError("MASTER_UNIT_SESSION_BUDGET_INVALID")
    completed=[];count=0
    for model in MASTER_MODEL_ORDER:
        for resolution in MASTER_RESOLUTIONS:
            if count>=MASTER_MAX_UNITS_PER_SESSION:
                break
            restored=phase2_restore(model,resolution)
            CGP_PHASE2_EXPECTED_RESTORED_FOLDS=list(restored)
            CGP_PHASE2_MAX_NEW_FOLDS=5
            result=run_phase2_model_resolution(model,resolution)
            completed.append({"model_id":model,"resolution":resolution,"status":result.get("status")})
            count+=1
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()
        if count>=MASTER_MAX_UNITS_PER_SESSION:
            break
    print("MASTER_FULL_RERUN_SESSION_RESULTS="+json.dumps(completed,sort_keys=True))
else:
    print("SEALED_RESULTS_REPLAY: no training, persistence mutation, Locked-Test inference, or external inference.")
'''

def results_cell(evidence_b64, evidence_sha):
    return """# All final numerical results and scientific evidence.
import base64, hashlib, io, json, pathlib, zipfile
import pandas as pd
from IPython.display import display
MASTER_EVIDENCE_ZIP_SHA256=%r
MASTER_EVIDENCE_B64=%r
blob=base64.b64decode(MASTER_EVIDENCE_B64)
if hashlib.sha256(blob).hexdigest()!=MASTER_EVIDENCE_ZIP_SHA256:
    raise RuntimeError("MASTER_EVIDENCE_SHA_MISMATCH")
root=(pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path.cwd())/"PNEUMONIA_PHASE2_MASTER_EVIDENCE"
root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    archive.extractall(root)
results=json.loads((root/"results/PHASE2_COMPACT_RESULTS.json").read_text())
closure=json.loads((root/"closure/PHASE2_FINAL_SCIENTIFIC_CLOSURE_V1.json").read_text())
if closure.get("status")!="CLOSED_PASS" or closure.get("closure_sha256")!=%r:
    raise RuntimeError("MASTER_CLOSURE_IDENTITY_MISMATCH")
unit_rows=[];fold_rows=[]
for unit in results["units"]:
    oof=unit.get("oof_metrics") or {}
    locked=((unit.get("locked_test_recovery") or {}).get("primary") or {}).get("metrics") or {}
    unit_rows.append({
        "model":unit["model_id"],"resolution":unit["resolution"],
        "oof_accuracy":oof.get("accuracy"),
        "oof_macro_precision":oof.get("macro_precision"),
        "oof_macro_recall":oof.get("macro_recall",oof.get("balanced_accuracy")),
        "oof_auroc":oof.get("auroc"),
        "locked_accuracy":locked.get("accuracy"),
        "locked_macro_precision":locked.get("macro_precision"),
        "locked_macro_recall":locked.get("macro_recall",locked.get("balanced_accuracy")),
        "locked_auroc":locked.get("auroc"),
    })
    for fold in unit.get("folds") or []:
        tm=fold.get("train_metrics");vm=fold.get("validation_metrics")
        fold_rows.append({
            "model":unit["model_id"],"resolution":unit["resolution"],"fold":fold.get("fold_id"),
            "train_accuracy":(tm or {}).get("accuracy") if isinstance(tm,dict) else None,
            "train_macro_precision":(tm or {}).get("macro_precision") if isinstance(tm,dict) else None,
            "train_macro_recall":(tm or {}).get("balanced_accuracy") if isinstance(tm,dict) else None,
            "val_accuracy":(vm or {}).get("accuracy") if isinstance(vm,dict) else None,
            "val_macro_precision":(vm or {}).get("macro_precision") if isinstance(vm,dict) else None,
            "val_macro_recall":(vm or {}).get("balanced_accuracy") if isinstance(vm,dict) else None,
        })
MASTER_UNIT_RESULTS=pd.DataFrame(unit_rows).sort_values(["model","resolution"]).reset_index(drop=True)
MASTER_FOLD_RESULTS=pd.DataFrame(fold_rows).sort_values(["model","resolution","fold"]).reset_index(drop=True)
print("FINAL_STATUS",closure["status"],"remaining_gate",closure["remaining_gate"],"training_folds",closure["training_folds"])
display(MASTER_UNIT_RESULTS)
display(MASTER_FOLD_RESULTS)
paired=root/"dual/DUAL_PR_PAIRED_HOLM_72.csv"
if paired.exists():
    MASTER_DUAL_PR_STATS=pd.read_csv(paired)
    display(MASTER_DUAL_PR_STATS)
finalists=[]
for row in closure["frozen_candidates"]:
    ext=row["external_report_only"]["expanded_pediatric_le18"]
    locked=row["locked_test_report_only"]["metrics"]
    dev=row["development_oof"]
    finalists.append({
        "model":row["model_id"],"resolution":row["resolution"],
        "development_precision":dev["macro_precision"],"development_recall":dev["macro_recall"],
        "locked_precision":locked["macro_precision"],"locked_recall":locked["macro_recall"],
        "external_precision":ext["macro_precision"],"external_recall":ext["macro_recall"],
        "external_accuracy":ext["accuracy"],
    })
MASTER_FINALISTS=pd.DataFrame(finalists)
display(MASTER_FINALISTS)
MASTER_REPLAY_RECEIPT={
    "schema":"pneumonia.phase2.master.integrated.replay.v1",
    "status":"PASS",
    "mode":"FULL_RERUN" if MASTER_RUN_FULL else "SEALED_RESULTS_REPLAY",
    "closure_sha256":closure["closure_sha256"],
    "units":36,"folds":180,
    "runtime_lzma_sha256":MASTER_RUNTIME_LZMA_SHA256,
    "evidence_zip_sha256":MASTER_EVIDENCE_ZIP_SHA256,
}
receipt_path=(pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path.cwd())/"PNEUMONIA_PHASE2_MASTER_REPLAY_RECEIPT.json"
receipt_path.write_text(json.dumps(MASTER_REPLAY_RECEIPT,indent=2,sort_keys=True))
print("MASTER_REPLAY_RECEIPT="+json.dumps(MASTER_REPLAY_RECEIPT,sort_keys=True))
""" % (evidence_sha, evidence_b64, CLOSURE_SHA)

def main():
    parser=argparse.ArgumentParser()
    for name in ("unit_artifact","full_results","closure","dual","paired","external_m09","external_m10","repo_root","output","receipt"):
        parser.add_argument("--"+name.replace("_","-"),dest=name,type=pathlib.Path,required=True)
    args=parser.parse_args()
    unit=find_one(args.unit_artifact,"PNEUMONIA_V17_D260914D_M07_PHASE2_UNLOCK.ipynb")
    raw=unit.read_bytes()
    if sha(raw)!=BASE_SHA:
        raise RuntimeError("BASE_EXECUTED_NOTEBOOK_SHA_MISMATCH")
    notebook=json.loads(raw.decode("utf-8"))
    runtime=patch_runtime(notebook)
    runtime_blob=lzma.compress(runtime.encode("utf-8"),preset=9)
    runtime_b64=base64.b64encode(runtime_blob).decode("ascii")
    runtime_sha=sha(runtime_blob)

    evidence,evidence_manifest=build_evidence(args,args.repo_root)
    evidence_b64=base64.b64encode(evidence).decode("ascii")
    evidence_sha=sha(evidence)

    fixes="\n".join("- %s : %s"%row for row in FIXES)
    intro="""# PNEUMONIA Phase-2 Master Integrated Final V1

This is the single-account master package for M01-M12 at R224/R320/R384.

It contains the exact corrected runtime source taken from a real successful Phase-2 Kaggle unit, then patched only for single-account resumability and a protected master-integrated state namespace. It also embeds every final numerical result needed for the scientific report: 180-fold train/validation evidence, 36-unit OOF metrics, recovered Locked-Test metrics, paired bootstrap/Holm tables, final external reports, selection freeze, and final CLOSED_PASS closure.

Default mode is SEALED_RESULTS_REPLAY and performs no training. To start or resume a fresh campaign under azadka, set PNEUMONIA_MASTER_RUN_FULL=1. The rerun deliberately reuses the already-frozen M07 recipe; it does not redo HPO because HPO was frozen before the final comparator campaign. The state dataset suffix master-integrated-v1 prevents overwrite of the sealed distributed campaign.

A full 180-fold campaign is resumable rather than forced into one Kaggle session. PNEUMONIA_MASTER_MAX_UNITS_PER_SESSION defaults to 1.

## Accepted runtime fix audit
""" + fixes

    control='''# MASTER CONTROL
import os
MASTER_RUN_FULL=os.environ.get("PNEUMONIA_MASTER_RUN_FULL","0").strip().lower() in {"1","true","yes"}
MASTER_RUN_EXTERNAL=os.environ.get("PNEUMONIA_MASTER_RUN_EXTERNAL","0").strip().lower() in {"1","true","yes"}
print("MASTER_RUN_FULL=",MASTER_RUN_FULL,"MASTER_RUN_EXTERNAL=",MASTER_RUN_EXTERNAL)
'''
    final_nb={
        "cells":[
            make_md(intro),
            make_code(control,("master-control",)),
            make_code(runtime_loader(runtime_b64,runtime_sha),("corrected-runtime",)),
            make_code(master_driver(),("master-driver",)),
            make_md("## Sealed final results"),
            make_code(results_cell(evidence_b64,evidence_sha),("master-results",)),
            make_md("## Embedded support source\nThe extracted evidence folder contains current-main source for the paired statistics, external-validation dispatcher/evaluator, and final scientific closure. These files carry the accepted post-training fixes used to produce the sealed external and closure results."),
        ],
        "metadata": notebook.get("metadata",{}),
        "nbformat":4,
        "nbformat_minor":5,
    }
    final_nb["metadata"]["pneumonia_master_integrated"]={
        "schema":"pneumonia.phase2.master.integrated.v1",
        "status":"BUILT_FROM_REAL_EXECUTED_RUNTIME",
        "master_owner":"azadka",
        "kaggle_kernel_ref":"azadka/pneumonia-phase2-master-integrated-v1",
        "base_executed_notebook_sha256":BASE_SHA,
        "runtime_lzma_sha256":runtime_sha,
        "evidence_zip_sha256":evidence_sha,
        "final_closure_sha256":CLOSURE_SHA,
        "default_mode":"SEALED_RESULTS_REPLAY",
        "state_namespace_suffix":"master-integrated-v1",
    }
    for index,cell in enumerate(final_nb["cells"]):
        if cell.get("cell_type")=="code":
            compile(cell_source(cell),"<master-cell-%d>"%index,"exec")

    runtime_text=lzma.decompress(base64.b64decode(runtime_b64)).decode("utf-8")
    critical={
        "frozen_recipe":"PHASE2_UNIT_FROZEN_M07_RECIPE_BOUND" in runtime_text,
        "transient_reads":"_PHASE2_TRANSIENT_HTTP_STATUSES" in runtime_text,
        "storage_headroom":"_phase2_require_restore_headroom" in runtime_text,
        "cgpzip":".cgpzip" in runtime_text,
        "five_fold_budget":"CGP_PHASE2_MAX_NEW_FOLDS = 5" in runtime_text,
        "master_namespace":"master-integrated-v1" in runtime_text,
        "master_owner":'PHASE2_PERSIST_OWNER = "azadka"' in runtime_text,
        "closure":CLOSURE_SHA in cell_source(final_nb["cells"][5]),
    }
    if not all(critical.values()):
        raise RuntimeError("MASTER_FIX_AUDIT_FAILED="+json.dumps(critical,sort_keys=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(final_nb,ensure_ascii=False,indent=1),encoding="utf-8")
    build={
        "schema":"pneumonia.phase2.master.integrated.build.v2",
        "status":"PASS",
        "output":args.output.name,
        "output_sha256":sha(args.output.read_bytes()),
        "bytes":args.output.stat().st_size,
        "cells":len(final_nb["cells"]),
        "base_executed_notebook_sha256":BASE_SHA,
        "runtime_lzma_sha256":runtime_sha,
        "runtime_source_bytes":len(runtime.encode("utf-8")),
        "evidence_zip_sha256":evidence_sha,
        "embedded_evidence_file_count":len(evidence_manifest),
        "final_closure_sha256":CLOSURE_SHA,
        "critical_fix_audit":critical,
        "master_owner":"azadka",
        "kaggle_kernel_ref":"azadka/pneumonia-phase2-master-integrated-v1",
    }
    args.receipt.write_text(json.dumps(build,indent=2,sort_keys=True)+"\n")
    print("PHASE2_MASTER_INTEGRATED_BUILD_PASS "+json.dumps(build,sort_keys=True))

if __name__=="__main__":
    main()
