from __future__ import annotations
import json,time
from pathlib import Path
import run_ensemble as E
res=[]
for i in range(6):
    f=E.HERE/"parallel_chunks"/f"chunk_{i}.json"
    if not f.exists():raise RuntimeError("MISSING_CHUNK:"+str(i))
    res.extend(json.loads(f.read_text(encoding="utf-8")))
if len(res)!=60:raise RuntimeError("EXPECTED_60_GOT_"+str(len(res)))
ids=[r["edge_id"] for r in res]
if len(set(ids))!=60:raise RuntimeError("DUPLICATE_EDGE")
res=sorted(res,key=lambda r:(r["source_pass"],r["target_pass"],r["source_sensor"],r["target_sensor"]))
out=E.summarize(res)
out["parallel_fanin"]=True
out["chunk_count"]=6
(E.OUT/"RESULTS.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
(E.OUT/"04_full_audit.json").write_text(json.dumps(res,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("PARALLEL_ENSEMBLE_FANIN_COMPLETE",json.dumps({"best":out["overall_best"],"hard":out["hard_LC5_T2_T6_best"]},ensure_ascii=False),flush=True)
