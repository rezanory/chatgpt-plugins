from __future__ import annotations
import json,sys,time
from pathlib import Path
import run_ensemble as E

idx=int(sys.argv[1]);parts=int(sys.argv[2])
edges=[]
for sp in range(1,7):
    for tp in range(sp+1,7):
        for ss in ("LC1","LC5"):
            for ts in ("LC1","LC5"):
                edges.append((sp,ss,tp,ts))
sel=[x for i,x in enumerate(edges) if i%parts==idx]
t0=time.time()
d,_=E.A.decode_data();wide=E.A.make_wide(d)
res=[]
for x in sel:
    res.append(E.run_edge(wide,*x))
    print("CHUNK_EDGE_DONE",idx,res[-1]["edge_id"],flush=True)
wd=E.HERE/"parallel_chunks";wd.mkdir(exist_ok=True)
(wd/f"chunk_{idx}.json").write_text(json.dumps(res,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print("CHUNK_COMPLETE",idx,len(res),time.time()-t0,flush=True)
