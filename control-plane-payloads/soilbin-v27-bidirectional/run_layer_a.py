from __future__ import annotations
import base64, csv, gzip, hashlib, itertools, json, math, os, platform, sys, time
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260928
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
EXPECTED_SHA = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
TGT = {"LC1": "LC1_peak_magnitude_N_delta", "LC5": "LC5_peak_magnitude_N_delta"}
DEPTH = {"LC1": 15.0, "LC5": 5.0}
ALPHAS = (0.1, 1.0, 10.0, 100.0)
PROGRESS_WEIGHTS = {
    "preflight_manifest": 5,
    "layer_A_132_edges": 15,
    "layer_B_66_cycle": 10,
    "layer_C_global_model": 10,
    "layer_D_1116_multihistory": 30,
    "regime_lc5_diagnostics": 10,
    "V27_probabilistic": 15,
    "final_outputs": 5,
}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def node_id(p: int, s: str) -> str:
    return f"T{p}-{s}"

NODES = [node_id(p, s) for p in range(1, 7) for s in ("LC1", "LC5")]

def decode_data() -> tuple[pd.DataFrame, bytes]:
    enc = (ROOT / "all_features_51.csv.gz.b64").read_text(encoding="utf-8").strip()
    raw = gzip.decompress(base64.b64decode(enc, validate=True))
    got = sha256_bytes(raw)
    if got != EXPECTED_SHA:
        raise RuntimeError(f"SOURCE_FINGERPRINT_MISMATCH:{got}")
    d = pd.read_csv(BytesIO(raw))
    if len(d) != 51 or d.Run_ID.nunique() != 51:
        raise RuntimeError("EXPECTED_51_UNIQUE_RUNS")
    expected_missing = {"V1W1T1", "V2W3T2", "V3W2T1"}
    expected = {f"V{v}W{w}T{t}" for v in range(1,4) for w in range(1,4) for t in range(1,7)}
    missing = expected - set(d.Run_ID.astype(str))
    if missing != expected_missing:
        raise RuntimeError(f"UNEXPECTED_MISSING:{sorted(missing)}")
    d["Group"] = d.Run_ID.str.extract(r"^(V\dW\d)")[0]
    return d, raw

def make_wide(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, z in d.groupby("Group"):
        rec = {
            "Group": g,
            "Speed": float(z.Speed_level.iloc[0]),
            "Load": float(z.Weight_level.iloc[0]) + 1.0,
        }
        for p in range(1, 7):
            zp = z[z.Pass_T == p]
            for s in ("LC1", "LC5"):
                rec[node_id(p,s)] = float(zp[TGT[s]].iloc[0]) if len(zp) else np.nan
        rows.append(rec)
    w = pd.DataFrame(rows).sort_values("Group").reset_index(drop=True)
    if len(w) != 9:
        raise RuntimeError("EXPECTED_9_GROUPS")
    return w

def cond_features(df: pd.DataFrame) -> np.ndarray:
    s = df["Speed"].to_numpy(float)
    l = df["Load"].to_numpy(float)
    return np.column_stack([s, l, s*s, l*l, s*l])

def design(df: pd.DataFrame, source_col: str | None = None) -> np.ndarray:
    x = cond_features(df)
    if source_col is not None:
        x = np.column_stack([x, df[source_col].to_numpy(float)])
    return x

def fit_predict_ridge(Xtr, ytr, Xte, alpha: float) -> np.ndarray:
    m = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    m.fit(np.asarray(Xtr,float), np.asarray(ytr,float))
    return m.predict(np.asarray(Xte,float))

def select_alpha(inner: pd.DataFrame, target: str, source: str | None, delta: bool) -> tuple[float, dict]:
    groups = sorted(inner.Group.unique())
    scores = []
    for a in ALPHAS:
        yy, pp = [], []
        for vg in groups:
            tr = inner[inner.Group != vg]
            va = inner[inner.Group == vg]
            if len(tr) < 3:
                continue
            ytr = tr[target].to_numpy(float)
            if delta:
                ytr = ytr - tr[source].to_numpy(float)
            pred = fit_predict_ridge(design(tr, source), ytr, design(va, source), a)
            if delta:
                pred = pred + va[source].to_numpy(float)
            yy.extend(va[target].to_numpy(float))
            pp.extend(pred)
        score = float(np.mean(np.abs(np.asarray(yy)-np.asarray(pp)))) if yy else float("inf")
        scores.append((score, float(a)))
    scores.sort(key=lambda q: (q[0], q[1]))
    best = scores[0]
    return best[1], {"selected_alpha": best[1], "inner_mae": best[0],
                     "alpha_scores": [{"alpha":a,"mae":s} for s,a in scores]}

def outer_oof(task: pd.DataFrame, target: str, source: str | None, delta: bool=False):
    pred = np.full(len(task), np.nan)
    audit = []
    for vg in sorted(task.Group.unique()):
        tr = task[task.Group != vg].copy()
        te = task[task.Group == vg].copy()
        alpha, meta = select_alpha(tr, target, source, delta)
        ytr = tr[target].to_numpy(float)
        if delta:
            ytr = ytr - tr[source].to_numpy(float)
        p = fit_predict_ridge(design(tr, source), ytr, design(te, source), alpha)
        if delta:
            p = p + te[source].to_numpy(float)
        pred[te.index.to_numpy()] = p
        audit.append({"outer_group": vg, **meta})
    if not np.isfinite(pred).all():
        raise RuntimeError(f"INCOMPLETE_OOF:{source}->{target}:delta={delta}")
    return pred, audit

def metrics(y, p):
    y = np.asarray(y,float); p = np.asarray(p,float)
    ae = np.abs(y-p)
    sd = float(np.std(y, ddof=1)) if len(y)>1 else float("nan")
    meanabs = float(np.mean(np.abs(y)))
    rmse = float(np.sqrt(mean_squared_error(y,p)))
    return {
        "MAE_N": float(np.mean(ae)),
        "Group_equal_MAE_N": float(np.mean(ae)),
        "RMSE_N": rmse,
        "Median_AE_N": float(np.median(ae)),
        "R2": float(r2_score(y,p)) if len(y)>1 else float("nan"),
        "Bias_N": float(np.mean(p-y)),
        "NMAE_mean_abs": float(np.mean(ae)/meanabs) if meanabs>0 else float("nan"),
        "NMAE_sd": float(np.mean(ae)/sd) if sd>0 else float("nan"),
        "NRMSE_sd": float(rmse/sd) if sd>0 else float("nan"),
    }

def partial_corr(df: pd.DataFrame, source: str, target: str) -> float:
    X = np.column_stack([df.Speed.to_numpy(float), df.Load.to_numpy(float)])
    xs = df[source].to_numpy(float); yt = df[target].to_numpy(float)
    rx = xs - LinearRegression().fit(X,xs).predict(X)
    ry = yt - LinearRegression().fit(X,yt).predict(X)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx,ry)[0,1])

def relation_class(sp, ss, tp, ts):
    if sp == tp:
        return "SAME_PASS_CROSS_SENSOR"
    if ss == ts:
        return "TEMPORAL_SAME_SENSOR"
    return "TEMPORAL_CROSS_SENSOR"

def edge_task(w: pd.DataFrame, sp:int, ss:str, tp:int, ts:str) -> pd.DataFrame:
    src=node_id(sp,ss); tgt=node_id(tp,ts)
    z=w[["Group","Speed","Load",src,tgt]].dropna().reset_index(drop=True)
    return z

def run_edge(w, sp, ss, tp, ts):
    src=node_id(sp,ss); tgt=node_id(tp,ts)
    z=edge_task(w,sp,ss,tp,ts)
    if len(z) < 6:
        raise RuntimeError(f"EDGE_TOO_SMALL:{src}->{tgt}:{len(z)}")
    p0,a0=outer_oof(z,tgt,None,False)
    p1,a1=outer_oof(z,tgt,src,False)
    y=z[tgt].to_numpy(float)
    m0=metrics(y,p0); m1=metrics(y,p1)
    pear=float(pearsonr(z[src],z[tgt]).statistic) if len(z)>=3 else float("nan")
    spear=float(spearmanr(z[src],z[tgt]).statistic) if len(z)>=3 else float("nan")
    row={
        "edge_id": f"{src}__TO__{tgt}",
        "source_node":src,"target_node":tgt,
        "source_pass":sp,"target_pass":tp,
        "source_sensor":ss,"target_sensor":ts,
        "source_depth_cm":DEPTH[ss],"target_depth_cm":DEPTH[ts],
        "depth_difference_cm":DEPTH[ts]-DEPTH[ss],
        "temporal_distance":tp-sp,
        "direction":"FORWARD" if tp>sp else ("REVERSE" if tp<sp else "SAME_PASS"),
        "relation_class":relation_class(sp,ss,tp,ts),
        "n":len(z),"n_groups":z.Group.nunique(),
        "A0_model":"nested_Ridge_conditions_poly",
        "A1_model":"nested_Ridge_conditions_plus_source",
        **{f"A0_{k}":v for k,v in m0.items()},
        **{f"A1_{k}":v for k,v in m1.items()},
        "gain_MAE_N":float(m0["MAE_N"]-m1["MAE_N"]),
        "gain_MAE_pct":float(100*(m0["MAE_N"]-m1["MAE_N"])/m0["MAE_N"]) if m0["MAE_N"]>0 else float("nan"),
        "Pearson_r":pear,"Spearman_rho":spear,
        "partial_corr_speed_load":partial_corr(z,src,tgt),
        "mutual_information":float("nan"),
        "mutual_information_status":"NOT_ESTIMATED_N_6_TO_9_TOO_SMALL",
        "persistence_MAE_N":float("nan"),
        "gain_vs_persistence_N":float("nan"),
        "delta_challenger_MAE_N":float("nan"),
        "delta_challenger_gain_vs_direct_N":float("nan"),
    }
    if ss==ts and sp!=tp:
        pp=z[src].to_numpy(float)
        pm=metrics(y,pp)
        pdlt,ad=outer_oof(z,tgt,src,True)
        md=metrics(y,pdlt)
        row["persistence_MAE_N"]=pm["MAE_N"]
        row["gain_vs_persistence_N"]=pm["MAE_N"]-m1["MAE_N"]
        row["delta_challenger_MAE_N"]=md["MAE_N"]
        row["delta_challenger_gain_vs_direct_N"]=m1["MAE_N"]-md["MAE_N"]
    return row, {"A0":a0,"A1":a1}

def make_pair_results(edges: pd.DataFrame) -> pd.DataFrame:
    lookup={(r.source_node,r.target_node):r for _,r in edges.iterrows()}
    rows=[]
    for i,a in enumerate(NODES):
        for b in NODES[i+1:]:
            ra=lookup[(a,b)]; rb=lookup[(b,a)]
            ap=int(a[1]); bp=int(b[1])
            if ap<bp:
                fwd,rev=ra,rb
            elif bp<ap:
                fwd,rev=rb,ra
            else:
                fwd,rev=ra,rb
            diff=float(fwd.A1_MAE_N-rev.A1_MAE_N)
            den=float((fwd.A1_MAE_N+rev.A1_MAE_N)/2)
            rows.append({
                "node_A":a,"node_B":b,
                "A_to_B_MAE_N":float(ra.A1_MAE_N),
                "B_to_A_MAE_N":float(rb.A1_MAE_N),
                "forward_edge":fwd.edge_id,"reverse_edge":rev.edge_id,
                "forward_MAE_N":float(fwd.A1_MAE_N),"reverse_MAE_N":float(rev.A1_MAE_N),
                "asymmetry_forward_minus_reverse_N":diff,
                "asymmetry_normalized":float(diff/den) if den>0 else float("nan"),
                "n_forward":int(fwd.n),"n_reverse":int(rev.n),
            })
    out=pd.DataFrame(rows)
    if len(out)!=66: raise RuntimeError(f"EXPECTED_66_PAIRS:{len(out)}")
    return out

def build_structural_manifest(w: pd.DataFrame):
    tasks=[]
    counts=Counter()
    for tp in range(1,7):
        others=[p for p in range(1,7) if p!=tp]
        for r in range(1,len(others)+1):
            for subset in itertools.combinations(others,r):
                past=all(p<tp for p in subset)
                future=all(p>tp for p in subset)
                kind="FORECAST" if past else ("RECONSTRUCTION" if future else "SMOOTHING")
                for mode in ("LC1_ONLY","LC5_ONLY","LC1_LC5"):
                    for ts in ("LC1","LC5"):
                        req=[node_id(tp,ts)]
                        for p in subset:
                            if mode=="LC1_ONLY": req.append(node_id(p,"LC1"))
                            elif mode=="LC5_ONLY": req.append(node_id(p,"LC5"))
                            else: req.extend([node_id(p,"LC1"),node_id(p,"LC5")])
                        n=int(w[req].notna().all(axis=1).sum())
                        task_id=f"{kind}__SRC_{'-'.join(map(str,subset))}__{mode}__TO_T{tp}_{ts}"
                        tasks.append({"task_id":task_id,"kind":kind,"source_passes":list(subset),
                                      "sensor_mode":mode,"target_pass":tp,"target_sensor":ts,
                                      "n_groups":n,"executable":bool(n>=6)})
                        counts[kind]+=1
    if len(tasks)!=1116: raise RuntimeError(f"EXPECTED_1116:{len(tasks)}")
    same=[]
    for p in range(1,7):
        for ss,ts in (("LC1","LC5"),("LC5","LC1")):
            req=[node_id(p,ss),node_id(p,ts)]
            n=int(w[req].notna().all(axis=1).sum())
            same.append({"task_id":f"SAMEPASS__T{p}_{ss}__TO__T{p}_{ts}",
                         "kind":"SAME_PASS_CROSS_SENSOR","n_groups":n,"executable":bool(n>=6)})
    if len(same)!=12: raise RuntimeError("EXPECTED_12_SAME_PASS")
    return tasks,same,counts

def heatmaps(edges: pd.DataFrame):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        mats = {
            "directed_mae": "A1_MAE_N",
            "normalized_mae": "A1_NMAE_sd",
            "r2": "A1_R2",
            "gain_over_conditions": "gain_MAE_N",
        }
        for fname,col in mats.items():
            M=np.full((12,12),np.nan)
            idx={n:i for i,n in enumerate(NODES)}
            for _,r in edges.iterrows(): M[idx[r.source_node],idx[r.target_node]]=float(r[col])
            fig,ax=plt.subplots(figsize=(11,9))
            im=ax.imshow(M,aspect="auto")
            ax.set_xticks(range(12),NODES,rotation=90)
            ax.set_yticks(range(12),NODES)
            ax.set_xlabel("Target node"); ax.set_ylabel("Source node")
            ax.set_title(fname.replace("_"," ").title())
            fig.colorbar(im,ax=ax)
            fig.tight_layout()
            fig.savefig(FIG/f"{fname}.png",dpi=160)
            plt.close(fig)
    except Exception as exc:
        (OUT/"FIGURE_ERROR.txt").write_text(type(exc).__name__+":"+str(exc),encoding="utf-8")

def main():
    t0=time.time()
    d,raw=decode_data(); w=make_wide(d)
    multi,same,counts=build_structural_manifest(w)
    script_hash=sha256_bytes(Path(__file__).read_bytes())
    manifest={
        "schema":"soilbin.v27.bidirectional.manifest.v1",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_sha256":sha256_bytes(raw),
        "source_rows":len(d),"groups":sorted(w.Group.tolist()),
        "missing_slots":["V1W1T1","V2W3T2","V3W2T1"],
        "nodes":NODES,"base_directed_edges":132,"unordered_pairs":66,
        "multi_history_tasks":len(multi),"same_pass_cross_sensor_tasks":len(same),
        "distinct_structural_tasks":len(multi)+len(same),
        "multi_history_kind_counts":dict(counts),
        "all_structural_tasks_executable":all(x["executable"] for x in multi+same),
        "n_groups_distribution":dict(Counter(x["n_groups"] for x in multi+same)),
        "progress_weights_pct":PROGRESS_WEIGHTS,
        "validation":{
            "random_split":False,"outer_holdout":"Speed×Load group",
            "hyperparameter_selection":"nested training-side leave-one-group-out",
            "forward_future_actual_features_forbidden":True,
            "reconstruction_and_smoothing_labeled_separately":True,
        },
        "layer_A_model":{
            "conditions_features":["Speed","Load","Speed^2","Load^2","Speed*Load"],
            "A0":"nested Ridge conditions only",
            "A1":"nested Ridge + source node value",
            "alpha_grid":list(ALPHAS),
            "delta_challenger":"same-sensor temporal only",
        },
        "reproducibility":{
            "seed":SEED,"code_sha256":script_hash,"python":sys.version,
            "platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,
            "scipy":scipy.__version__,"sklearn":sklearn.__version__,
        },
        "tasks_1116":multi,"same_pass_12":same,
    }
    (OUT/"01_experiment_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

    rows=[]; audits={}
    for sp in range(1,7):
        for ss in ("LC1","LC5"):
            for tp in range(1,7):
                for ts in ("LC1","LC5"):
                    if sp==tp and ss==ts: continue
                    row,audit=run_edge(w,sp,ss,tp,ts)
                    rows.append(row);audits[row["edge_id"]]=audit
    edges=pd.DataFrame(rows)
    if len(edges)!=132: raise RuntimeError(f"EXPECTED_132_EDGES:{len(edges)}")
    edges.to_csv(OUT/"02_base_132_directed_edges.csv",index=False)
    pairs=make_pair_results(edges)
    pairs.to_csv(OUT/"03_forward_reverse_pairs.csv",index=False)
    (OUT/"layerA_nested_selection_audit.json").write_text(json.dumps(audits,indent=2,allow_nan=False),encoding="utf-8")
    heatmaps(edges)

    best_gain=edges.sort_values("gain_MAE_N",ascending=False).head(10)
    worst_gain=edges.sort_values("gain_MAE_N").head(10)
    max_asym=pairs.reindex(pairs.asymmetry_normalized.abs().sort_values(ascending=False).index).head(10)
    summary={
        "status":"COMPLETE","source_sha256":EXPECTED_SHA,"edge_count":len(edges),"pair_count":len(pairs),
        "n_groups_min":int(edges.n_groups.min()),"n_groups_max":int(edges.n_groups.max()),
        "edges_source_improved_count":int((edges.gain_MAE_N>0).sum()),
        "edges_source_worsened_count":int((edges.gain_MAE_N<0).sum()),
        "median_gain_MAE_N":float(edges.gain_MAE_N.median()),
        "mean_gain_MAE_N":float(edges.gain_MAE_N.mean()),
        "LC5_target_mean_A1_MAE_N":float(edges[edges.target_sensor=="LC5"].A1_MAE_N.mean()),
        "LC1_target_mean_A1_MAE_N":float(edges[edges.target_sensor=="LC1"].A1_MAE_N.mean()),
        "best_gain_edges":best_gain[["edge_id","A0_MAE_N","A1_MAE_N","gain_MAE_N","A1_NMAE_sd"]].to_dict("records"),
        "worst_gain_edges":worst_gain[["edge_id","A0_MAE_N","A1_MAE_N","gain_MAE_N","A1_NMAE_sd"]].to_dict("records"),
        "largest_asymmetry_pairs":max_asym[["node_A","node_B","forward_edge","reverse_edge","forward_MAE_N","reverse_MAE_N","asymmetry_forward_minus_reverse_N","asymmetry_normalized"]].to_dict("records"),
        "structural_tasks_verified":1128,
        "structural_tasks_layerA_executed":132,
        "structural_coverage_pct":100*132/1128,
        "integrated_program_progress_pct":PROGRESS_WEIGHTS["preflight_manifest"]+PROGRESS_WEIGHTS["layer_A_132_edges"],
        "elapsed_seconds":time.time()-t0,
    }
    (OUT/"LAYER_A_SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    main()
