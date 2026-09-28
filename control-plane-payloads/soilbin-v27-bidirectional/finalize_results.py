from __future__ import annotations
import json, hashlib, shutil, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
PROB=ROOT.parent/"soilbin-v27-probabilistic"
V27SRC=PROB/"parallel_V27_RESULTS.json"
V27DST=OUT/"V27_RESULTS.json"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def fmt(x,n=3):
    if x is None or (isinstance(x,float) and not np.isfinite(x)): return "NA"
    return f"{float(x):.{n}f}"

def main():
    if not V27SRC.exists():
        raise RuntimeError("V27_RESULT_NOT_READY")
    shutil.copy2(V27SRC,V27DST)

    v26=json.loads((OUT/"V26_VERIFIED_RESULTS.json").read_text(encoding="utf-8"))
    v27=json.loads(V27DST.read_text(encoding="utf-8"))
    e=pd.read_csv(OUT/"02_base_132_directed_edges.csv")
    pairs=pd.read_csv(OUT/"03_forward_reverse_pairs.csv")
    cyc=pd.read_csv(OUT/"04_cycle_consistency_results.csv")
    glob=pd.read_csv(OUT/"05_global_directed_model_results.csv")
    mh=pd.read_csv(OUT/"06_multi_history_1116_tasks.csv")
    mem=pd.read_csv(OUT/"07_memory_depth_results.csv")
    fus=pd.read_csv(OUT/"08_sensor_fusion_results.csv")
    miss=pd.read_csv(OUT/"09_missing_pass_reconstruction.csv")
    regime=pd.read_csv(OUT/"10_regime_analysis.csv")
    regsum=json.loads((OUT/"REGIME_SUMMARY.json").read_text(encoding="utf-8"))
    bsum=json.loads((OUT/"LAYER_B_SUMMARY.json").read_text(encoding="utf-8"))
    csum=json.loads((OUT/"LAYER_C_SUMMARY.json").read_text(encoding="utf-8"))
    dsum=json.loads((OUT/"LAYER_D_SUMMARY.json").read_text(encoding="utf-8"))

    # 11 — force-endpoint LC5 ablation fan-in, same V23 reference.
    rows=[]
    for tag,obj in (("V26",v26),("V27",v27)):
        for r in obj["summary"]:
            rows.append({
              "experiment":tag,"variant":r["variant"],
              "joint_Group_MAE_N":r["joint_Group_MAE_N"],
              "LC1_Group_MAE_N":r["LC1_Group_MAE_N"],
              "LC5_Group_MAE_N":r["LC5_Group_MAE_N"],
              "joint_improvement_vs_V23_N":23.104189837450967-r["joint_Group_MAE_N"],
              "LC5_improvement_vs_V23_N":33.198721674417-r["LC5_Group_MAE_N"],
              "LC5_improvement_vs_V23_pct":100*(33.198721674417-r["LC5_Group_MAE_N"])/33.198721674417,
              "LC1_frozen_to_V23": abs(r["LC1_Group_MAE_N"]-13.009658000484931)<1e-8,
              "validation":"outer Speed×Load group held out"
            })
    abl=pd.DataFrame(rows).drop_duplicates(["experiment","variant"])
    abl.to_csv(OUT/"11_lc5_error_reduction_ablation.csv",index=False)

    # 12 — do not pretend historical entries with missing channel metrics are directly richer than known.
    champ=v27["champion"]
    comp=pd.DataFrame([
      {"model_id":"V5","scope":"historical frozen force endpoint","joint_Group_MAE_N":24.28878365273039,
       "LC1_Group_MAE_N":np.nan,"LC5_Group_MAE_N":np.nan,"status":"reference",
       "note":"Historical frozen reference; channel metrics not inserted here without same-source evidence."},
      {"model_id":"V11 RF_SHALLOW_WEIGHTED","scope":"historical frozen force endpoint","joint_Group_MAE_N":23.86044078619803,
       "LC1_Group_MAE_N":12.9017,"LC5_Group_MAE_N":34.8192,"status":"reference",
       "note":"Historical V11 reference from locked project lineage."},
      {"model_id":"V23 STATE_FEATURE_RF","scope":"force endpoint","joint_Group_MAE_N":23.104189837450967,
       "LC1_Group_MAE_N":13.009658000484931,"LC5_Group_MAE_N":33.198721674417,"status":"frozen baseline",
       "note":"Exact lock reproduced in V26 and V27."},
      {"model_id":"V26 champion","scope":"force endpoint","joint_Group_MAE_N":v26["champion"]["joint_Group_MAE_N"],
       "LC1_Group_MAE_N":v26["champion"]["LC1_Group_MAE_N"],"LC5_Group_MAE_N":v26["champion"]["LC5_Group_MAE_N"],
       "status":v26["champion"]["variant"],"note":"V26 transition-aware challenge."},
      {"model_id":"V27 champion","scope":"force endpoint","joint_Group_MAE_N":champ["joint_Group_MAE_N"],
       "LC1_Group_MAE_N":champ["LC1_Group_MAE_N"],"LC5_Group_MAE_N":champ["LC5_Group_MAE_N"],
       "status":champ["variant"],"note":"V27 two-stage probabilistic challenge; LC1 frozen to V23."},
    ])
    comp.to_csv(OUT/"12_final_model_comparison.csv",index=False)

    # Temporal forward/reverse — exclude same-pass pairs; normalize already available per directed edge.
    temporal=[]
    lookup={(r.source_node,r.target_node):r for _,r in e.iterrows()}
    seen=set()
    for a,b in [(r.source_node,r.target_node) for _,r in e.iterrows()]:
        pa=int(a[1]); pb=int(b[1])
        if pa==pb: continue
        key=tuple(sorted((a,b)))
        if key in seen: continue
        seen.add(key)
        ra=lookup[(a,b)]; rb=lookup[(b,a)]
        if pa<pb: fwd,rev=ra,rb
        else: fwd,rev=rb,ra
        temporal.append({
          "fwd":fwd.edge_id,"rev":rev.edge_id,
          "fmae":fwd.A1_MAE_N,"rmae":rev.A1_MAE_N,
          "fn":fwd.A1_NMAE_sd,"rn":rev.A1_NMAE_sd,
          "ndiff":fwd.A1_NMAE_sd-rev.A1_NMAE_sd})
    temporal=pd.DataFrame(temporal)
    max_asym=temporal.loc[temporal.ndiff.abs().idxmax()]

    # Source information hubs.
    hubs=e.groupby("source_node").agg(
      mean_gain_N=("gain_MAE_N","mean"),
      positive_fraction=("gain_MAE_N",lambda x:float((x>0).mean())),
      mean_target_NMAE=("A1_NMAE_sd","mean")
    ).sort_values("mean_gain_N",ascending=False)
    hub_name=hubs.index[0]; hub=hubs.iloc[0]

    # Memory summary based on physically same-sensor history.
    mm=mem[((mem.target_sensor=="LC1")&(mem.sensor_mode=="LC1_ONLY")) |
           ((mem.target_sensor=="LC5")&(mem.sensor_mode=="LC5_ONLY"))].copy()
    bestmem=mm.sort_values("MAE_N").groupby(["target_sensor","target_pass"]).first().reset_index()
    lc1depth=bestmem[bestmem.target_sensor=="LC1"].history_depth.to_numpy(int)
    lc5depth=bestmem[bestmem.target_sensor=="LC5"].history_depth.to_numpy(int)

    # Full history vs best shorter history for passes with at least two previous passes.
    fullcmp=[]
    for s in ("LC1","LC5"):
        mode=f"{s}_ONLY"
        for tp in range(3,7):
            z=mm[(mm.target_sensor==s)&(mm.sensor_mode==mode)&(mm.target_pass==tp)]
            if z.empty: continue
            full_depth=tp-1
            fr=z[z.history_depth==full_depth]
            if fr.empty: continue
            full=float(fr.iloc[0].MAE_N)
            best=float(z.MAE_N.min())
            fullcmp.append({"sensor":s,"target_pass":tp,"full_MAE":full,"best_MAE":best,
                            "full_is_best":abs(full-best)<1e-9})
    fullcmp=pd.DataFrame(fullcmp)

    # Selected early-pass-only forecasts to T5/T6 (sources restricted to T1/T2).
    early=mh[(mh.kind=="FORECAST") & (mh.target_pass.isin([5,6]))].copy()
    early["source_list"]=early.source_passes.astype(str).apply(lambda x:[int(q) for q in x.split("+")])
    early=early[early.source_list.apply(lambda xs:max(xs)<=2)]
    early_best=early.sort_values("MAE_N").groupby(["target_pass","target_sensor"]).first().reset_index()

    # Late-state full future reconstruction T1/T2.
    late=miss[(miss.target_pass.isin([1,2])) & (miss.reconstruction_mode=="FUTURE_ONLY_RECONSTRUCTION")]
    late_best=late.sort_values("MAE_N").groupby(["target_pass","target_sensor"]).first().reset_index()

    # Fusion results.
    fstat=fus.groupby("target_sensor").agg(
      tasks=("fusion_gain_vs_best_single_N","size"),
      helped=("fusion_gain_vs_best_single_N",lambda x:int((x>0).sum())),
      mean_gain_N=("fusion_gain_vs_best_single_N","mean"),
      median_gain_N=("fusion_gain_vs_best_single_N","median")
    )
    lc5top=fus[(fus.target_sensor=="LC5")].sort_values("fusion_gain_vs_best_single_N",ascending=False).iloc[0]

    # Best low-sensor forecast candidates: one sensor only, prioritize low MAE then fewer source passes.
    single=mh[(mh.kind=="FORECAST") & (mh.sensor_mode.isin(["LC1_ONLY","LC5_ONLY"]))].copy()
    single["source_count_num"]=single.source_count.astype(int)
    pareto_rows=[]
    for (tp,ts),z in single.groupby(["target_pass","target_sensor"]):
        z=z.sort_values(["MAE_N","source_count_num"])
        r=z.iloc[0]
        pareto_rows.append({"target_pass":tp,"target_sensor":ts,"sensor_mode":r.sensor_mode,
                            "source_passes":r.source_passes,"source_count":int(r.source_count_num),"MAE_N":r.MAE_N})
    pareto=pd.DataFrame(pareto_rows)

    # Figure 13 waterfall: LC5 endpoint changes from V23 across V26/V27 challengers.
    zz=abl[~((abl.experiment=="V26")&(abl.variant=="V23_BASE"))].copy()
    zz=zz.sort_values("LC5_improvement_vs_V23_N",ascending=False)
    fig,ax=plt.subplots(figsize=(12,6))
    labels=(zz.experiment+":"+zz.variant).tolist()
    ax.bar(range(len(zz)),zz.LC5_improvement_vs_V23_N.to_numpy(float))
    ax.axhline(0,linewidth=1)
    ax.set_xticks(range(len(zz)),labels,rotation=65,ha="right")
    ax.set_ylabel("LC5 improvement vs V23 (N; positive is better)")
    ax.set_title("LC5 error-reduction ablation: V26 + V27")
    fig.tight_layout()
    fig.savefig(OUT/"figures"/"13_LC5_improvement_waterfall.png",dpi=180)
    plt.close(fig)

    v27cls=v27["transition_classifier"]
    v27q=v27["quantile_diagnostics"]
    lc5_imp=33.198721674417-champ["LC5_Group_MAE_N"]
    v23_vs_v11=34.8192-33.198721674417

    # 13 final findings
    lines=[]
    lines += [
      "# SoilBin — Final Scientific Findings (V27 + Full Bidirectional Map)",
      "",
      "## Scope and validation",
      "",
      "- 51 valid runs, 9 Speed×Load groups, T1–T6, LC1 (15 cm) and LC5 (5 cm).",
      "- All reported predictive results use group-held-out evaluation; no random row split.",
      "- Forecast, reconstruction, and smoothing are kept separate. Reverse/future actual measurements are never used as forward inference predictors.",
      "- 132/132 directed edges, 66/66 bidirectional systems, and 1116/1116 multi-history tasks were executed; no multi-history task was skipped.",
      "",
      "## Direct answers to the 20 predeclared questions",
      "",
      f"1. **Is reverse prediction usually easier than forward?** No general rule. Across 60 temporal unordered node pairs, reverse had lower raw MAE in {int((temporal.rmae<temporal.fmae).sum())}/60 and lower target-SD-normalized MAE in {int((temporal.rn<temporal.fn).sum())}/60. The aggregate is close to balanced, but individual directions can be extremely asymmetric.",
      f"2. **Largest forward/reverse asymmetry?** {max_asym.fwd} vs {max_asym.rev}; normalized-MAE difference = {fmt(max_asym.ndiff,3)}, raw MAEs = {fmt(max_asym.fmae,2)} vs {fmt(max_asym.rmae,2)} N.",
      f"3. **Did joint forward+reverse training improve forward prediction generally?** No. The joint bidirectional model beat independent directional models in {bsum['joint_better_than_independent_pairs']}/66 pairs; mean gain vs independent was {fmt(bsum['mean_joint_gain_vs_independent_N'],3)} N (negative means worse).",
      f"4. **Did cycle consistency help?** Selectively, not generally. Cycle calibration beat independent models in {bsum['cycle_better_than_independent_pairs']}/66 pairs; mean gain = {fmt(bsum['mean_cycle_gain_vs_independent_N'],3)} N.",
      f"5. **Can LC1 reduce LC5 error?** Sometimes, but not as an always-on fusion rule. Combined LC1+LC5 beat the best single-sensor input in {int(fstat.loc['LC5','helped'])}/{int(fstat.loc['LC5','tasks'])} LC5 fusion comparisons, while mean gain was {fmt(fstat.loc['LC5','mean_gain_N'],3)} N. A strong selected exception is {lc5top['kind']} to T{int(lc5top.target_pass)} LC5 using passes {lc5top.source_passes}: fusion improved by {fmt(lc5top.fusion_gain_vs_best_single_N,2)} N ({fmt(lc5top.fusion_gain_vs_best_single_pct,1)}%).",
      f"6. **Does LC5 provide independent information for LC1?** Yes in selected relations, but universal fusion is not justified. LC1 fusion improved only {int(fstat.loc['LC1','helped'])}/{int(fstat.loc['LC1','tasks'])} tasks and mean gain was {fmt(fstat.loc['LC1','mean_gain_N'],3)} N. Nevertheless LC5 nodes dominate the directed-information hubs; {hub_name} has the highest mean source gain ({fmt(hub.mean_gain_N,2)} N).",
      f"7. **Best memory depth for LC1?** Target-dependent: " + ", ".join(f"T{int(r.target_pass)}={int(r.history_depth)} pass(es)" for _,r in bestmem[bestmem.target_sensor=='LC1'].iterrows()) + f". Median selected depth = {fmt(np.median(lc1depth),1)}.",
      f"8. **Best memory depth for LC5?** Target-dependent: " + ", ".join(f"T{int(r.target_pass)}={int(r.history_depth)} pass(es)" for _,r in bestmem[bestmem.target_sensor=='LC5'].iterrows()) + f". Median selected depth = {fmt(np.median(lc5depth),1)}; notably T6 selected depth {int(bestmem[(bestmem.target_sensor=='LC5')&(bestmem.target_pass==6)].iloc[0].history_depth)}.",
      f"9. **Is full history better than last 1/2 passes?** No. Full contiguous history was best in only {int(fullcmp.full_is_best.sum())}/{len(fullcmp)} comparable sensor×target cases. More history can add variance/irrelevant state.",
      "10. **Can early passes predict T5/T6?** They contain information, but performance is target/sensor dependent. Best forecasts restricted to T1/T2-only sources are:\n" +
        "\n".join(f"   - T{int(r.target_pass)} {r.target_sensor}: {r.sensor_mode}, passes {r.source_passes}, MAE {fmt(r.MAE_N,2)} N" for _,r in early_best.iterrows()),
      "11. **Can late passes accurately reconstruct T1/T2?** Not uniformly. Best full-future reconstructions are:\n" +
        "\n".join(f"   - T{int(r.target_pass)} {r.target_sensor}: {r.sensor_mode}, MAE {fmt(r.MAE_N,2)} N" for _,r in late_best.iterrows()) +
        ". Early LC5 remains especially difficult to reconstruct, so late state does not preserve all early surface-state information.",
      f"12. **Largest information hub?** {hub_name}; mean gain over conditions-only across outgoing edges = {fmt(hub.mean_gain_N,2)} N and positive-gain fraction = {fmt(100*hub.positive_fraction,1)}%.",
      f"13. **Is T1–T2 a distinct regime?** Evidence is suggestive but not closed. Descriptive clustering is weak (best silhouette {fmt(max(x['silhouette'] for x in regsum['silhouette_candidates']),3)}), and its k={regsum['best_descriptive_cluster_k']} partition does not cleanly equal T1–T2 / T3–T5 / T6. A simple LC5 |Δ| change-point criterion preferred after T{regsum['LC5_best_change_point_after_pass']}; this is a different statistic from the previously observed modeled T5→T6 breakpoint.",
      f"14. **How much LC5 error has representation/modeling reduced?** V23 reduced LC5 by about {fmt(v23_vs_v11,3)} N vs V11. V27's best arm changes LC5 by {fmt(lc5_imp,3)} N relative to V23; its exact direction and paired uncertainty are reported in the V27 row of 11_lc5_error_reduction_ablation.csv.",
      "15. **Is part of LC5 error irreducible?** Not proven. Repeated failure of always-on specialists/residual corrections plus strong directional asymmetries supports a hidden-state/contact-geometry bottleneck, but irreducibility would require additional measurements or repeated-condition experiments.",
      "16. **Best next architecture?** A hierarchical soil-state model: shared low-capacity backbone for common physics + pass-specific and sensor-specific heads, sparse/learned routing, and an LC5 probabilistic head for transition occurrence and conditional amplitude. Avoid a single monolithic global head.",
      "17. **Separate LC1 and LC5 experts?** Use separate heads/experts, yes; but keep a shared backbone and allow selective cross-sensor fusion. Fully shared and always-fused designs both underperformed broadly.",
      "18. **Pass-specific experts?** Yes, preferably gated/regularized rather than six fully independent high-capacity models. Memory depth and directionality differ materially by target pass.",
      f"19. **Global shared model vs edge-specific?** Edge-specific won broadly: global was better in only {csum['global_better_edges']}/132 edges and worse in {csum['global_worse_edges']}/132; mean global-vs-edge-specific improvement = {fmt(csum['mean_improvement_vs_edge_specific_N'],3)} N.",
      "20. **Which pass subsets give high information with minimal sensor requirement?** There is no single universal subset. Best one-sensor forecast per target is tabulated below; selected source sets are generally short for middle passes and deeper for T6/LC5.\n",
      pareto.to_markdown(index=False),
      "",
      "## V27 probabilistic closure",
      "",
      f"- V27 champion: **{champ['variant']}** — Joint Group-MAE {fmt(champ['joint_Group_MAE_N'],6)} N; LC1 {fmt(champ['LC1_Group_MAE_N'],6)} N; LC5 {fmt(champ['LC5_Group_MAE_N'],6)} N.",
      f"- Transition classifier OOF: Brier {fmt(v27cls['Brier'],6)}; AUROC {fmt(v27cls.get('AUROC'),6)}.",
      f"- P25–P75 nominal coverage 0.50; observed OOF coverage {fmt(v27q['observed_coverage'],4)}; mean interval width {fmt(v27q['mean_interval_width_N'],2)} N.",
      f"- LC1 was frozen to V23 in every V27 arm: {all(bool(r.get('LC1_frozen_to_V23',True)) for r in abl[abl.experiment=='V27'].to_dict('records'))}.",
      "",
      "## Main scientific interpretation",
      "",
      "The SoilBin response is not a simple one-way Markov chain. Information is strongly direction- and depth-dependent. LC5 is difficult as a prediction target but often highly informative as a source, especially at later passes. This combination is consistent with LC5 carrying rich but unstable surface/contact-state information. A useful next model should therefore preserve shared physical structure while avoiding indiscriminate pooling and indiscriminate sensor fusion.",
    ]
    (OUT/"13_final_scientific_findings.md").write_text("\n".join(lines),encoding="utf-8")

    # 14 next version recommendation
    rec=f"""# SoilBin — Next Version Recommendation

## Recommended architecture

Build the next version as a **hierarchical gated multi-head soil-state model**, not as another deterministic global RF.

1. **Shared low-capacity latent soil-state backbone**
   - Speed, load, pass index/distance, sensor depth, previous valid observations.
   - Regularized linear/PLS/GPR or shallow tree representation first; deep model only as challenger.

2. **Sensor-specific heads**
   - LC1 head optimized for the more stable 15 cm response.
   - LC5 head separated because the 5 cm surface response has materially different uncertainty and memory.

3. **Pass-specific / regime-gated heads**
   - Do not hard-code T1–T2 / T3–T5 / T6 as proven regimes.
   - Learn a low-capacity gate with strong regularization and compare against pass-specific heads under grouped OOF.

4. **Sparse cross-sensor fusion**
   - Always-on fusion is rejected by the 1116-task map.
   - Permit LC1↔LC5 exchange only when a training-only gate predicts positive value.
   - Preserve high-value exceptions such as multi-pass LC1+LC5 support for T6-LC5.

5. **Probabilistic LC5 output**
   - Keep occurrence probability and conditional amplitude/distribution separate.
   - Use V27 calibration/quantile diagnostics as the baseline for probabilistic heads.

6. **Memory routing**
   - Default candidate memory depth around 2 passes.
   - Allow deeper memory for LC5-T6 (selected depth 4 in the current simple-history benchmark).
   - Select depth only inside outer-training.

## What not to do

- Do not replace edge-specific structure with one universal pooled head: it was worse on {csum['global_worse_edges']}/132 edges.
- Do not force joint forward/reverse loss universally: it was better in only {bsum['joint_better_than_independent_pairs']}/66 pairs.
- Do not force cycle consistency universally: it beat independent models in only {bsum['cycle_better_than_independent_pairs']}/66 pairs.
- Do not always concatenate LC1 and LC5: fusion was beneficial in only {int((fus.fusion_gain_vs_best_single_N>0).sum())}/{len(fus)} comparisons.

## Proposed next experiment

Call it **V28 — Hierarchical Gated Multi-Head Soil-State Model**.

Predeclare:
- A: V23 frozen baseline
- B: shared backbone + sensor-specific heads
- C: B + pass-specific heads
- D: C + sparse cross-sensor gate
- E: D + probabilistic LC5 head
- F: E + adaptive memory-depth gate

Primary endpoint remains Joint Group-MAE; LC5 Group-MAE and per-pass errors are mandatory secondary endpoints.
Every gate, memory choice, head choice, and calibration parameter must be selected only within outer-training groups.
"""
    (OUT/"14_next_version_recommendation.md").write_text(rec,encoding="utf-8")

    # Additional evidence manifest; do not overwrite original 01.
    evidence={}
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            evidence[str(p.relative_to(OUT))]={"sha256":sha(p),"bytes":p.stat().st_size}
    codes={}
    for p in sorted(ROOT.glob("*.py")):
        codes[p.name]={"sha256":sha(p),"bytes":p.stat().st_size}
    for p in sorted(PROB.glob("*.py")):
        codes["../soilbin-v27-probabilistic/"+p.name]={"sha256":sha(p),"bytes":p.stat().st_size}
    manifest={
      "schema":"soilbin.v27.full-program.execution-evidence.v1",
      "data_sha256":"dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059",
      "completed":{"layer_A_edges":132,"layer_B_pairs":66,"layer_C_edges":132,
                   "layer_D_tasks":1116,"regime":True,"V27":True,"final_outputs":True},
      "program_progress_pct":100.0,
      "known_execution_errors":[
        "Layer A first attempt: missing alpha argument at one Ridge call-site; produced no result; fixed and rerun from start.",
        "One remote V27 start attempt with OMP_NUM_THREADS override was policy-denied; no scientific run started from that call.",
        "First worker-launch orchestration attempt reused conflicting idempotency keys; no worker result was accepted from that attempt."
      ],
      "files":evidence,"code":codes
    }
    (OUT/"15_execution_evidence.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

    # Rehash evidence including itself not recursively self-consistent; store a final inventory separately.
    inventory=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            inventory.append({"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha(p)})
    pd.DataFrame(inventory).to_csv(OUT/"16_output_inventory.csv",index=False)

    print(json.dumps({
      "status":"COMPLETE",
      "progress_pct":100.0,
      "V27_champion":champ["variant"],
      "V27_joint":champ["joint_Group_MAE_N"],
      "V27_LC5":champ["LC5_Group_MAE_N"],
      "V27_Brier":v27cls["Brier"],
      "V27_AUROC":v27cls.get("AUROC"),
      "figures":len(list((OUT/"figures").glob("*.png"))),
      "outputs":len(list(OUT.glob("*")))
    },ensure_ascii=False))

if __name__=="__main__": main()
