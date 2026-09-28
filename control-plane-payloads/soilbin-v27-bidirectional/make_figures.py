from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"; FIG=OUT/"figures"; FIG.mkdir(exist_ok=True)
nodes=[f"T{p}-{s}" for p in range(1,7) for s in ("LC1","LC5")]
idx={n:i for i,n in enumerate(nodes)}
e=pd.read_csv(OUT/"02_base_132_directed_edges.csv")
m=pd.read_csv(OUT/"07_memory_depth_results.csv")
reg=pd.read_csv(OUT/"10_regime_analysis.csv")
clusters=pd.read_csv(OUT/"10_regime_pass_clusters.csv")

def heat6(df,spass="source_pass",tpass="target_pass",val="A1_MAE_N",title="",fn=""):
    M=np.full((6,6),np.nan)
    for _,r in df.iterrows(): M[int(r[spass])-1,int(r[tpass])-1]=r[val]
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(M,aspect="auto")
    ax.set_xticks(range(6),[f"T{i}" for i in range(1,7)]);ax.set_yticks(range(6),[f"T{i}" for i in range(1,7)])
    ax.set_xlabel("Target pass");ax.set_ylabel("Source pass");ax.set_title(title);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIG/fn,dpi=160);plt.close(fig)

# 5 asymmetry 12x12
M=np.full((12,12),np.nan)
lookup={(r.source_node,r.target_node):r.A1_MAE_N for _,r in e.iterrows()}
for a in nodes:
  for b in nodes:
    if a==b: continue
    if (a,b) in lookup and (b,a) in lookup: M[idx[a],idx[b]]=lookup[(a,b)]-lookup[(b,a)]
fig,ax=plt.subplots(figsize=(11,9));im=ax.imshow(M,aspect="auto");ax.set_xticks(range(12),nodes,rotation=90);ax.set_yticks(range(12),nodes)
ax.set_xlabel("Target");ax.set_ylabel("Source");ax.set_title("Forward/Reverse Raw-MAE Asymmetry");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIG/"05_forward_reverse_asymmetry.png",dpi=160);plt.close(fig)

# 6 cross directions
heat6(e[(e.source_sensor=="LC1")&(e.target_sensor=="LC5")],"source_pass","target_pass","A1_MAE_N","LC1 to LC5 directed MAE","06a_LC1_to_LC5_map.png")
heat6(e[(e.source_sensor=="LC5")&(e.target_sensor=="LC1")],"source_pass","target_pass","A1_MAE_N","LC5 to LC1 directed MAE","06b_LC5_to_LC1_map.png")
# 7/8 same sensor
heat6(e[(e.source_sensor=="LC1")&(e.target_sensor=="LC1")],"source_pass","target_pass","A1_MAE_N","LC1 temporal directed MAE","07_LC1_temporal_map.png")
heat6(e[(e.source_sensor=="LC5")&(e.target_sensor=="LC5")],"source_pass","target_pass","A1_MAE_N","LC5 temporal directed MAE","08_LC5_temporal_map.png")
# 9 cross-sensor gain
heat6(e[(e.source_sensor=="LC1")&(e.target_sensor=="LC5")],"source_pass","target_pass","gain_MAE_N","LC1 to LC5 gain over conditions","09a_LC1_to_LC5_gain.png")
heat6(e[(e.source_sensor=="LC5")&(e.target_sensor=="LC1")],"source_pass","target_pass","gain_MAE_N","LC5 to LC1 gain over conditions","09b_LC5_to_LC1_gain.png")

# 10 memory depth curves
for target_sensor,mode,fn in [("LC1","LC1_ONLY","10a_LC1_memory_depth.png"),("LC5","LC5_ONLY","10b_LC5_memory_depth.png")]:
    z=m[(m.target_sensor==target_sensor)&(m.sensor_mode==mode)]
    fig,ax=plt.subplots(figsize=(8,5))
    for tp,g in z.groupby("target_pass"):
        g=g.sort_values("history_depth");ax.plot(g.history_depth,g.MAE_N,marker="o",label=f"T{tp}")
    ax.set_xlabel("Contiguous history depth");ax.set_ylabel("OOF MAE (N)");ax.set_title(f"{target_sensor} memory depth");ax.legend(ncol=2);fig.tight_layout();fig.savefig(FIG/fn,dpi=160);plt.close(fig)

# 11 error vs temporal distance
z=e[e.temporal_distance!=0].copy();z["abs_distance"]=z.temporal_distance.abs()
a=z.groupby(["abs_distance","target_sensor"]).A1_MAE_N.mean().reset_index()
fig,ax=plt.subplots(figsize=(8,5))
for s,g in a.groupby("target_sensor"): ax.plot(g.abs_distance,g.A1_MAE_N,marker="o",label=s)
ax.set_xlabel("|Target pass - Source pass|");ax.set_ylabel("Mean edge OOF MAE (N)");ax.set_title("Error vs temporal distance");ax.legend();fig.tight_layout();fig.savefig(FIG/"11_error_vs_temporal_distance.png",dpi=160);plt.close(fig)

# 12 error vs depth difference
a=e.groupby("depth_difference_cm").A1_MAE_N.agg(["mean","median","count"]).reset_index()
fig,ax=plt.subplots(figsize=(7,5));ax.plot(a.depth_difference_cm,a["mean"],marker="o",label="mean");ax.plot(a.depth_difference_cm,a["median"],marker="o",label="median")
ax.set_xlabel("Target depth - source depth (cm)");ax.set_ylabel("OOF MAE (N)");ax.set_title("Error vs source-target depth difference");ax.legend();fig.tight_layout();fig.savefig(FIG/"12_error_vs_depth_difference.png",dpi=160);plt.close(fig)

# 14 regime map
fig,ax=plt.subplots(figsize=(8,4));ax.scatter(clusters["pass"],clusters["cluster"],s=100)
for _,r in clusters.iterrows(): ax.text(r["pass"],r["cluster"]+0.03,f"T{int(r['pass'])}",ha="center")
ax.set_xticks(range(1,7));ax.set_xlabel("Pass");ax.set_ylabel("Descriptive cluster");ax.set_title("Pass regime clustering (descriptive)");fig.tight_layout();fig.savefig(FIG/"14_regime_map.png",dpi=160);plt.close(fig)

# 15 directed graph - all edges, line alpha/width scaled by positive information gain
fig,ax=plt.subplots(figsize=(13,6))
pos={}
for p in range(1,7):
    pos[f"T{p}-LC1"]=(p,1.0);pos[f"T{p}-LC5"]=(p,0.0)
for n,(x,y) in pos.items():
    ax.scatter([x],[y],s=350);ax.text(x,y,n,ha="center",va="center",fontsize=8)
mx=max(float(e.gain_MAE_N.abs().max()),1)
for _,r in e.iterrows():
    a=pos[r.source_node];b=pos[r.target_node];g=float(r.gain_MAE_N)
    alpha=0.08+0.55*min(abs(g)/mx,1);lw=0.3+2.0*min(abs(g)/mx,1)
    patch=FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=6,linewidth=lw,alpha=alpha,
                          connectionstyle="arc3,rad=0.08")
    ax.add_patch(patch)
ax.set_xlim(.5,6.5);ax.set_ylim(-.45,1.45);ax.set_yticks([0,1],["LC5 (5 cm)","LC1 (15 cm)"]);ax.set_xticks(range(1,7),[f"T{i}" for i in range(1,7)])
ax.set_title("Directed SoilBin information graph (edge opacity/width ~ |gain|)");fig.tight_layout();fig.savefig(FIG/"15_directed_information_graph.png",dpi=180);plt.close(fig)
print("FIGURES_DONE",len(list(FIG.glob("*.png"))))
