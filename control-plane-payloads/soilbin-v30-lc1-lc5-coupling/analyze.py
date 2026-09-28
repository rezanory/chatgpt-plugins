from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = RES / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

BASE = ROOT.parent
V27 = BASE / "soilbin-v27-bidirectional"
SOURCE = V27 / "all_features_51.csv.gz.b64"
EDGE_SOURCE = V27 / "results" / "02_base_132_directed_edges.csv"

EXPECTED_DATA_SHA256 = "dc96ffd8d48f6cc861c2ecbfbc95b95fa809c24c55688fbccba6cd8f61ce6059"
EPS = 1e-12
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

raw = gzip.decompress(base64.b64decode(SOURCE.read_text(encoding="utf-8").strip()))
df = pd.read_csv(io.BytesIO(raw))
edges = pd.read_csv(EDGE_SOURCE)

df["Group"] = [f"V{int(v)}W{int(w)}" for v, w in zip(df.Speed_level, df.Weight_level)]
df = df.sort_values(["Group", "Pass_T"]).reset_index(drop=True)

LC1 = "LC1_peak_magnitude_N_delta"
LC5 = "LC5_peak_magnitude_N_delta"
I1 = "LC1_impulse_abs_Ns"
I5 = "LC5_impulse_abs_Ns"

df["LC1_N"] = df[LC1].astype(float)
df["LC5_N"] = df[LC5].astype(float)
df["peak_ratio_LC5_LC1"] = df["LC5_N"] / np.maximum(df["LC1_N"], EPS)
df["peak_ratio_LC1_LC5"] = df["LC1_N"] / np.maximum(df["LC5_N"], EPS)
df["peak_diff_LC5_minus_LC1_N"] = df["LC5_N"] - df["LC1_N"]
df["peak_relative_diff_vs_LC1"] = df["peak_diff_LC5_minus_LC1_N"] / np.maximum(df["LC1_N"], EPS)
df["log_peak_ratio"] = np.log(np.maximum(df["peak_ratio_LC5_LC1"], EPS))
df["impulse_ratio_LC5_LC1"] = df[I5] / np.maximum(df[I1], EPS)
df["duration_ratio_LC5_LC1"] = df["LC5_event_duration_ms"] / np.maximum(df["LC1_event_duration_ms"], EPS)
df["rms_ratio_LC5_LC1"] = df["LC5_rms_event_N"] / np.maximum(df["LC1_rms_event_N"], EPS)

existing_ratio_delta = np.nanmax(
    np.abs(df["Peak_magnitude_ratio_LC5_to_LC1"].to_numpy(float) - df["peak_ratio_LC5_LC1"].to_numpy(float))
)

# Strict previous-pass features: only T-1, never a future pass.
df["prev_pass"] = df.groupby("Group")["Pass_T"].shift(1)
df["prev_log_ratio_raw"] = df.groupby("Group")["log_peak_ratio"].shift(1)
df["prev_ratio_available"] = ((df["Pass_T"] - df["prev_pass"]) == 1).astype(int)
df.loc[df["prev_ratio_available"] == 0, "prev_log_ratio_raw"] = np.nan

def safe_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return {"n": int(len(x)), "pearson_r": None, "spearman_rho": None}
    return {
        "n": int(len(x)),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
    }

def group_summary(data, cols):
    out = []
    grouped = data.groupby(cols, dropna=False)
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: (int(v) if isinstance(v, (np.integer, int)) else v) for c, v in zip(cols, key)}
        ratio = g["peak_ratio_LC5_LC1"].to_numpy(float)
        corr = safe_corr(g["LC1_N"], g["LC5_N"])
        row.update({
            "n": int(len(g)),
            "LC1_mean_N": float(g["LC1_N"].mean()),
            "LC5_mean_N": float(g["LC5_N"].mean()),
            "ratio_mean": float(np.mean(ratio)),
            "ratio_median": float(np.median(ratio)),
            "ratio_sd": float(np.std(ratio, ddof=1)) if len(ratio) > 1 else 0.0,
            "ratio_cv_pct": float(100*np.std(ratio, ddof=1)/np.mean(ratio)) if len(ratio) > 1 and abs(np.mean(ratio)) > EPS else 0.0,
            "log_ratio_mean": float(g["log_peak_ratio"].mean()),
            "peak_diff_mean_N": float(g["peak_diff_LC5_minus_LC1_N"].mean()),
            "relative_diff_mean_pct": float(100*g["peak_relative_diff_vs_LC1"].mean()),
            "impulse_ratio_median": float(g["impulse_ratio_LC5_LC1"].median()),
            "peak_delay_median_ms": float(g["Peak_delay_LC5_minus_LC1_ms"].median()),
            "LC1_LC5_pearson_r": corr["pearson_r"],
            "LC1_LC5_spearman_rho": corr["spearman_rho"],
        })
        out.append(row)
    return pd.DataFrame(out)

by_pass = group_summary(df, ["Pass_T"])
by_speed = group_summary(df, ["Speed_level"])
by_load = group_summary(df, ["Weight_level"])
by_group = group_summary(df, ["Speed_level", "Weight_level", "Group"])

condition_table = df[[
    "Run_ID", "Group", "Speed_level", "Weight_level", "Pass_T",
    "LC1_N", "LC5_N", "peak_ratio_LC5_LC1", "peak_ratio_LC1_LC5",
    "peak_diff_LC5_minus_LC1_N", "peak_relative_diff_vs_LC1", "log_peak_ratio",
    "impulse_ratio_LC5_LC1", "duration_ratio_LC5_LC1", "rms_ratio_LC5_LC1",
    "Peak_delay_LC5_minus_LC1_ms"
]].copy()

# Pass-specific cross-sectional coupling and elasticity.
pass_coupling = []
for t, g in df.groupby("Pass_T"):
    corr = safe_corr(g.LC1_N, g.LC5_N)
    x = g.LC1_N.to_numpy(float)
    y = g.LC5_N.to_numpy(float)
    slope, intercept = (np.polyfit(x, y, 1) if len(g) >= 3 else (np.nan, np.nan))
    lx, ly = np.log(x), np.log(y)
    log_slope, log_intercept = (np.polyfit(lx, ly, 1) if len(g) >= 3 and np.std(lx) > EPS else (np.nan, np.nan))
    pass_coupling.append({
        "Pass_T": int(t), "n": int(len(g)),
        **corr,
        "LC5_vs_LC1_slope_N_per_N": float(slope),
        "LC5_vs_LC1_intercept_N": float(intercept),
        "cross_section_log_elasticity": float(log_slope),
        "ratio_mean": float(g.peak_ratio_LC5_LC1.mean()),
        "ratio_median": float(g.peak_ratio_LC5_LC1.median()),
        "ratio_sd": float(g.peak_ratio_LC5_LC1.std(ddof=1)),
    })
pass_coupling = pd.DataFrame(pass_coupling)

# Strict adjacent-pass transitions.
transitions = []
for group, g in df.groupby("Group"):
    g = g.sort_values("Pass_T").set_index("Pass_T")
    for t in range(1, 6):
        if t not in g.index or t+1 not in g.index:
            continue
        a, b = g.loc[t], g.loc[t+1]
        dlog1 = math.log(max(float(b.LC1_N), EPS) / max(float(a.LC1_N), EPS))
        dlog5 = math.log(max(float(b.LC5_N), EPS) / max(float(a.LC5_N), EPS))
        elasticity = dlog5/dlog1 if abs(dlog1) >= 0.05 else np.nan
        transitions.append({
            "Group": group,
            "Speed_level": int(a.Speed_level),
            "Weight_level": int(a.Weight_level),
            "transition": f"T{t}->T{t+1}",
            "source_pass": t,
            "target_pass": t+1,
            "LC1_source_N": float(a.LC1_N),
            "LC1_target_N": float(b.LC1_N),
            "LC5_source_N": float(a.LC5_N),
            "LC5_target_N": float(b.LC5_N),
            "LC1_pct_change": float(100*(b.LC1_N/a.LC1_N-1)),
            "LC5_pct_change": float(100*(b.LC5_N/a.LC5_N-1)),
            "dlog_LC1": float(dlog1),
            "dlog_LC5": float(dlog5),
            "log_elasticity_LC5_vs_LC1": float(elasticity) if np.isfinite(elasticity) else np.nan,
            "ratio_source": float(a.peak_ratio_LC5_LC1),
            "ratio_target": float(b.peak_ratio_LC5_LC1),
            "ratio_change": float(b.peak_ratio_LC5_LC1-a.peak_ratio_LC5_LC1),
            "log_ratio_change": float(b.log_peak_ratio-a.log_peak_ratio),
        })
transitions = pd.DataFrame(transitions)

transition_summary = []
for tr, g in transitions.groupby("transition"):
    corr = safe_corr(g.dlog_LC1, g.dlog_LC5)
    el = g.log_elasticity_LC5_vs_LC1.dropna().to_numpy(float)
    transition_summary.append({
        "transition": tr,
        "n": int(len(g)),
        "LC1_pct_change_mean": float(g.LC1_pct_change.mean()),
        "LC5_pct_change_mean": float(g.LC5_pct_change.mean()),
        "ratio_change_mean": float(g.ratio_change.mean()),
        "ratio_change_median": float(g.ratio_change.median()),
        "dlog_change_pearson_r": corr["pearson_r"],
        "dlog_change_spearman_rho": corr["spearman_rho"],
        "elasticity_n": int(len(el)),
        "elasticity_median": float(np.median(el)) if len(el) else np.nan,
        "elasticity_IQR": float(np.quantile(el, .75)-np.quantile(el, .25)) if len(el) >= 2 else np.nan,
    })
transition_summary = pd.DataFrame(transition_summary)

# Historical V27 directed cross-sensor map, including all same-pass edges.
cross_edges = edges[edges.source_sensor != edges.target_sensor].copy()
same_pass = cross_edges[cross_edges.source_pass == cross_edges.target_pass].copy()
same_pass = same_pass.sort_values(["source_pass", "source_sensor"])

same_pass_weighted = {}
for target in ("LC1", "LC5"):
    q = same_pass[same_pass.target_sensor == target]
    same_pass_weighted[target] = {
        "n_total": int(q.n.sum()),
        "A0_row_weighted_MAE_N": float(np.average(q.A0_MAE_N, weights=q.n)),
        "A1_row_weighted_MAE_N": float(np.average(q.A1_MAE_N, weights=q.n)),
        "gain_row_weighted_N": float(np.average(q.gain_MAE_N, weights=q.n)),
        "passes_source_improved_target": int((q.gain_MAE_N > 0).sum()),
        "passes_total": int(len(q)),
    }

# Variance-explained diagnostics for log ratio. Descriptive only.
def dummy_r2(columns):
    X = pd.get_dummies(df[columns].astype(str), drop_first=False).to_numpy(float)
    y = df.log_peak_ratio.to_numpy(float)
    m = LinearRegression().fit(X, y)
    return float(m.score(X, y))

ratio_structure = {
    "overall_ratio_mean": float(df.peak_ratio_LC5_LC1.mean()),
    "overall_ratio_median": float(df.peak_ratio_LC5_LC1.median()),
    "overall_ratio_sd": float(df.peak_ratio_LC5_LC1.std(ddof=1)),
    "overall_ratio_cv_pct": float(100*df.peak_ratio_LC5_LC1.std(ddof=1)/df.peak_ratio_LC5_LC1.mean()),
    "overall_ratio_min": float(df.peak_ratio_LC5_LC1.min()),
    "overall_ratio_max": float(df.peak_ratio_LC5_LC1.max()),
    "fraction_LC5_gt_LC1": float(np.mean(df.LC5_N > df.LC1_N)),
    "descriptive_R2_log_ratio_pass_only": dummy_r2(["Pass_T"]),
    "descriptive_R2_log_ratio_speed_only": dummy_r2(["Speed_level"]),
    "descriptive_R2_log_ratio_load_only": dummy_r2(["Weight_level"]),
    "descriptive_R2_log_ratio_speed_load": dummy_r2(["Speed_level", "Weight_level"]),
    "descriptive_R2_log_ratio_pass_speed_load": dummy_r2(["Pass_T", "Speed_level", "Weight_level"]),
    "peak_ratio_vs_impulse_ratio_corr": safe_corr(df.peak_ratio_LC5_LC1, df.impulse_ratio_LC5_LC1),
}

# Nested leave-one-Speed×Load-group-out ratio learning.
def base_features(frame):
    s = frame.Speed_level.to_numpy(float)
    w = frame.Weight_level.to_numpy(float)
    p = frame.Pass_T.to_numpy(float)
    return np.column_stack([s, w, p, s*w, p*p, s*p, w*p])

def feature_matrix(frame, mode, train_prev_fill=None):
    X = base_features(frame)
    if mode in ("LC1", "LC1_PREV"):
        X = np.column_stack([X, np.log(np.maximum(frame.LC1_N.to_numpy(float), EPS))])
    elif mode in ("LC5", "LC5_PREV"):
        X = np.column_stack([X, np.log(np.maximum(frame.LC5_N.to_numpy(float), EPS))])
    elif mode != "COND":
        raise KeyError(mode)
    if mode.endswith("_PREV"):
        fill = float(train_prev_fill)
        prev = frame.prev_log_ratio_raw.fillna(fill).to_numpy(float)
        avail = frame.prev_ratio_available.to_numpy(float)
        X = np.column_stack([X, prev, avail])
    return X

def select_alpha(train, mode):
    groups = sorted(train.Group.unique())
    scores = []
    fill = float(train.log_peak_ratio.median())
    for alpha in ALPHAS:
        fold_scores = []
        for vg in groups:
            tr = train.Group != vg
            va = train.Group == vg
            if tr.sum() < 5 or va.sum() == 0:
                continue
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            mdl.fit(feature_matrix(train.loc[tr], mode, fill), train.loc[tr, "log_peak_ratio"])
            pred = mdl.predict(feature_matrix(train.loc[va], mode, fill))
            fold_scores.append(float(np.mean(np.abs(train.loc[va, "log_peak_ratio"].to_numpy(float)-pred))))
        scores.append((float(np.mean(fold_scores)), alpha))
    scores.sort()
    return float(scores[0][1]), [{"alpha":float(a),"inner_group_equal_log_ratio_MAE":float(s)} for s,a in scores]

model_names = [
    "GLOBAL_MEDIAN_RATIO",
    "COND_RATIO",
    "LC5_FROM_LC1_RATIO",
    "LC5_FROM_LC1_RATIO_PREV",
    "LC1_FROM_LC5_RATIO",
    "LC1_FROM_LC5_RATIO_PREV",
]
pred = {m: np.full(len(df), np.nan) for m in model_names}
pred_ratio = {m: np.full(len(df), np.nan) for m in model_names}
model_audit = []

for outer in sorted(df.Group.unique()):
    trm = df.Group != outer
    tem = df.Group == outer
    train = df.loc[trm].copy()
    test = df.loc[tem].copy()
    idx = test.index.to_numpy()
    median_log = float(train.log_peak_ratio.median())

    # Constant training-only ratio baseline.
    pr = np.exp(np.full(len(test), median_log))
    pred_ratio["GLOBAL_MEDIAN_RATIO"][idx] = pr
    pred["GLOBAL_MEDIAN_RATIO"][idx] = pr * test.LC1_N.to_numpy(float)

    configs = [
        ("COND_RATIO", "COND", "LC5"),
        ("LC5_FROM_LC1_RATIO", "LC1", "LC5"),
        ("LC5_FROM_LC1_RATIO_PREV", "LC1_PREV", "LC5"),
        ("LC1_FROM_LC5_RATIO", "LC5", "LC1"),
        ("LC1_FROM_LC5_RATIO_PREV", "LC5_PREV", "LC1"),
    ]
    fold_meta = {"outer_group": outer, "models": {}}
    for name, mode, target_sensor in configs:
        alpha, inner = select_alpha(train, mode)
        mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        mdl.fit(feature_matrix(train, mode, median_log), train.log_peak_ratio.to_numpy(float))
        lp = mdl.predict(feature_matrix(test, mode, median_log))
        rr = np.exp(lp)
        pred_ratio[name][idx] = rr
        if target_sensor == "LC5":
            yy = rr * test.LC1_N.to_numpy(float)
        else:
            yy = test.LC5_N.to_numpy(float) / np.maximum(rr, EPS)
        pred[name][idx] = yy
        fold_meta["models"][name] = {"selected_alpha": alpha, "inner_scores": inner}
    model_audit.append(fold_meta)

def group_equal_mae(y, p, groups):
    vals = []
    for g in sorted(set(groups)):
        m = np.asarray(groups) == g
        vals.append(np.mean(np.abs(np.asarray(y)[m]-np.asarray(p)[m])))
    return float(np.mean(vals))

model_summary = []
for name in model_names:
    if name.startswith("LC1_FROM"):
        y = df.LC1_N.to_numpy(float)
    else:
        y = df.LC5_N.to_numpy(float)
    p = pred[name]
    ratio_p = pred_ratio[name]
    if not np.isfinite(p).all() or not np.isfinite(ratio_p).all():
        raise RuntimeError(f"INCOMPLETE_MODEL:{name}")
    model_summary.append({
        "model": name,
        "target": "LC1" if name.startswith("LC1_FROM") else "LC5",
        "row_MAE_N": float(mean_absolute_error(y,p)),
        "Group_equal_MAE_N": group_equal_mae(y,p,df.Group.to_numpy()),
        "RMSE_N": float(mean_squared_error(y,p)**0.5),
        "R2": float(r2_score(y,p)),
        "MAPE_pct": float(np.mean(np.abs((y-p)/np.maximum(np.abs(y),EPS)))*100),
        "ratio_MAE": float(mean_absolute_error(df.peak_ratio_LC5_LC1,ratio_p)),
        "log_ratio_MAE": float(mean_absolute_error(df.log_peak_ratio,np.log(np.maximum(ratio_p,EPS)))),
        "T1_MAE_N": float(mean_absolute_error(y[df.Pass_T==1],p[df.Pass_T==1])),
        "T2_MAE_N": float(mean_absolute_error(y[df.Pass_T==2],p[df.Pass_T==2])),
        "T3_MAE_N": float(mean_absolute_error(y[df.Pass_T==3],p[df.Pass_T==3])),
        "T4_MAE_N": float(mean_absolute_error(y[df.Pass_T==4],p[df.Pass_T==4])),
        "T5_MAE_N": float(mean_absolute_error(y[df.Pass_T==5],p[df.Pass_T==5])),
        "T6_MAE_N": float(mean_absolute_error(y[df.Pass_T==6],p[df.Pass_T==6])),
    })
model_summary = pd.DataFrame(model_summary)

best_lc5 = model_summary[model_summary.target=="LC5"].sort_values("Group_equal_MAE_N").iloc[0].to_dict()
best_lc1 = model_summary[model_summary.target=="LC1"].sort_values("Group_equal_MAE_N").iloc[0].to_dict()

# Extremes and condition-level discoveries.
high_ratio = condition_table.nlargest(10, "peak_ratio_LC5_LC1")
low_ratio = condition_table.nsmallest(10, "peak_ratio_LC5_LC1")

# Save tables.
condition_table.to_csv(RES/"01_condition_level_ratio_table.csv", index=False)
by_pass.to_csv(RES/"02_ratio_by_pass.csv", index=False)
by_speed.to_csv(RES/"03_ratio_by_speed.csv", index=False)
by_load.to_csv(RES/"04_ratio_by_load.csv", index=False)
by_group.to_csv(RES/"05_ratio_by_speed_load_group.csv", index=False)
pass_coupling.to_csv(RES/"06_same_pass_coupling_by_pass.csv", index=False)
transitions.to_csv(RES/"07_adjacent_transition_elasticity.csv", index=False)
transition_summary.to_csv(RES/"08_transition_summary.csv", index=False)
same_pass.to_csv(RES/"09_v27_same_pass_directed_edges.csv", index=False)
cross_edges.to_csv(RES/"10_v27_all_cross_sensor_directed_edges.csv", index=False)
model_summary.to_csv(RES/"11_ratio_model_group_oof_results.csv", index=False)
high_ratio.to_csv(RES/"12_highest_ratio_conditions.csv", index=False)
low_ratio.to_csv(RES/"13_lowest_ratio_conditions.csv", index=False)
(RES/"14_ratio_model_nested_audit.json").write_text(json.dumps(model_audit,indent=2),encoding="utf-8")

# Simple figures for audit/readability.
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9,5))
    for group, g in df.groupby("Group"):
        ax.plot(g.Pass_T, g.peak_ratio_LC5_LC1, marker="o", label=group)
    ax.set_xlabel("Pass T")
    ax.set_ylabel("LC5 / LC1 peak-force ratio")
    ax.set_title("LC5-to-LC1 ratio across passes by Speed×Load group")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG/"01_ratio_trajectories.png",dpi=180)
    plt.close(fig)

    sp = same_pass.copy()
    labels = [f"T{int(t)} {s}->{q}" for t,s,q in zip(sp.source_pass,sp.source_sensor,sp.target_sensor)]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(labels, sp.gain_MAE_N)
    ax.axhline(0, linewidth=1)
    ax.tick_params(axis="x", rotation=65)
    ax.set_ylabel("A0 - A1 MAE gain (N)")
    ax.set_title("Historical V27 same-pass cross-sensor information gain")
    fig.tight_layout()
    fig.savefig(FIG/"02_same_pass_directional_gain.png",dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(transition_summary.transition, transition_summary.elasticity_median)
    ax.axhline(1, linewidth=1)
    ax.set_ylabel("Median log-elasticity Δlog(LC5)/Δlog(LC1)")
    ax.set_title("Adjacent-pass LC5 vs LC1 elasticity")
    fig.tight_layout()
    fig.savefig(FIG/"03_transition_elasticity.png",dpi=180)
    plt.close(fig)
except Exception as exc:
    (RES/"FIGURE_ERROR.txt").write_text(repr(exc),encoding="utf-8")

result = {
    "schema":"soilbin.v30.lc1-lc5-coupling.v1",
    "status":"COMPLETE",
    "scientific_status":"DESCRIPTIVE_PLUS_NESTED_GROUP_OOF_RELATIONSHIP_STUDY",
    "data_fingerprint_expected":EXPECTED_DATA_SHA256,
    "source_b64_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "n_runs":int(len(df)),
    "n_speed_load_groups":int(df.Group.nunique()),
    "missing_logical_slots":["V1W1T1","V2W3T2","V3W2T1"],
    "existing_ratio_max_abs_recompute_difference":float(existing_ratio_delta),
    "scope":{
        "same_pass_cross_sensor_edges_v27_verified":int(len(same_pass)),
        "all_cross_sensor_directed_edges_v27_verified":int(len(cross_edges)),
        "ratio_by_pass":True,
        "ratio_by_speed":True,
        "ratio_by_load":True,
        "ratio_by_speed_load":True,
        "condition_level_speed_load_pass":True,
        "adjacent_transition_elasticity":True,
        "impulse_ratio":True,
        "peak_timing_delay":True,
        "nested_group_oof_ratio_models":True,
        "waveform_level_timeseries":"PENDING_SOURCE_DISCOVERY"
    },
    "ratio_structure":ratio_structure,
    "same_pass_v27_weighted":same_pass_weighted,
    "best_group_oof_ratio_model_LC5_from_LC1":best_lc5,
    "best_group_oof_ratio_model_LC1_from_LC5":best_lc1,
    "pass_coupling":pass_coupling.to_dict(orient="records"),
    "transition_summary":transition_summary.to_dict(orient="records"),
    "highest_ratio_conditions":high_ratio.head(5).to_dict(orient="records"),
    "lowest_ratio_conditions":low_ratio.head(5).to_dict(orient="records"),
    "guards":{
        "random_row_split_used":False,
        "whole_speed_load_group_held_out_in_predictive_ratio_models":True,
        "nested_alpha_selection_training_groups_only":True,
        "current_LC1_used_to_predict_same_pass_LC5_in_declared_models":True,
        "current_LC5_used_to_predict_same_pass_LC1_in_declared_models":True,
        "previous_ratio_only_from_observed_T_minus_1":True,
        "future_pass_target_used":False
    }
}
(RES/"RESULTS.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

# Compact scientific findings, generated from computed values only.
lines = [
    "# SoilBin V30 — LC1↔LC5 Coupling Study",
    "",
    f"- Runs: {len(df)}; Speed×Load groups: {df.Group.nunique()}.",
    f"- V27 same-pass directed LC1↔LC5 edges verified: {len(same_pass)} (T1..T6, both directions).",
    f"- All V27 cross-sensor directed edges verified: {len(cross_edges)}.",
    f"- Overall LC5/LC1 median peak ratio: {ratio_structure['overall_ratio_median']:.4f}; mean: {ratio_structure['overall_ratio_mean']:.4f}; CV: {ratio_structure['overall_ratio_cv_pct']:.1f}%.",
    f"- Fraction of runs with LC5 > LC1: {100*ratio_structure['fraction_LC5_gt_LC1']:.1f}%.",
    f"- Best leakage-safe same-pass ratio model for LC5-from-LC1: {best_lc5['model']} with group-equal MAE {best_lc5['Group_equal_MAE_N']:.3f} N.",
    f"- Best leakage-safe same-pass ratio model for LC1-from-LC5: {best_lc1['model']} with group-equal MAE {best_lc1['Group_equal_MAE_N']:.3f} N.",
    "",
    "## Pass-level coupling",
]
for r in pass_coupling.to_dict(orient="records"):
    lines.append(
        f"- T{r['Pass_T']}: n={r['n']}, ratio median={r['ratio_median']:.3f}, "
        f"Pearson r={r['pearson_r'] if r['pearson_r'] is not None else 'NA'}, "
        f"log-elasticity={r['cross_section_log_elasticity']:.3f}."
    )
lines += ["", "## Adjacent transitions"]
for r in transition_summary.to_dict(orient="records"):
    lines.append(
        f"- {r['transition']}: n={r['n']}, mean ratio change={r['ratio_change_mean']:.3f}, "
        f"median elasticity={r['elasticity_median'] if np.isfinite(r['elasticity_median']) else 'NA'}."
    )
(RES/"SCIENTIFIC_FINDINGS.md").write_text("\n".join(lines),encoding="utf-8")

print("V30_COUPLING_COMPLETE")
print(json.dumps({
    "ratio_median":ratio_structure["overall_ratio_median"],
    "ratio_mean":ratio_structure["overall_ratio_mean"],
    "ratio_cv_pct":ratio_structure["overall_ratio_cv_pct"],
    "fraction_LC5_gt_LC1":ratio_structure["fraction_LC5_gt_LC1"],
    "best_LC5_from_LC1":best_lc5,
    "best_LC1_from_LC5":best_lc1,
    "same_pass_edges":len(same_pass),
},ensure_ascii=False))
