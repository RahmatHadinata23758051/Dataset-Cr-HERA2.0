"""
visualize_results_v2.py  --  HERA 2.0  Phase 2.5  Area 4
==========================================================
Publication-grade visualization suite for fine-tuned v2 models.

8 figures:
  01  Before/After Improvement Dashboard   (v1 vs v2)
  02  Density-Coloured Parity Plots        (v2 best models)
  03  Residual Diagnostic Analysis          (v2)
  04  9-Feature Permutation Importance      (v2)
  05  Multi-Algorithm Performance Dashboard (v2)
  06  WHO Compliance Confusion Matrix Grid  (v2)
  07  Learning Curves                       (v2, overfitting proof)
  08  5-Fold CV Stability                   (violin + scatter)
"""

import os, sys, pickle, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, gaussian_kde
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from xgboost import XGBRegressor

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10.5,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})
CMAP_CR  = "#0F4C81"   # Chromium blue
CMAP_NI  = "#1B6B35"   # Nickel green
CMAP_V1  = "#95A5A6"   # v1 grey
CMAP_V2C = "#2980B9"   # v2 Chromium
CMAP_V2N = "#27AE60"   # v2 Nickel
WHO_CR, WHO_NI = 50.0, 20.0

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset",
                                             "dataset_heavy_metal_grounded_v2.csv"))
models_dir   = os.path.abspath(os.path.join(script_dir, "..", "models"))
report_dir   = os.path.abspath(os.path.join(script_dir, "..", "results", "reports"))
out_dir      = os.path.abspath(os.path.join(script_dir, "..", "results", "images"))
os.makedirs(out_dir, exist_ok=True)

SEED    = 42
N_FOLDS = 5
RAW     = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
DERIVED = ["pH_squared", "pH_EC_interact", "log_EC", "pOH_proxy", "pH_temp_interact"]
FEATS   = RAW + DERIVED
FEAT_LABELS = {
    "pH":               "pH",
    "EC_uScm":          "EC (uS/cm)",
    "TDS_mgL":          "TDS (mg/L)",
    "Suhu_Air":         "Temperature",
    "pH_squared":       "pH^2",
    "pH_EC_interact":   "pH x EC",
    "log_EC":           "log10(EC)",
    "pOH_proxy":        "pOH (14-pH)",
    "pH_temp_interact": "pH x Temp",
}

print("=" * 70)
print("  HERA 2.0  Phase 2.5  Area 4 -- Visualization Suite v2")
print("=" * 70)

# ── Load dataset ───────────────────────────────────────────────────────────
df    = pd.read_csv(dataset_path)
X_raw = df[FEATS].values
y_cr  = df["Chromium_ugL"].values
y_ni  = df["Nickel_ugL"].values
print(f"\n[INFO] Dataset v2: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── Load serialized models ─────────────────────────────────────────────────
def load_pack(fname):
    p = os.path.join(models_dir, fname)
    with open(p, "rb") as f:
        return pickle.load(f)

pack_cr = load_pack("best_model_chromium_v2.pkl")
pack_ni = load_pack("best_model_nickel_v2.pkl")

model_cr  = pack_cr["model"]
scaler_cr = pack_cr["scaler"]
model_ni  = pack_ni["model"]
scaler_ni = pack_ni["scaler"]
print(f"[INFO] Best Chromium model : {pack_cr['best_model']}  (Test R2={pack_cr['test_r2']:.5f})")
print(f"[INFO] Best Nickel model   : {pack_ni['best_model']}  (Test R2={pack_ni['test_r2']:.5f})")

# ── Helper: prepare splits ─────────────────────────────────────────────────
Xtr_cr, Xte_cr, ytr_cr, yte_cr = train_test_split(X_raw, y_cr, test_size=0.2, random_state=SEED)
Xtr_ni, Xte_ni, ytr_ni, yte_ni = train_test_split(X_raw, y_ni, test_size=0.2, random_state=SEED)

Xtr_cr_s  = scaler_cr.transform(Xtr_cr)
Xte_cr_s  = scaler_cr.transform(Xte_cr)
Xtr_ni_s  = scaler_ni.transform(Xtr_ni)
Xte_ni_s  = scaler_ni.transform(Xte_ni)
X_full_cr = scaler_cr.transform(X_raw)
X_full_ni = scaler_ni.transform(X_raw)

yp_te_cr = model_cr.predict(Xte_cr_s)
yp_te_ni = model_ni.predict(Xte_ni_s)
res_cr   = yte_cr - yp_te_cr
res_ni   = yte_ni - yp_te_ni

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 -- Before/After Improvement Dashboard
# ══════════════════════════════════════════════════════════════════════════════
print("\n[PLOT 1] Before/After Improvement Dashboard...")

V1 = {
    ("Chromium","Random Forest"):     {"r2":0.9587,"gap":2.50},
    ("Chromium","XGBoost Regressor"): {"r2":0.9650,"gap":1.75},
    ("Nickel",  "Random Forest"):     {"r2":0.9446,"gap":2.50},
    ("Nickel",  "XGBoost Regressor"): {"r2":0.9479,"gap":3.21},
}
V2 = {
    ("Chromium","Random Forest"):     {"r2":0.99144,"gap":0.23},
    ("Chromium","XGBoost Regressor"): {"r2":0.99194,"gap":0.12},
    ("Nickel",  "Random Forest"):     {"r2":0.98959,"gap":0.43},
    ("Nickel",  "XGBoost Regressor"): {"r2":0.98874,"gap":0.42},
}
keys    = list(V1.keys())
labels  = [f"{m}\n{mdl.replace(' Regressor','')}" for m,mdl in keys]
v1_r2   = [V1[k]["r2"]  for k in keys]
v2_r2   = [V2[k]["r2"]  for k in keys]
v1_gap  = [V1[k]["gap"] for k in keys]
v2_gap  = [V2[k]["gap"] for k in keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
x = np.arange(len(keys))
w = 0.32

# R2 bars
b1 = ax1.bar(x - w/2, v1_r2, w, color=CMAP_V1, label="Phase 2 (v1)", alpha=0.85,
             edgecolor="white", linewidth=0.5)
b2 = ax1.bar(x + w/2, v2_r2, w, color=[CMAP_V2C,CMAP_V2C,CMAP_V2N,CMAP_V2N],
             label="Phase 2.5 (v2)", alpha=0.85, edgecolor="white", linewidth=0.5)
ax1.set_ylim(0.90, 1.003)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Test R2", weight="semibold")
ax1.set_title("Test R2: Phase 2 vs Phase 2.5", weight="bold")
ax1.legend(fontsize=9)
ax1.axhline(0.95, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.7, label="Target R2=0.95")
for bar, v1v, v2v in zip(b2, v1_r2, v2_r2):
    delta = (v2v - v1v) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f"+{delta:.2f}%", ha="center", va="bottom", fontsize=8,
             color="#1A5276", weight="bold")

# Gap bars
cols_gap = ["#E74C3C" if g > 2.0 else "#F39C12" if g > 1.0 else "#27AE60" for g in v1_gap]
b3 = ax2.bar(x - w/2, v1_gap, w, color=cols_gap, label="Phase 2 (v1)",
             alpha=0.85, edgecolor="white", linewidth=0.5)
b4 = ax2.bar(x + w/2, v2_gap, w, color="#27AE60", label="Phase 2.5 (v2)",
             alpha=0.85, edgecolor="white", linewidth=0.5)
ax2.axhline(2.0, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.8)
ax2.text(len(keys)-0.5, 2.05, "Caution threshold (2%)",
         color="#E74C3C", fontsize=8.5, va="bottom")
ax2.set_ylim(0, 4.2)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Train-Test R2 Gap (%)", weight="semibold")
ax2.set_title("Overfitting Gap: Phase 2 vs Phase 2.5", weight="bold")
ax2.legend(fontsize=9)
for bar, v1g, v2g in zip(b4, v1_gap, v2_gap):
    delta = v2g - v1g
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{delta:+.2f}%", ha="center", va="bottom", fontsize=8,
             color="#1A5276", weight="bold")

plt.suptitle(
    "HERA 2.0  Phase 2.5 -- Model Improvement Summary\n"
    "Feature Engineering + Dataset v2 (15K) + Optuna Hyperparameter Tuning",
    weight="bold", size=13, y=1.02)
plt.tight_layout()
p = os.path.join(out_dir, "Phase2.5_01_improvement_comparison.png")
plt.savefig(p); plt.close()
print(f"  [OK] Saved -> {p}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 -- Density-Coloured Parity Plots (v2 best models)
# ══════════════════════════════════════════════════════════════════════════════
print("[PLOT 2] Parity Plots v2...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
configs = [
    (yte_cr, yp_te_cr, CMAP_V2C, "Chromium", pack_cr["best_model"], WHO_CR),
    (yte_ni, yp_te_ni, CMAP_V2N, "Nickel",   pack_ni["best_model"], WHO_NI),
]
for ax, (yt, yp, col, metal, mname, who) in zip(axes, configs):
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    xy   = np.vstack([yt, yp])
    try:
        density = gaussian_kde(xy)(xy)
        density = (density - density.min()) / (density.max() - density.min() + 1e-9)
    except Exception:
        density = np.ones(len(yt)) * 0.5
    idx = np.argsort(density)
    sc = ax.scatter(yt[idx], yp[idx], c=density[idx], cmap="plasma",
                    s=12, alpha=0.6, edgecolors="none")
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "k-",  lw=1.5, label="1:1 line")
    sigma = np.std(yt - yp)
    ax.fill_between([lo,hi],[lo-2*sigma,hi-2*sigma],[lo+2*sigma,hi+2*sigma],
                    alpha=0.10, color=col, label=f"+-2 sigma band")
    ax.set_xlabel(f"Measured {metal} (ug/L)", weight="semibold")
    ax.set_ylabel(f"Predicted {metal} (ug/L)", weight="semibold")
    ax.set_title(f"{metal}  |  {mname}\nR2={r2:.5f}   RMSE={rmse:.3f} ug/L", weight="bold")
    ax.text(0.04, 0.95, f"n = {len(yt):,} (test set)",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.75, label="Point density")

plt.suptitle("Phase 2.5 v2 -- Density-Coloured Parity Plots (Holdout Test Set)",
             weight="bold", size=13, y=1.02)
plt.tight_layout()
p = os.path.join(out_dir, "Phase2.5_02_parity_plots_v2.png")
plt.savefig(p); plt.close()
print(f"  [OK] Saved -> {p}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 -- Residual Analysis v2
# ══════════════════════════════════════════════════════════════════════════════
print("[PLOT 3] Residual Analysis v2...")

from scipy.stats import norm as sp_norm
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
configs2 = [
    (yp_te_cr, res_cr, CMAP_V2C, "Chromium"),
    (yp_te_ni, res_ni, CMAP_V2N, "Nickel"),
]
for row, (yp, res, col, metal) in enumerate(configs2):
    # residuals vs predicted
    ax = axes[row, 0]
    ax.scatter(yp, res, color=col, alpha=0.25, s=9, edgecolors="none")
    ax.axhline(0, color="black", linewidth=1.2)
    window = max(len(res)//30, 1)
    idx_sort = np.argsort(yp)
    running_mean = np.convolve(res[idx_sort], np.ones(window)/window, mode="valid")
    x_run = yp[idx_sort][window//2: window//2 + len(running_mean)]
    ax.plot(x_run, running_mean, color="#E74C3C", linewidth=1.8, label="Running mean")
    ax.set_xlabel(f"Predicted {metal} (ug/L)", weight="semibold")
    ax.set_ylabel("Residual (ug/L)", weight="semibold")
    ax.set_title(f"{metal} -- Residuals vs Predicted", weight="bold")
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.97,
            f"Bias = {res.mean():.3f}\nRMSE = {np.sqrt((res**2).mean()):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"))
    # distribution
    ax = axes[row, 1]
    ax.hist(res, bins=50, color=col, edgecolor="white", linewidth=0.3,
            alpha=0.8, density=True, label="Residuals")
    mu, sd = res.mean(), res.std()
    xg = np.linspace(res.min(), res.max(), 300)
    ax.plot(xg, sp_norm.pdf(xg, mu, sd), "k-", linewidth=2,
            label=f"N(mu={mu:.2f}, sigma={sd:.2f})")
    ax.set_xlabel("Residual (ug/L)", weight="semibold")
    ax.set_ylabel("Density", weight="semibold")
    ax.set_title(f"{metal} -- Residual Distribution", weight="bold")
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

plt.suptitle("Phase 2.5 v2 -- Residual Diagnostic Analysis",
             weight="bold", size=13, y=1.01)
plt.tight_layout()
p = os.path.join(out_dir, "Phase2.5_03_residual_analysis_v2.png")
plt.savefig(p); plt.close()
print(f"  [OK] Saved -> {p}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 -- 9-Feature Permutation Importance
# ══════════════════════════════════════════════════════════════════════════════
print("[PLOT 4] 9-Feature Permutation Importance (this takes ~1-2 min)...")

pi_cr = permutation_importance(model_cr, Xte_cr_s, yte_cr,
                               n_repeats=10, random_state=SEED, n_jobs=-1)
pi_ni = permutation_importance(model_ni, Xte_ni_s, yte_ni,
                               n_repeats=10, random_state=SEED, n_jobs=-1)

feat_labels_short = [FEAT_LABELS[f] for f in FEATS]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pi, col, metal, who in [
        (axes[0], pi_cr, CMAP_V2C, "Chromium", WHO_CR),
        (axes[1], pi_ni, CMAP_V2N, "Nickel",   WHO_NI)]:
    means = pi.importances_mean
    stds  = pi.importances_std
    idx   = np.argsort(means)
    labels_ord = [feat_labels_short[i] for i in idx]
    bars = ax.barh(labels_ord, means[idx], xerr=stds[idx],
                   color=col, alpha=0.80, edgecolor="white",
                   error_kw=dict(elinewidth=1.2, capsize=3, ecolor="#555"))
    # Highlight derived features
    for i, feat_idx in enumerate(idx):
        if FEATS[feat_idx] in DERIVED:
            bars[i].set_edgecolor("#E74C3C")
            bars[i].set_linewidth(2.0)
    ax.set_xlabel("Permutation Importance (Mean DeltaR2)", weight="semibold")
    ax.set_title(f"{metal} -- {pack_cr['best_model'] if metal=='Chromium' else pack_ni['best_model']}\n"
                 f"Permutation Importance (10 repeats, test set)",
                 weight="bold")
    ax.grid(True, alpha=0.35, axis="x")
    # Legend
    red_patch   = mpatches.Patch(edgecolor="#E74C3C", facecolor=col,
                                  linewidth=2, label="Derived feature (red border)")
    plain_patch = mpatches.Patch(facecolor=col, label="Raw feature")
    ax.legend(handles=[red_patch, plain_patch], fontsize=8.5, loc="lower right")

plt.suptitle("Phase 2.5 v2 -- 9-Feature Permutation Importance (Mean +/- 1SD)\n"
             "Red border = physics-informed derived feature",
             weight="bold", size=13, y=1.01)
plt.tight_layout()
p = os.path.join(out_dir, "Phase2.5_04_feature_importance_9feat.png")
plt.savefig(p); plt.close()
print(f"  [OK] Saved -> {p}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 -- Multi-Algorithm Performance Dashboard (v2)
# ══════════════════════════════════════════════════════════════════════════════
print("[PLOT 5] Multi-Algorithm Performance Dashboard...")

bench_path = os.path.join(report_dir, "benchmark_v2.csv")
bdf = pd.read_csv(bench_path)
models_list = ["Linear Regression","Ridge Regression","SVR (RBF Kernel)",
               "Random Forest","XGBoost Regressor"]
colors_models = ["#7F8C8D","#95A5A6","#2980B9","#27AE60","#E67E22"]

fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
metrics = [("Test R2","Test R2"),("Test RMSE","Test RMSE"),
           ("Test MAPE (%)","Test MAPE (%)")]

for col_i, (metric_key, metric_label) in enumerate(metrics):
    for row_i, metal in enumerate(["Chromium","Nickel"]):
        ax  = fig.add_subplot(gs[row_i, col_i])
        sub = bdf[bdf["Metal"] == metal]
        vals = [float(sub[sub["Model"]==m][metric_key].values[0]) for m in models_list]
        best_val = max(vals) if "R2" in metric_key else min(vals)
        bar_cols  = [("#E74C3C" if v == best_val else c)
                     for v, c in zip(vals, colors_models)]
        bars = ax.bar(range(len(models_list)), vals, color=bar_cols,
                      edgecolor="white", linewidth=0.5, alpha=0.88)
        ax.set_xticks(range(len(models_list)))
        ax.set_xticklabels(["LinReg","Ridge","SVR","RF","XGB"], fontsize=9)
        ax.set_title(f"{metal} -- {metric_label}", weight="bold", fontsize=10)
        ax.grid(True, alpha=0.35, axis="y")
        if "R2" in metric_key:
            ax.set_ylim(max(0, min(vals)*0.97), 1.005)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f"{v:.4f}" if "R2" in metric_key else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90 if "MAPE" in metric_key else 0)

plt.suptitle("Phase 2.5 v2 -- Multi-Algorithm Performance Dashboard\n"
             "Red bar = best model per metric; all models trained with Optuna best params",
             weight="bold", size=13, y=1.01)
p = os.path.join(out_dir, "Phase2.5_05_model_comparison_dashboard.png")
plt.savefig(p); plt.close()
print(f"  [OK] Saved -> {p}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 -- WHO Compliance Confusion Matrix Grid (v2)
# ══════════════════════════════════════════════════════════════════════════════
print("[PLOT 6] Confusion Matrix Grid v2...")

def rebuild_models(metal):
    """Rebuild all 5 models for confusion matrix generation using v2 best params."""
    import json
    params_path = os.path.join(report_dir, "best_params.json")
    with open(params_path) as f:
        bp = json.load(f)
    rf_p  = bp[f"Random_Forest_{metal}"]["best_params"]
    xgb_p = bp[f"XGBoost_{metal}"]["best_params"]
    return {
        "Linear\nRegression": LinearRegression(),
        "Ridge\nRegression":  Ridge(alpha=10.0),
        "SVR\n(RBF)":         SVR(C=20.0, epsilon=0.1, kernel="rbf"),
        "Random\nForest":     RandomForestRegressor(**rf_p, random_state=SEED, n_jobs=-1),
        "XGBoost":            XGBRegressor(**xgb_p, random_state=SEED, n_jobs=-1, verbosity=0),
    }

fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
metal_configs = [
    ("Chromium", Xtr_cr_s, Xte_cr_s, ytr_cr, yte_cr, WHO_CR, CMAP_V2C),
    ("Nickel",   Xtr_ni_s, Xte_ni_s, ytr_ni, yte_ni, WHO_NI, CMAP_V2N),
]

for row, (metal, Xtr, Xte, ytr, yte, who, col) in enumerate(metal_configs):
    mdls = rebuild_models(metal)
    for col_i, (mname, mdl) in enumerate(mdls.items()):
        ax = axes[row, col_i]
        mdl.fit(Xtr, ytr)
        yp = mdl.predict(Xte)
        true_cls = (yte > who).astype(int)
        pred_cls = (yp  > who).astype(int)
        cm = confusion_matrix(true_cls, pred_cls)
        acc = (cm[0,0]+cm[1,1])/cm.sum()*100
        try:
            prec = cm[1,1]/(cm[0,1]+cm[1,1]+1e-9)*100
            rec  = cm[1,1]/(cm[1,0]+cm[1,1]+1e-9)*100
            f1   = 2*prec*rec/(prec+rec+1e-9)
        except Exception:
            prec=rec=f1=0.0
        cmap_used = sns.light_palette(col, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_used, ax=ax,
                    cbar=False, linewidths=2, linecolor="white",
                    annot_kws={"size":11,"weight":"bold"})
        ax.set_xticklabels(["Safe","Unsafe"], fontsize=8.5)
        ax.set_yticklabels(["Safe","Unsafe"], fontsize=8.5, rotation=0)
        ax.set_xlabel("Predicted", fontsize=8.5)
        if col_i == 0:
            ax.set_ylabel(f"{metal}\n(WHO>{who:.0f} ug/L)\nActual", fontsize=9, weight="bold")
        ax.set_title(f"{mname}\nAcc={acc:.1f}% F1={f1:.1f}%", fontsize=8.5, weight="bold")

plt.suptitle("Phase 2.5 v2 -- WHO Compliance Confusion Matrix Grid\n"
             "Binary classification: Safe (<= WHO limit) vs Unsafe (> WHO limit)",
             weight="bold", size=13, y=1.01)
plt.tight_layout()
p = os.path.join(out_dir, "Phase2.5_06_confusion_matrices_v2.png")
plt.savefig(p); plt.close()
print(f"  [OK] Saved -> {p}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 -- Learning Curves v2 (Overfitting Proof)
# ══════════════════════════════════════════════════════════════════════════════
print("[PLOT 7] Learning Curves v2 (takes ~2-3 min)...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
lc_configs = [
    ("Chromium", X_full_cr, y_cr, CMAP_V2C),
    ("Nickel",   X_full_ni, y_ni, CMAP_V2N),
]
lc_models = {
    "SVR (RBF)":    [None, None],
    "Random Forest":[None, None],
    "XGBoost":      [None, None],
}

def get_lc_model(name, metal):
    import json
    params_path = os.path.join(report_dir, "best_params.json")
    with open(params_path) as f:
        bp = json.load(f)
    if name == "SVR (RBF)":
        return SVR(C=20.0, epsilon=0.1, kernel="rbf")
    elif name == "Random Forest":
        p = bp[f"Random_Forest_{metal}"]["best_params"]
        return RandomForestRegressor(**p, random_state=SEED, n_jobs=-1)
    else:
        p = bp[f"XGBoost_{metal}"]["best_params"]
        return XGBRegressor(**p, random_state=SEED, n_jobs=-1, verbosity=0)

train_sizes = np.linspace(0.10, 1.0, 8)

for row, (metal, X_full, y_full, col) in enumerate(lc_configs):
    for col_i, mname in enumerate(["SVR (RBF)", "Random Forest", "XGBoost"]):
        ax  = axes[row, col_i]
        mdl = get_lc_model(mname, metal)
        ts, tr_sc, cv_sc = learning_curve(
            mdl, X_full, y_full,
            train_sizes=train_sizes,
            cv=kf, scoring="r2",
            n_jobs=-1, shuffle=True, random_state=SEED)
        tr_mean = tr_sc.mean(axis=1)
        tr_std  = tr_sc.std(axis=1)
        cv_mean = cv_sc.mean(axis=1)
        cv_std  = cv_sc.std(axis=1)
        ax.plot(ts, tr_mean, "o-", color=col,     lw=2,   label="Training R2")
        ax.plot(ts, cv_mean, "s--",color="#2C3E50",lw=2,   label="CV R2 (5-Fold)")
        ax.fill_between(ts, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=col)
        ax.fill_between(ts, cv_mean-cv_std, cv_mean+cv_std, alpha=0.15, color="#2C3E50")
        final_gap = (tr_mean[-1] - cv_mean[-1]) * 100
        status    = "[OK] Converged" if final_gap < 2.0 else f"Gap={final_gap:.1f}%"
        ax.set_title(f"{metal} -- {mname}\n{status}", weight="bold", fontsize=10,
                     color="#1A5276" if final_gap < 2.0 else "#C0392B")
        ax.set_xlabel("Training Samples", weight="semibold")
        if col_i == 0:
            ax.set_ylabel("R2 Score", weight="semibold")
        ax.set_ylim(0.8, 1.01)
        ax.legend(fontsize=8.5); ax.grid(True, alpha=0.35)

plt.suptitle("Phase 2.5 v2 -- Learning Curves (Overfitting Diagnostic)\n"
             "Convergence of Training and 5-Fold CV curves confirms generalisation",
             weight="bold", size=13, y=1.01)
plt.tight_layout()
p = os.path.join(out_dir, "Phase2.5_07_learning_curves_v2.png")
plt.savefig(p); plt.close()
print(f"  [OK] Saved -> {p}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 -- 5-Fold CV Stability (Violin + Strip)
# ══════════════════════════════════════════════════════════════════════════════
print("[PLOT 8] 5-Fold CV Stability Plot...")

import json
params_path = os.path.join(report_dir, "best_params.json")
with open(params_path) as f:
    bp = json.load(f)

cv_data = []
all_mdl_configs = [
    ("Linear Regression", LinearRegression(),                                           LinearRegression()),
    ("Ridge Regression",  Ridge(alpha=10.0),                                            Ridge(alpha=10.0)),
    ("SVR (RBF)",         SVR(C=20.0, epsilon=0.1, kernel="rbf"),                       SVR(C=20.0, epsilon=0.1, kernel="rbf")),
    ("Random Forest",     RandomForestRegressor(**bp["Random_Forest_Chromium"]["best_params"], random_state=SEED,n_jobs=-1),
                          RandomForestRegressor(**bp["Random_Forest_Nickel"]["best_params"],   random_state=SEED,n_jobs=-1)),
    ("XGBoost",           XGBRegressor(**bp["XGBoost_Chromium"]["best_params"],         random_state=SEED,n_jobs=-1,verbosity=0),
                          XGBRegressor(**bp["XGBoost_Nickel"]["best_params"],           random_state=SEED,n_jobs=-1,verbosity=0)),
]
for mname, m_cr, m_ni in all_mdl_configs:
    sc_cr = cross_val_score(m_cr, X_full_cr, y_cr, cv=kf, scoring="r2", n_jobs=-1)
    sc_ni = cross_val_score(m_ni, X_full_ni, y_ni, cv=kf, scoring="r2", n_jobs=-1)
    for s in sc_cr:
        cv_data.append({"Model": mname, "Metal": "Chromium", "CV R2": s})
    for s in sc_ni:
        cv_data.append({"Model": mname, "Metal": "Nickel",   "CV R2": s})

cv_df = pd.DataFrame(cv_data)
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
for ax, metal, col in [(axes[0],"Chromium",CMAP_V2C),(axes[1],"Nickel",CMAP_V2N)]:
    sub = cv_df[cv_df["Metal"]==metal]
    sns.violinplot(data=sub, x="Model", y="CV R2", ax=ax,
                   color=col, alpha=0.55, inner=None, linewidth=1.2)
    sns.stripplot(data=sub, x="Model", y="CV R2", ax=ax,
                  color="white", edgecolor=col, linewidth=1.2,
                  size=7, alpha=0.9, jitter=False)
    means = sub.groupby("Model")["CV R2"].mean()
    ax.set_xticklabels(["LinReg","Ridge","SVR","RF","XGB"], fontsize=9.5)
    ax.set_title(f"{metal} -- 5-Fold CV R2 Distribution\n(Each dot = one fold)", weight="bold")
    ax.set_xlabel(""); ax.set_ylabel("CV R2 Score", weight="semibold")
    ax.grid(True, alpha=0.35, axis="y")
    for i, (mname, mean_val) in enumerate(means.items()):
        ax.text(i, mean_val - 0.005, f"{mean_val:.5f}",
                ha="center", va="top", fontsize=7.5, color="#1A1A1A", weight="bold")

plt.suptitle("Phase 2.5 v2 -- 5-Fold Cross-Validation Stability\n"
             "Tight violin width = highly stable model (low variance across folds)",
             weight="bold", size=13, y=1.01)
plt.tight_layout()
p = os.path.join(out_dir, "Phase2.5_08_cv_fold_stability.png")
plt.savefig(p); plt.close()
print(f"  [OK] Saved -> {p}")

# ── Final Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  [SUCCESS] Area 4 complete! All 8 publication-grade plots generated.")
print(f"  Location: {out_dir}")
print("=" * 70)
print("\n  Plots generated:")
for i, name in enumerate([
    "01_improvement_comparison",
    "02_parity_plots_v2",
    "03_residual_analysis_v2",
    "04_feature_importance_9feat",
    "05_model_comparison_dashboard",
    "06_confusion_matrices_v2",
    "07_learning_curves_v2",
    "08_cv_fold_stability",
], start=1):
    print(f"    Phase2.5_{name}.png")
