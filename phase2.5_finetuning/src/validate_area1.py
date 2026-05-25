"""
validate_area1.py
Validasi visual Area 1: Feature Engineering & Dataset Regeneration (v1 vs v2)

Menghasilkan 4 figure publikasi-grade:
  V1  - pH Distribution Comparison (v1 vs v2) - Histogram
  V2  - Derived Feature Spearman Correlation Heatmap
  V3  - Derived Features vs Targets Scatter Grid
  V4  - Feature Distribution Comparison v1 vs v2 (raw inputs)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset"))
out_dir     = os.path.abspath(os.path.join(script_dir, "..", "results", "images", "area1_validation"))
os.makedirs(out_dir, exist_ok=True)

path_v1 = os.path.join(dataset_dir, "dataset_heavy_metal_grounded.csv")
path_v2 = os.path.join(dataset_dir, "dataset_heavy_metal_grounded_v2.csv")

print("=" * 65)
print("  HERA 2.0 -- Area 1 Validation Report")
print("=" * 65)

if not os.path.exists(path_v1) or not os.path.exists(path_v2):
    print("[ERROR] Dataset file(s) not found. Run generate_dataset.py first.")
    sys.exit(1)

df1 = pd.read_csv(path_v1)
df2 = pd.read_csv(path_v2)

print(f"\n  v1 dataset : {df1.shape[0]:>6,} rows x {df1.shape[1]} columns")
print(f"  v2 dataset : {df2.shape[0]:>6,} rows x {df2.shape[1]} columns")
print(f"  New features added: {df2.shape[1] - df1.shape[1]} (5 derived + same 4 raw + 2 targets)\n")

RAW      = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
DERIVED  = ["pH_squared", "pH_EC_interact", "log_EC", "pOH_proxy", "pH_temp_interact"]
TARGETS  = ["Chromium_ugL", "Nickel_ugL"]
COLORS   = {"v1": "#6c757d", "v2": "#0F4C81"}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: pH Distribution Comparison — v1 (normal) vs v2 (stratified uniform)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bins = np.linspace(4.7, 8.6, 45)

# v1 histogram
axes[0].hist(df1["pH"], bins=bins, color=COLORS["v1"], edgecolor="white",
             linewidth=0.4, alpha=0.85, label=f"v1 (n={len(df1):,})")
axes[0].set_title("Dataset v1 — pH Distribution\n(Normal sampling, clustered at pH 6.5)", weight="bold")
axes[0].set_xlabel("pH", weight="semibold")
axes[0].set_ylabel("Sample Count", weight="semibold")
axes[0].axvline(df1["pH"].mean(), color="#C0392B", linestyle="--", linewidth=1.5,
                label=f"Mean = {df1['pH'].mean():.2f}")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.4)

# v2 histogram — expect 4 flat plateaus
axes[1].hist(df2["pH"], bins=bins, color=COLORS["v2"], edgecolor="white",
             linewidth=0.4, alpha=0.85, label=f"v2 (n={len(df2):,})")
axes[1].set_title("Dataset v2 — pH Distribution\n(Stratified uniform sampling, 4 equal bands)", weight="bold")
axes[1].set_xlabel("pH", weight="semibold")
axes[1].set_ylabel("Sample Count", weight="semibold")

# Mark stratum boundaries
for boundary in [5.5, 6.5, 7.5]:
    axes[1].axvline(boundary, color="#E74C3C", linestyle=":", linewidth=1.5, alpha=0.8)
axes[1].axvline(df2["pH"].mean(), color="#C0392B", linestyle="--", linewidth=1.5,
                label=f"Mean = {df2['pH'].mean():.2f}")

# Annotate strata
for label, x in [("Acidic", 5.15), ("Mod.\nAcidic", 6.0), ("Near-\nNeutral", 7.0), ("Alkaline", 8.0)]:
    axes[1].text(x, axes[1].get_ylim()[1] * 0.88 if axes[1].get_ylim()[1] > 0 else 200,
                 label, ha="center", fontsize=8.5, color="#1A5276", weight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.4)

plt.suptitle("Area 1 Validation — pH Sampling Strategy: v1 vs v2",
             weight="bold", size=13, y=1.01)
plt.tight_layout()
p1 = os.path.join(out_dir, "V1_pH_distribution_comparison.png")
plt.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[OK] Figure 1 saved -> {p1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Full Spearman Correlation Heatmap — v2 (all 11 columns)
# ══════════════════════════════════════════════════════════════════════════════
all_cols = RAW + DERIVED + TARGETS
corr_v2  = df2[all_cols].corr(method="spearman")

label_map = {
    "pH":               r"$\mathrm{pH}$",
    "EC_uScm":          r"$\mathrm{EC}$",
    "TDS_mgL":          r"$\mathrm{TDS}$",
    "Suhu_Air":         r"$\mathrm{Temp}$",
    "pH_squared":       r"$\mathrm{pH^2}$",
    "pH_EC_interact":   r"$\mathrm{pH \times EC}$",
    "log_EC":           r"$\mathrm{log_{10}EC}$",
    "pOH_proxy":        r"$\mathrm{pOH}$",
    "pH_temp_interact": r"$\mathrm{pH \times T}$",
    "Chromium_ugL":     r"$\mathrm{Cr\ (\mu g/L)}$",
    "Nickel_ugL":       r"$\mathrm{Ni\ (\mu g/L)}$",
}
corr_v2 = corr_v2.rename(index=label_map, columns=label_map)

fig, ax = plt.subplots(figsize=(9, 7.5))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr_v2, dtype=bool), k=1)
sns.heatmap(corr_v2, mask=mask, cmap=cmap, vmin=-1, vmax=1,
            annot=True, fmt=".2f", square=True,
            linewidths=1.5, linecolor="white",
            annot_kws={"size": 8, "weight": "semibold"},
            cbar_kws={"shrink": 0.75, "label": "Spearman rs"},
            ax=ax)
ax.set_title("Spearman Correlation Matrix — v2 Dataset (Raw + Derived Features)",
             weight="bold", pad=12)

# Highlight derived features vs targets region with rectangle
# derived features are rows 4-8, targets are cols 9-10 (lower triangle only shown)
plt.tight_layout()
p2 = os.path.join(out_dir, "V2_spearman_heatmap_v2_full.png")
plt.savefig(p2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[OK] Figure 2 saved -> {p2}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Derived Features vs Targets Scatter (5 features x 2 targets = 10 panels)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(16, 7))

feat_labels = {
    "pH_squared":       r"$\mathrm{pH^2}$",
    "pH_EC_interact":   r"$\mathrm{pH \times EC}$",
    "log_EC":           r"$\mathrm{log_{10}(EC)}$",
    "pOH_proxy":        r"$\mathrm{pOH\ (14 - pH)}$",
    "pH_temp_interact": r"$\mathrm{pH \times T_{water}}$",
}

sample = df2.sample(2000, random_state=42)

for col_i, feat in enumerate(DERIVED):
    for row_i, target in enumerate(TARGETS):
        ax = axes[row_i, col_i]
        color = "#0F4C81" if target == "Chromium_ugL" else "#1E6B38"
        metal = "Cr" if target == "Chromium_ugL" else "Ni"

        rs, _ = spearmanr(sample[feat], sample[target])

        ax.scatter(sample[feat], sample[target],
                   color=color, alpha=0.20, s=8, edgecolors="none")

        ax.set_xlabel(feat_labels[feat], fontsize=9.5, weight="semibold")
        if col_i == 0:
            ax.set_ylabel(rf"$\mathrm{{{metal}\ (\mu g/L)}}$", fontsize=9.5, weight="semibold")
        ax.set_title(rf"$r_s = {rs:+.3f}$", fontsize=9.5, weight="bold",
                     color="#C0392B" if abs(rs) > 0.7 else "#555")
        ax.grid(True, alpha=0.35)

# Row labels
axes[0, 0].set_ylabel(r"$\mathrm{Chromium\ (\mu g/L)}$", fontsize=10, weight="bold")
axes[1, 0].set_ylabel(r"$\mathrm{Nickel\ (\mu g/L)}$", fontsize=10, weight="bold")

plt.suptitle(
    "Area 1 Validation — Derived Features vs. Targets (Spearman rs, n=2,000 sample)\n"
    "Red title = strong correlation (|rs| > 0.70) confirming feature informativeness",
    weight="bold", size=12, y=1.01
)
plt.tight_layout()
p3 = os.path.join(out_dir, "V3_derived_features_vs_targets.png")
plt.savefig(p3, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[OK] Figure 3 saved -> {p3}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Raw Feature Distributions — v1 vs v2 (KDE overlay)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

feat_info = [
    ("pH",       "pH",             axes[0, 0]),
    ("EC_uScm",  "EC (uS/cm)",     axes[0, 1]),
    ("TDS_mgL",  "TDS (mg/L)",     axes[1, 0]),
    ("Suhu_Air", "Temperature (C)", axes[1, 1]),
]

for feat, label, ax in feat_info:
    sns.kdeplot(df1[feat], ax=ax, color=COLORS["v1"], linewidth=2.2, fill=True,
                alpha=0.30, label=f"v1 (n={len(df1):,})")
    sns.kdeplot(df2[feat], ax=ax, color=COLORS["v2"], linewidth=2.2, fill=True,
                alpha=0.30, label=f"v2 (n={len(df2):,})")
    ax.set_xlabel(label, weight="semibold")
    ax.set_ylabel("Density", weight="semibold")
    ax.set_title(f"{label} — v1 vs v2 Distribution", weight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    # Annotate std
    ax.text(0.97, 0.95,
            f"v1 std = {df1[feat].std():.2f}\nv2 std = {df2[feat].std():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

plt.suptitle("Area 1 Validation — Raw Feature KDE: Dataset v1 vs v2",
             weight="bold", size=13, y=1.01)
plt.tight_layout()
p4 = os.path.join(out_dir, "V4_raw_feature_distributions_v1_vs_v2.png")
plt.savefig(p4, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[OK] Figure 4 saved -> {p4}")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  AREA 1 VALIDATION SUMMARY")
print("=" * 65)

print(f"\n{'CHECK':<45} {'RESULT'}")
print("-" * 65)

# Check 1: sample size
ok1 = len(df2) == 15000
print(f"  Sample size = 15,000                           {'PASS' if ok1 else 'FAIL'}")

# Check 2: column count
ok2 = df2.shape[1] == 11
print(f"  Columns = 11 (4 raw + 5 derived + 2 targets)  {'PASS' if ok2 else 'FAIL'}")

# Check 3: stratified pH
bands = [(4.8, 5.5), (5.5, 6.5), (6.5, 7.5), (7.5, 8.5)]
band_counts = [((df2["pH"] >= lo) & (df2["pH"] <= hi)).sum() for lo, hi in bands]
ok3 = all(2900 <= c <= 4000 for c in band_counts)
print(f"  pH stratification balanced (3,750 per band)    {'PASS' if ok3 else 'FAIL'}")
for (lo, hi), cnt in zip(bands, band_counts):
    print(f"      pH [{lo:.1f}-{hi:.1f}]: {cnt:,} samples")

# Check 4: derived features present
derived_present = all(c in df2.columns for c in DERIVED)
print(f"  All 5 derived features present                 {'PASS' if derived_present else 'FAIL'}")

# Check 5: derived feature correlations with targets
print(f"\n  Spearman |rs| of derived features vs targets:")
print(f"  {'Feature':<22} {'vs Chromium':>12} {'vs Nickel':>12} {'Informative?':>14}")
print("  " + "-" * 62)
all_informative = True
for feat in DERIVED:
    rs_cr = abs(spearmanr(df2[feat], df2["Chromium_ugL"])[0])
    rs_ni = abs(spearmanr(df2[feat], df2["Nickel_ugL"])[0])
    info  = rs_cr > 0.3 or rs_ni > 0.3
    if not info:
        all_informative = False
    print(f"  {feat:<22} {rs_cr:>12.3f} {rs_ni:>12.3f} {'YES' if info else 'NO - WEAK':>14}")

print(f"\n  All derived features informative (|rs|>0.3)    {'PASS' if all_informative else 'FAIL'}")

# Check 6: v1 intact
ok6 = os.path.exists(path_v1) and pd.read_csv(path_v1).shape[0] == 5000
print(f"  v1 dataset unchanged (5,000 rows)              {'PASS' if ok6 else 'FAIL'}")

# Check 7: no NaN or inf
ok7 = df2.isnull().sum().sum() == 0 and np.isfinite(df2.values).all()
print(f"  No NaN or Inf values in v2 dataset             {'PASS' if ok7 else 'FAIL'}")

all_pass = all([ok1, ok2, ok3, derived_present, all_informative, ok6, ok7])
print("\n" + "=" * 65)
print(f"  OVERALL AREA 1 STATUS: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
print("=" * 65)
print(f"\n  Validation figures saved to:\n  {out_dir}")
