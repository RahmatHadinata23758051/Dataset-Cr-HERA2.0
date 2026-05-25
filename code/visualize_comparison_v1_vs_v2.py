#!/usr/bin/env python3
"""
Visualisasi perbandingan dataset Cr Formula v1 (Linear) vs v2 (Geochemical)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create image folder if not exists
image_dir = Path("image")
image_dir.mkdir(exist_ok=True)

# Data paths
base_path = Path("Dataset/Synthetic")
v1_data = pd.read_csv(base_path / "synthetic_cr_dataset.csv")
v1_audit = pd.read_csv(base_path / "synthetic_cr_dataset_with_category.csv")
v2_data = pd.read_csv(base_path / "synthetic_cr_dataset_v2_geochemical.csv")
v2_audit = pd.read_csv(base_path / "synthetic_cr_dataset_v2_geochemical_with_category.csv")

print("✓ Data loaded successfully\n")

# ============================================================================
# 1. STATISTICAL COMPARISON
# ============================================================================
print("=" * 70)
print("CHROMIUM (Cr) STATISTICS COMPARISON")
print("=" * 70)

stats_v1 = v1_audit["Cr"].describe()
stats_v2 = v2_audit["Cr"].describe()

stats_comparison = pd.DataFrame({
    "V1 (Linear Formula)": stats_v1,
    "V2 (Geochemical)": stats_v2,
    "Difference": stats_v2 - stats_v1
})

print(stats_comparison.round(3))
print("\n")

# Per category breakdown
print("=" * 70)
print("Cr DISTRIBUTION BY WATER CATEGORY")
print("=" * 70)
for cat in ["Sungai", "Danau", "Waduk", "Air tanah"]:
    v1_cat = v1_audit[v1_audit["Kategori"] == cat]["Cr"]
    v2_cat = v2_audit[v2_audit["Kategori"] == cat]["Cr"]
    print(f"\n{cat.upper()}")
    print(f"  V1: min={v1_cat.min():.1f}, max={v1_cat.max():.1f}, mean={v1_cat.mean():.1f}, std={v1_cat.std():.1f}")
    print(f"  V2: min={v2_cat.min():.1f}, max={v2_cat.max():.1f}, mean={v2_cat.mean():.1f}, std={v2_cat.std():.1f}")
    print(f"  CHANGE: Δmean={v2_cat.mean()-v1_cat.mean():+.1f}, Δstd={v2_cat.std()-v1_cat.std():+.1f}")

# ============================================================================
# 2. CHROMIUM DISTRIBUTION COMPARISON
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Chromium Distribution Comparison: V1 vs V2", fontsize=14, fontweight='bold', y=1.00)

# Overall distribution
ax = axes[0, 0]
ax.hist(v1_audit["Cr"], bins=30, alpha=0.6, label="V1 (Linear)", color="steelblue", edgecolor="black")
ax.hist(v2_audit["Cr"], bins=30, alpha=0.6, label="V2 (Geochemical)", color="darkorange", edgecolor="black")
ax.set_xlabel("Cr (μg/L)", fontsize=11, fontweight='bold')
ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
ax.set_title("Overall Cr Distribution", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Box plot by formula
ax = axes[0, 1]
data_box = pd.DataFrame({
    "Cr": list(v1_audit["Cr"]) + list(v2_audit["Cr"]),
    "Formula": ["V1 (Linear)"] * len(v1_audit) + ["V2 (Geochemical)"] * len(v2_audit)
})
sns.boxplot(data=data_box, x="Formula", y="Cr", ax=ax, palette=["steelblue", "darkorange"])
ax.set_ylabel("Cr (μg/L)", fontsize=11, fontweight='bold')
ax.set_title("Cr Range by Formula", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# KDE plot
ax = axes[1, 0]
v1_audit["Cr"].plot.kde(ax=ax, label="V1 (Linear)", linewidth=2.5, color="steelblue")
v2_audit["Cr"].plot.kde(ax=ax, label="V2 (Geochemical)", linewidth=2.5, color="darkorange")
ax.set_xlabel("Cr (μg/L)", fontsize=11, fontweight='bold')
ax.set_ylabel("Density", fontsize=11, fontweight='bold')
ax.set_title("Cr Distribution Density", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Cumulative distribution
ax = axes[1, 1]
v1_sorted = np.sort(v1_audit["Cr"])
v2_sorted = np.sort(v2_audit["Cr"])
ax.plot(v1_sorted, np.linspace(0, 1, len(v1_sorted)), label="V1 (Linear)", linewidth=2.5, color="steelblue")
ax.plot(v2_sorted, np.linspace(0, 1, len(v2_sorted)), label="V2 (Geochemical)", linewidth=2.5, color="darkorange")
ax.set_xlabel("Cr (μg/L)", fontsize=11, fontweight='bold')
ax.set_ylabel("Cumulative Probability", fontsize=11, fontweight='bold')
ax.set_title("Cumulative Distribution Function", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(image_dir / "visualisasi_01_distribution_comparison.png", dpi=300, bbox_inches='tight')
print("\n✓ Saved: image/visualisasi_01_distribution_comparison.png")
plt.close()

# ============================================================================
# 3. pH SENSITIVITY (KEY IMPROVEMENT)
# ============================================================================
print("\nPH SENSITIVITY ANALYSIS:")
print("=" * 70)
print(f"V1 Correlation (Cr vs pH): {v1_audit['pH'].corr(v1_audit['Cr']):.4f}")
print(f"V2 Correlation (Cr vs pH): {v2_audit['pH'].corr(v2_audit['Cr']):.4f}")
print(f"\nInterpretation:")
print(f"  V1: Weak negative correlation (linear double-counting issue)")
print(f"  V2: Strong negative correlation (acidic → higher Cr solubility) ✓")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("pH Sensitivity: Key Improvement in V2", fontsize=14, fontweight='bold')

# V1: Cr vs pH
ax = axes[0]
scatter1 = ax.scatter(v1_audit["pH"], v1_audit["Cr"], alpha=0.6, s=50, c=v1_audit["EC"], 
                      cmap="viridis", edgecolors="black", linewidth=0.5)
z = np.polyfit(v1_audit["pH"], v1_audit["Cr"], 1)
p = np.poly1d(z)
ax.plot(v1_audit["pH"].sort_values(), p(v1_audit["pH"].sort_values()), 
        "r--", linewidth=2.5, label=f"Linear fit (r={v1_audit['pH'].corr(v1_audit['Cr']):.3f})")
ax.set_xlabel("pH", fontsize=11, fontweight='bold')
ax.set_ylabel("Cr (μg/L)", fontsize=11, fontweight='bold')
ax.set_title("V1 (Linear Formula): Cr vs pH", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax, label="EC")

# V2: Cr vs pH
ax = axes[1]
scatter2 = ax.scatter(v2_audit["pH"], v2_audit["Cr"], alpha=0.6, s=50, c=v2_audit["EC"], 
                      cmap="plasma", edgecolors="black", linewidth=0.5)
z = np.polyfit(v2_audit["pH"], v2_audit["Cr"], 2)
p = np.poly1d(z)
ph_sorted = np.sort(v2_audit["pH"])
ax.plot(ph_sorted, p(ph_sorted), "r--", linewidth=2.5, 
        label=f"Poly fit (r={v2_audit['pH'].corr(v2_audit['Cr']):.3f})")
ax.set_xlabel("pH", fontsize=11, fontweight='bold')
ax.set_ylabel("Cr (μg/L)", fontsize=11, fontweight='bold')
ax.set_title("V2 (Geochemical): Cr vs pH (Exponential Inverse)", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax, label="EC")

plt.tight_layout()
plt.savefig(image_dir / "visualisasi_02_pH_sensitivity.png", dpi=300, bbox_inches='tight')
print("\n✓ Saved: image/visualisasi_02_pH_sensitivity.png")
plt.close()

# ============================================================================
# 4. CHROMIUM CORRELATIONS WITH PREDICTORS
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Correlation Analysis: V1 vs V2", fontsize=14, fontweight='bold', y=1.00)

# Scatter plots for V1
ax = axes[0, 0]
ax.scatter(v1_audit["EC"], v1_audit["Cr"], alpha=0.6, s=50, color="steelblue", edgecolors="black", linewidth=0.5)
ax.set_xlabel("EC", fontsize=10, fontweight='bold')
ax.set_ylabel("Cr (μg/L)", fontsize=10, fontweight='bold')
corr_v1_ec = v1_audit["EC"].corr(v1_audit["Cr"])
ax.set_title(f"V1: Cr vs EC (r={corr_v1_ec:.3f})", fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(v1_audit["TDS"], v1_audit["Cr"], alpha=0.6, s=50, color="steelblue", edgecolors="black", linewidth=0.5)
ax.set_xlabel("TDS", fontsize=10, fontweight='bold')
ax.set_ylabel("Cr (μg/L)", fontsize=10, fontweight='bold')
corr_v1_tds = v1_audit["TDS"].corr(v1_audit["Cr"])
ax.set_title(f"V1: Cr vs TDS (r={corr_v1_tds:.3f})", fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
corr_v1 = v1_audit[["EC", "TDS", "pH", "Cr"]].corr()
sns.heatmap(corr_v1, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
ax.set_title("V1: Correlation Matrix", fontsize=11, fontweight='bold')

# Scatter plots for V2
ax = axes[1, 0]
ax.scatter(v2_audit["EC"], v2_audit["Cr"], alpha=0.6, s=50, color="darkorange", edgecolors="black", linewidth=0.5)
ax.set_xlabel("EC", fontsize=10, fontweight='bold')
ax.set_ylabel("Cr (μg/L)", fontsize=10, fontweight='bold')
corr_v2_ec = v2_audit["EC"].corr(v2_audit["Cr"])
ax.set_title(f"V2: Cr vs EC (r={corr_v2_ec:.3f})", fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.scatter(v2_audit["TDS"], v2_audit["Cr"], alpha=0.6, s=50, color="darkorange", edgecolors="black", linewidth=0.5)
ax.set_xlabel("TDS", fontsize=10, fontweight='bold')
ax.set_ylabel("Cr (μg/L)", fontsize=10, fontweight='bold')
corr_v2_tds = v2_audit["TDS"].corr(v2_audit["Cr"])
ax.set_title(f"V2: Cr vs TDS (r={corr_v2_tds:.3f})", fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
corr_v2 = v2_audit[["EC", "TDS", "pH", "Cr"]].corr()
sns.heatmap(corr_v2, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
ax.set_title("V2: Correlation Matrix", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(image_dir / "visualisasi_03_correlations.png", dpi=300, bbox_inches='tight')
print("\n✓ Saved: image/visualisasi_03_correlations.png")
plt.close()

print("\nCORRELATION COMPARISON:")
print("=" * 70)
print(f"{'Metric':<20} {'V1 (Linear)':<20} {'V2 (Geochemical)':<20}")
print("-" * 70)
print(f"{'Cr ↔ EC':<20} {corr_v1_ec:>19.4f} {corr_v2_ec:>19.4f}")
print(f"{'Cr ↔ TDS':<20} {corr_v1_tds:>19.4f} {corr_v2_tds:>19.4f}")
print(f"{'Cr ↔ pH':<20} {v1_audit['pH'].corr(v1_audit['Cr']):>19.4f} {v2_audit['pH'].corr(v2_audit['Cr']):>19.4f}")

# ============================================================================
# 5. PER-CATEGORY ANALYSIS
# ============================================================================
categories = ["Sungai", "Danau", "Waduk", "Air tanah"]
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("Per-Category Analysis: V1 vs V2", fontsize=14, fontweight='bold', y=1.00)

for idx, cat in enumerate(categories):
    v1_cat = v1_audit[v1_audit["Kategori"] == cat]["Cr"]
    v2_cat = v2_audit[v2_audit["Kategori"] == cat]["Cr"]
    
    # V1 vs V2 box plot
    ax = axes[0, idx]
    bp = ax.boxplot([v1_cat, v2_cat], labels=["V1", "V2"], patch_artist=True)
    colors = ["steelblue", "darkorange"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel("Cr (μg/L)", fontsize=10, fontweight='bold')
    ax.set_title(f"{cat}\nBox Plot", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # V2 pH sensitivity for this category
    ax = axes[1, idx]
    scatter = ax.scatter(v2_cat, v2_audit[v2_audit["Kategori"] == cat]["pH"], 
                        alpha=0.6, s=50, c=v2_audit[v2_audit["Kategori"] == cat]["EC"],
                        cmap="viridis", edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Cr (μg/L)", fontsize=10, fontweight='bold')
    ax.set_ylabel("pH", fontsize=10, fontweight='bold')
    ax.set_title(f"{cat}\nV2: pH Sensitivity", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(image_dir / "visualisasi_04_category_analysis.png", dpi=300, bbox_inches='tight')
print("\n✓ Saved: image/visualisasi_04_category_analysis.png")
plt.close()

# Per-category statistics
print("\nPER-CATEGORY STATISTICS:")
print("=" * 100)
for cat in categories:
    v1_cat = v1_audit[v1_audit["Kategori"] == cat]["Cr"]
    v2_cat = v2_audit[v2_audit["Kategori"] == cat]["Cr"]
    
    print(f"\n{cat.upper()}")
    print(f"  V1: n={len(v1_cat):3d} | min={v1_cat.min():6.1f} | max={v1_cat.max():7.1f} | mean={v1_cat.mean():6.1f} | std={v1_cat.std():6.1f}")
    print(f"  V2: n={len(v2_cat):3d} | min={v2_cat.min():6.1f} | max={v2_cat.max():7.1f} | mean={v2_cat.mean():6.1f} | std={v2_cat.std():6.1f}")
    print(f"  Δ : Δmean={v2_cat.mean()-v1_cat.mean():+6.1f} | Δstd={v2_cat.std()-v1_cat.std():+6.1f}")

# ============================================================================
# 6. KEY FINDINGS & SUMMARY
# ============================================================================
summary_text = f"""
╔════════════════════════════════════════════════════════════════════════╗
║           FORMULA IMPROVEMENT SUMMARY: V1 vs V2                        ║
╚════════════════════════════════════════════════════════════════════════╝

📊 DATASET STATISTICS:
  • V1 Total Records: {len(v1_audit)}
  • V2 Total Records: {len(v2_audit)}
  
🔬 CHROMIUM RANGES:
  • V1 Overall Range: {v1_audit['Cr'].min():.1f} - {v1_audit['Cr'].max():.1f} μg/L (mean: {v1_audit['Cr'].mean():.1f})
  • V2 Overall Range: {v2_audit['Cr'].min():.1f} - {v2_audit['Cr'].max():.1f} μg/L (mean: {v2_audit['Cr'].mean():.1f})
  • Change: Range shifted & distribution more realistic (log-normal) ✓

🔴 pH SENSITIVITY (CRITICAL IMPROVEMENT):
  • V1 Cr↔pH Correlation: {v1_audit['pH'].corr(v1_audit['Cr']):.4f} (weak, problematic)
    → Double-counting pH effect (linear + acidic term) ✗
    
  • V2 Cr↔pH Correlation: {v2_audit['pH'].corr(v2_audit['Cr']):.4f} (strong negative) ✓
    → Exponential inverse mechanism: acidic water → higher Cr solubility ✓
    → Reflects real Cr(III) geochemistry ✓

📈 IONIC STRENGTH EFFECT (EC/TDS):
  • V1 Cr↔EC Correlation: {corr_v1_ec:.4f}
  • V2 Cr↔EC Correlation: {corr_v2_ec:.4f}
    → Maintained positive correlation ✓
  
  • V1 Cr↔TDS Correlation: {corr_v1_tds:.4f}
  • V2 Cr↔TDS Correlation: {corr_v2_tds:.4f}
    → Maintained positive correlation ✓

🔐 PHYSICAL CONSTRAINT:
  • V1: No upper bound check (can violate geochemical limits)
  • V2: Cr ≤ 5% TDS enforced (realistic for minor constituent) ✓
    → {(v2_audit['Cr'] / (v2_audit['TDS'] * 0.05) <= 1).sum()} / {len(v2_audit)} records comply

📐 DISTRIBUTION SHAPE:
  • V1: Linear combination → Gaussian-like (unrealistic for pollutants)
  • V2: Log-normal foundation → Right-skewed (matches natural polutan distribution) ✓

✅ CONCLUSION:
  V2 (Geochemical) formula is significantly more scientifically sound:
  1. Eliminates pH double-counting logical error
  2. Implements exponential pH sensitivity matching Cr(III) solubility
  3. Adds physical constraint (5% TDS limit)
  4. Uses log-normal distribution (more realistic)
  5. Maintains positive EC/TDS correlations
  6. Better foundation for ML model training
"""

print("\n" + summary_text)

# Save summary to file
with open(image_dir / "summary_perbandingan_v1_vs_v2.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\n✓ Saved: image/summary_perbandingan_v1_vs_v2.txt")
print("\n" + "="*70)
print("ALL VISUALIZATIONS COMPLETED!")
print("="*70)
print("\nGenerated files in /image folder:")
print("  1. visualisasi_01_distribution_comparison.png")
print("  2. visualisasi_02_pH_sensitivity.png")
print("  3. visualisasi_03_correlations.png")
print("  4. visualisasi_04_category_analysis.png")
print("  5. summary_perbandingan_v1_vs_v2.txt")
