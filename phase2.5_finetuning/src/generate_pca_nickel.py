"""
generate_pca_nickel.py  --  HERA 2.0  Phase 2.5
==================================================
Generates an interactive 3D PCA visualization of the HERA 2.0 v2 geochemical space
to evaluate the distribution of Nickel concentrations in relation to raw and derived features.

Saves a publication-grade interactive HTML file to results/images/nickel_pca_3d.html.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset",
                                             "dataset_heavy_metal_grounded_v2.csv"))
out_dir      = os.path.abspath(os.path.join(script_dir, "..", "results", "images"))
os.makedirs(out_dir, exist_ok=True)

print("=" * 70)
print("  HERA 2.0 — Interactive 3D PCA Generator for Nickel")
print("=" * 70)

# ── Load Dataset ───────────────────────────────────────────────────────────
if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset v2 not found at: {dataset_path}")
    import sys
    sys.exit(1)

df = pd.read_csv(dataset_path)
print(f"[INFO] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── Features Definition ────────────────────────────────────────────────────
RAW = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
DERIVED = ["pH_squared", "pH_EC_interact", "log_EC", "pOH_proxy", "pH_temp_interact"]
FEATURES = RAW + DERIVED

X = df[FEATURES].values
y_ni = df["Nickel_ugL"].values

# ── Standardize & Perform PCA ──────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
total_var = explained_variance.sum() * 100

print(f"\n[PCA Results]")
print(f"  PC1 Explained Variance: {explained_variance[0]*100:.2f}%")
print(f"  PC2 Explained Variance: {explained_variance[1]*100:.2f}%")
print(f"  PC3 Explained Variance: {explained_variance[2]*100:.2f}%")
print(f"  Total Explained Variance (3 PCs): {total_var:.2f}%")

# ── Extract PCA Loadings ───────────────────────────────────────────────────
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2", "PC3"],
    index=FEATURES
)
print("\n[PCA Loading Matrix (Feature Weights)]")
print(loadings.to_string())

# ── Build Interactive DataFrame ────────────────────────────────────────────
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
pca_df["Nickel_ugL"] = y_ni
# Classify Nickel safety based on WHO 20 ug/L threshold
pca_df["Status_WHO"] = np.where(y_ni > 20.0, "Unsafe (>20 ug/L)", "Safe (<=20 ug/L)")

# Add raw columns for rich hover tooltips
for f in RAW:
    pca_df[f] = df[f]

# ── Create 3D Plotly Scatter Plot ──────────────────────────────────────────
fig = px.scatter_3d(
    pca_df,
    x="PC1",
    y="PC2",
    z="PC3",
    color="Nickel_ugL",
    color_continuous_scale="Viridis",
    labels={
        "PC1": f"PC1 ({explained_variance[0]*100:.1f}%)",
        "PC2": f"PC2 ({explained_variance[1]*100:.1f}%)",
        "PC3": f"PC3 ({explained_variance[2]*100:.1f}%)",
        "Nickel_ugL": "Nickel (ug/L)"
    },
    hover_data={
        "PC1": False,
        "PC2": False,
        "PC3": False,
        "Nickel_ugL": ":.2f",
        "Status_WHO": True,
        "pH": ":.2f",
        "EC_uScm": ":.1f",
        "TDS_mgL": ":.1f",
        "Suhu_Air": ":.2f"
    },
    title=f"HERA 2.0 — 3D PCA Geochemical Space for Nickel (Total Var Explained: {total_var:.1f}%)"
)

# Customize layout for elite academic appearance
fig.update_layout(
    title={
        'text': f"<b>HERA 2.0 — 3D PCA Geochemical Space for Nickel</b><br><sup>Standardized 9 Features Space | 15,000 Samples | Explained Var: {total_var:.1f}%</sup>",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    scene=dict(
        xaxis=dict(backgroundcolor="rgba(240, 240, 240, 0.4)", gridcolor="white", showbackground=True),
        yaxis=dict(backgroundcolor="rgba(240, 240, 240, 0.4)", gridcolor="white", showbackground=True),
        zaxis=dict(backgroundcolor="rgba(240, 240, 240, 0.4)", gridcolor="white", showbackground=True),
    ),
    margin=dict(l=0, r=0, b=0, t=80),
    font=dict(family="DejaVu Sans, Arial", size=11),
    coloraxis_colorbar=dict(
        title="Nickel (ug/L)",
        thicknessmode="pixels", thickness=15,
        lenmode="fraction", len=0.7,
        yanchor="middle", y=0.5
    )
)

# ── Save as HTML ───────────────────────────────────────────────────────────
out_path = os.path.join(out_dir, "nickel_pca_3d.html")
pio.write_html(fig, file=out_path, auto_open=False)
print(f"\n[SUCCESS] Saved interactive 3D PCA HTML -> {out_path}")
print("=" * 70)
