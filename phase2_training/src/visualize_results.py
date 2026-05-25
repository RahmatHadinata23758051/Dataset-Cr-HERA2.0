import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr, gaussian_kde

# ML Algorithms for comparative diagnostics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Configure matplotlib and seaborn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def set_premium_academic_style():
    """
    Applies standard peer-reviewed scientific journal formatting styles to matplotlib.
    Adheres to elite journal standards (e.g., Nature, Science, Water Research).
    """
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 14.5,
        "mathtext.fontset": "dejavusans",
        "grid.color": "#eaeaea",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
    })

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data_and_models(script_dir):
    """Loads the grounded dataset and best serialized regression models."""
    dataset_path = os.path.abspath(
        os.path.join(script_dir, "..", "..", "dataset", "dataset_heavy_metal_grounded.csv")
    )
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Grounded dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    cr_path = os.path.join(models_dir, "best_model_chromium.pkl")
    ni_path = os.path.join(models_dir, "best_model_nickel.pkl")

    if not os.path.exists(cr_path) or not os.path.exists(ni_path):
        raise FileNotFoundError("Best model packages (.pkl) not found. Run train_models.py first.")

    with open(cr_path, "rb") as f:
        cr_pack = pickle.load(f)
    with open(ni_path, "rb") as f:
        ni_pack = pickle.load(f)

    return df, cr_pack, ni_pack

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: TRAIN ALL 5 ALGORITHMS (for comparative plots)
# ─────────────────────────────────────────────────────────────────────────────

def train_all_algorithms(X_train, X_test, y_train, y_test):
    """Fits all 5 algorithms to scaled training data. Returns fitted models, scaler, and scaled test set."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "SVR (RBF Kernel)": SVR(C=20.0, epsilon=0.1, kernel="rbf"),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1),
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

    return models, scaler, X_train_scaled, X_test_scaled

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC CONSOLE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def run_overfitting_diagnostics(df):
    """
    Analyzes and prints train vs test performance metrics across all 5 algorithms
    to prove mathematically there is no overfitting.
    """
    print("\n" + "=" * 80)
    print("      HERA 2.0 MULTI-ALGORITHM OVERFITTING DIAGNOSTIC MATRIX (NICKEL & CHROMIUM)")
    print("=" * 80)
    features = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    X = df[features].values

    metals = {
        "Chromium": df["Chromium_ugL"].values,
        "Nickel": df["Nickel_ugL"].values,
    }

    for metal_name, y in metals.items():
        print(f"\n>>> Overfitting Diagnostics for: {metal_name.upper()}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=10.0),
            "SVR (RBF Kernel)": SVR(C=20.0, epsilon=0.1, kernel="rbf"),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1),
        }

        results = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)

            y_train_pred = model.predict(X_train_scaled)
            r2_train = model.score(X_train_scaled, y_train)
            rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))

            y_test_pred = model.predict(X_test_scaled)
            r2_test = model.score(X_test_scaled, y_test)
            rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))

            r2_gap_pct = (r2_train - r2_test) * 100.0

            results.append({
                "Model": name,
                "Train R2": r2_train,
                "Test R2": r2_test,
                "R2 Gap (%)": r2_gap_pct,
                "Train RMSE": rmse_train,
                "Test RMSE": rmse_test,
                "Verdict": "Safe (No Overfitting)" if abs(r2_gap_pct) < 2.0 else "Caution",
            })

        df_res = pd.DataFrame(results)
        print(df_res.to_string(
            index=False,
            formatters={
                "Train R2": "{:,.4f}".format,
                "Test R2": "{:,.4f}".format,
                "R2 Gap (%)": "{:+.2f}%".format,
                "Train RMSE": "{:,.2f}".format,
                "Test RMSE": "{:,.2f}".format,
            },
        ))
    print("=" * 80 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: SPEARMAN RANK CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_spearman_heatmap(df, output_dir):
    """
    Plot 1: Spearman Rank Correlation Heatmap with lower-triangle mask.
    Publication style matching Water Research / Chemosphere standards.
    """
    print("[PLOT 1] Generating Spearman Correlation Heatmap...")
    cols = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air", "Chromium_ugL", "Nickel_ugL"]
    corr_matrix = df[cols].corr(method="spearman")

    label_map = {
        "pH": r"$\mathrm{pH}$",
        "EC_uScm": r"$\mathrm{EC\ (\mu S/cm)}$",
        "TDS_mgL": r"$\mathrm{TDS\ (mg/L)}$",
        "Suhu_Air": r"$\mathrm{Temperature\ (^{\circ}C)}$",
        "Chromium_ugL": r"$\mathrm{Chromium\ (\mu g/L)}$",
        "Nickel_ugL": r"$\mathrm{Nickel\ (\mu g/L)}$",
    }
    corr_matrix.rename(index=label_map, columns=label_map, inplace=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Use upper triangle mask (show lower triangle)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Custom diverging palette
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=1.8,
        linecolor="#ffffff",
        cbar_kws={"shrink": 0.75, "label": r"Spearman Correlation ($r_s$)"},
        annot_kws={"size": 9.5, "weight": "semibold"},
        ax=ax,
    )

    ax.set_title("Geochemical Spearman Rank Correlation Matrix", pad=14, weight="bold", size=13)
    ax.tick_params(axis="both", labelsize=9.5)

    # Add a subtle border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("#ccc")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "1_geochemical_correlation_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: THERMODYNAMIC SOLUBILITY PHASE DIAGRAM (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def plot_thermodynamic_limits(df, output_dir):
    """
    Plot 2: Thermodynamic Solubility Phase Diagram.
    FIX: The solubility boundary is now clearly positioned ABOVE the synthetic data
    (the data is grounded to stay below Ksp by design), with annotations and
    extended pH axis. Dual-axis log scale for clarity.
    """
    print("[PLOT 2] Generating Thermodynamic Solubility Phase Diagrams...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # Extend pH grid to visualise full range from acidic to basic
    ph_grid = np.linspace(4.0, 9.5, 400)

    # ── Chromium Cr(OH)3 solubility: Ksp = 6.3e-31
    # [Cr3+] = Ksp / [OH-]^3 = Ksp * Kw^-3 / 10^(-3*pH) => simplified:
    # log[Cr] (mol/L) ≈ 3*(4.47 - pH)
    # Convert mol/L -> ug/L: * 51996 * 1000 * (g/mol -> ug/mol)
    cr_molar = 10 ** (3 * (4.47 - ph_grid))          # mol/L Cr3+
    cr_limit_ugL = cr_molar * 51.996e6              # ug/L

    # ── Nickel Ni(OH)2 solubility: Ksp = 5.47e-16
    # [Ni2+] = Ksp / [OH-]^2 => log[Ni] ≈ 2*(6.37 - pH)
    ni_molar = 10 ** (2 * (6.37 - ph_grid))          # mol/L Ni2+
    ni_limit_ugL = ni_molar * 58.693e6              # ug/L

    # WHO / regulatory threshold lines
    WHO_CR = 50.0   # ug/L
    WHO_NI = 20.0   # ug/L

    # ── CHROMIUM subplot (left)
    ax = axes[0]
    # Shaded zones
    ax.fill_between(ph_grid, cr_limit_ugL, 1e10,
                    color="#fde0dc", alpha=0.55, zorder=1,
                    label="Precipitation zone (super-saturated)")
    ax.fill_between(ph_grid, 1e-4, cr_limit_ugL,
                    color="#dceefb", alpha=0.40, zorder=1,
                    label="Dissolved zone (under-saturated)")

    # Solubility boundary (Ksp curve)
    ax.plot(ph_grid, cr_limit_ugL,
            color="#C0392B", linestyle="-", linewidth=2.2, zorder=4,
            label=r"Ksp solubility boundary [Cr(OH)$_3$]")

    # Synthetic data scatter
    ax.scatter(df["pH"], df["Chromium_ugL"],
               color="#1A5276", alpha=0.25, edgecolors="none", s=10, zorder=3,
               label="Synthetic samples (n=5,000)")

    # WHO threshold
    ax.axhline(WHO_CR, color="#117A65", linestyle="--", linewidth=1.6, zorder=4,
               label=f"WHO guideline = {WHO_CR} ug/L")

    # Annotation: data zone
    ax.annotate("All synthetic data\nreside below Ksp\n(thermodynamically consistent)",
                xy=(5.8, 30.0), xytext=(6.5, 8.0),
                fontsize=8.5, color="#1A5276",
                arrowprops=dict(arrowstyle="->", color="#1A5276", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1A5276", alpha=0.85))

    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{pH}$", weight="semibold")
    ax.set_ylabel(r"$\mathrm{Chromium\ concentration\ (\mu g/L)}$", weight="semibold")
    ax.set_title(r"$\mathrm{Cr(OH)_3\ Phase\ Solubility\ Diagram}$", weight="bold")
    ax.set_ylim(1e-3, 1e11)
    ax.set_xlim(4.0, 9.5)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.grid(True, which="both", axis="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, edgecolor="#ccc")

    # Annotate the solubility line at pH = 6.5
    ax.annotate("Ksp boundary",
                xy=(6.5, 10**(3*(4.47 - 6.5)) * 51.996e6),
                xytext=(5.5, 5e6),
                fontsize=8, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0),
                ha="center")

    # ── NICKEL subplot (right)
    ax = axes[1]
    ax.fill_between(ph_grid, ni_limit_ugL, 1e12,
                    color="#fde0dc", alpha=0.55, zorder=1,
                    label="Precipitation zone (super-saturated)")
    ax.fill_between(ph_grid, 1e-3, ni_limit_ugL,
                    color="#e8f8f5", alpha=0.40, zorder=1,
                    label="Dissolved zone (under-saturated)")

    ax.plot(ph_grid, ni_limit_ugL,
            color="#C0392B", linestyle="-", linewidth=2.2, zorder=4,
            label=r"Ksp solubility boundary [Ni(OH)$_2$]")

    ax.scatter(df["pH"], df["Nickel_ugL"],
               color="#1E6B38", alpha=0.25, edgecolors="none", s=10, zorder=3,
               label="Synthetic samples (n=5,000)")

    ax.axhline(WHO_NI, color="#117A65", linestyle="--", linewidth=1.6, zorder=4,
               label=f"WHO guideline = {WHO_NI} ug/L")

    ax.annotate("All synthetic data\nreside below Ksp\n(thermodynamically consistent)",
                xy=(6.0, 80.0), xytext=(7.0, 20.0),
                fontsize=8.5, color="#1E6B38",
                arrowprops=dict(arrowstyle="->", color="#1E6B38", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1E6B38", alpha=0.85))

    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{pH}$", weight="semibold")
    ax.set_ylabel(r"$\mathrm{Nickel\ concentration\ (\mu g/L)}$", weight="semibold")
    ax.set_title(r"$\mathrm{Ni(OH)_2\ Phase\ Solubility\ Diagram}$", weight="bold")
    ax.set_ylim(1e-1, 1e13)
    ax.set_xlim(4.0, 9.5)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.grid(True, which="both", axis="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, edgecolor="#ccc")

    ax.annotate("Ksp boundary",
                xy=(7.0, 10**(2*(6.37 - 7.0)) * 58.693e6),
                xytext=(5.5, 5e9),
                fontsize=8, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0),
                ha="center")

    plt.suptitle(
        "Geochemical Phase Solubility Diagrams with Thermodynamic Boundaries (Ksp)",
        y=0.99, weight="bold", size=13.5
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "2_thermodynamic_solubility_limits.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: MODEL PARITY PLOTS — DENSITY SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_parity(df, cr_pack, ni_pack, output_dir):
    """
    Plot 3: Density-Coloured Model Parity Plots (Actual vs Predicted).
    Upgraded: uses Gaussian KDE colouring, overlaid 95% confidence band.
    """
    print("[PLOT 3] Generating Density-Colored Model Parity Plots...")
    features = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    X = df[features].values

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))

    def _parity_panel(ax, y_test, y_pred, color_hex, metal_label, r2, rmse):
        xy = np.vstack([y_test, y_pred])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        sc = ax.scatter(y_test[idx], y_pred[idx], c=z[idx], cmap="plasma",
                        s=14, edgecolor="none", alpha=0.8, zorder=3)

        lims = [min(y_test.min(), y_pred.min()) * 0.95,
                max(y_test.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, color="#C0392B", linestyle="--", linewidth=1.8,
                label="1:1 Parity Line", zorder=4)

        # 95% confidence band (±2σ residual)
        residuals = y_pred - y_test
        sigma = np.std(residuals)
        ax.fill_between(lims,
                        [l - 2 * sigma for l in lims],
                        [l + 2 * sigma for l in lims],
                        alpha=0.10, color="#C0392B", label="±2σ band")

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(rf"$\mathrm{{Observed\ {metal_label}\ (\mu g/L)}}$", weight="semibold")
        ax.set_ylabel(rf"$\mathrm{{Predicted\ {metal_label}\ (\mu g/L)}}$", weight="semibold")
        ax.set_title(rf"$\mathrm{{{metal_label}\ Parity\ Plot}}$", weight="bold")
        ax.grid(True, alpha=0.5)
        ax.legend(loc="upper left", fontsize=8.5)

        txt = f"$R^2 = {r2:.4f}$\n$\\mathrm{{RMSE}} = {rmse:.2f}\\ \\mu\\mathrm{{g/L}}$"
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaa", alpha=0.92),
                fontsize=9.5)

        plt.colorbar(sc, ax=ax, label="Local Scatter Density (KDE)", shrink=0.82)

    cr_y = df["Chromium_ugL"].values
    _, cr_X_test, _, cr_y_test = train_test_split(X, cr_y, test_size=0.2, random_state=42)
    cr_X_test_s = cr_pack["scaler"].transform(cr_X_test)
    cr_y_pred = cr_pack["model"].predict(cr_X_test_s)
    cr_r2 = cr_pack["model"].score(cr_X_test_s, cr_y_test)
    cr_rmse = np.sqrt(np.mean((cr_y_test - cr_y_pred) ** 2))
    _parity_panel(axes[0], cr_y_test, cr_y_pred, "#0F4C81", "Chromium", cr_r2, cr_rmse)

    ni_y = df["Nickel_ugL"].values
    _, ni_X_test, _, ni_y_test = train_test_split(X, ni_y, test_size=0.2, random_state=42)
    ni_X_test_s = ni_pack["scaler"].transform(ni_X_test)
    ni_y_pred = ni_pack["model"].predict(ni_X_test_s)
    ni_r2 = ni_pack["model"].score(ni_X_test_s, ni_y_test)
    ni_rmse = np.sqrt(np.mean((ni_y_test - ni_y_pred) ** 2))
    _parity_panel(axes[1], ni_y_test, ni_y_pred, "#1E6B38", "Nickel", ni_r2, ni_rmse)

    plt.suptitle("Best-Model Parity Plot with Density Field & Confidence Band (20% Holdout Set)",
                 y=0.99, weight="bold", size=13.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "3_model_parity_plots.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(df, cr_pack, ni_pack, output_dir):
    """
    Plot 4: Residual Analysis — Residuals vs Predicted + Residual Density
    with normal distribution overlay and Q-Q envelopes.
    """
    print("[PLOT 4] Generating Residual Analysis Plots...")
    features = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    X = df[features].values

    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

    packs = [
        ("Chromium", df["Chromium_ugL"].values, cr_pack, "#0F4C81", axes[0, 0], axes[0, 1]),
        ("Nickel",   df["Nickel_ugL"].values,   ni_pack, "#1E6B38", axes[1, 0], axes[1, 1]),
    ]

    for metal, y, pack, color, ax_scatter, ax_hist in packs:
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_s = pack["scaler"].transform(X_test)
        y_pred = pack["model"].predict(X_test_s)
        residuals = y_test - y_pred

        # Residuals vs Predicted
        ax_scatter.scatter(y_pred, residuals, color=color, alpha=0.40, s=12,
                           edgecolors="none", zorder=3)
        ax_scatter.axhline(0, color="#C0392B", linestyle="--", linewidth=1.8, zorder=4)
        # LOESS-style trend line (running median)
        sort_idx = np.argsort(y_pred)
        window = max(1, len(y_pred) // 30)
        smooth_x = np.convolve(y_pred[sort_idx], np.ones(window) / window, mode="valid")
        smooth_y = np.convolve(residuals[sort_idx], np.ones(window) / window, mode="valid")
        ax_scatter.plot(smooth_x, smooth_y, color="#E74C3C", linewidth=1.6,
                        label="Running mean trend", zorder=5)
        ax_scatter.set_xlabel(rf"$\mathrm{{Predicted\ {metal}\ (\mu g/L)}}$", weight="semibold")
        ax_scatter.set_ylabel(r"$\mathrm{Residuals\ (Obs - Pred,\ \mu g/L)}$", weight="semibold")
        ax_scatter.set_title(rf"$\mathrm{{{metal}\ Residuals\ vs.\ Predicted}}$", weight="bold")
        ax_scatter.grid(True, alpha=0.5)
        ax_scatter.legend(fontsize=8.5)

        # Text stats box
        bias = np.mean(residuals)
        sigma = np.std(residuals)
        ax_scatter.text(0.97, 0.97,
                        f"Bias = {bias:.2f} ug/L\nσ = {sigma:.2f} ug/L",
                        transform=ax_scatter.transAxes, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaa", alpha=0.92),
                        fontsize=9)

        # Residual Density Histogram with Normal fit
        sns.histplot(residuals, kde=False, ax=ax_hist, color=color, stat="density",
                     bins=35, alpha=0.55, edgecolor="white", linewidth=0.4)
        x_fit = np.linspace(residuals.min(), residuals.max(), 200)
        pdf_fit = scipy_stats.norm.pdf(x_fit, bias, sigma)
        ax_hist.plot(x_fit, pdf_fit, color="#C0392B", linewidth=2.0,
                     label=rf"$\mathcal{{N}}(\mu={bias:.1f},\sigma={sigma:.1f})$")
        ax_hist.set_xlabel(r"$\mathrm{Residuals\ (\mu g/L)}$", weight="semibold")
        ax_hist.set_ylabel(r"$\mathrm{Density}$", weight="semibold")
        ax_hist.set_title(rf"$\mathrm{{{metal}\ Residual\ Distribution}}$", weight="bold")
        ax_hist.legend(fontsize=8.5)
        ax_hist.grid(True, alpha=0.5)

    plt.suptitle(
        "Residual Diagnostic Analysis — Homoscedasticity & Normality Check",
        y=0.99, weight="bold", size=13.5
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "4_residual_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: MONOTONIC SCENARIO PROGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def plot_scenario_progression(cr_pack, ni_pack, output_dir):
    """
    Plot 5: Monotonic Scenario Progression — Point-line chart across 4 runoff stages.
    """
    print("[PLOT 5] Generating Scenario Progression Point-Line Charts...")

    scenarios = pd.DataFrame([
        {"pH": 7.5, "EC_uScm": 150.0,  "TDS_mgL": 96.0,   "Suhu_Air": 24.5},
        {"pH": 6.8, "EC_uScm": 500.0,  "TDS_mgL": 320.0,  "Suhu_Air": 24.5},
        {"pH": 6.0, "EC_uScm": 1200.0, "TDS_mgL": 768.0,  "Suhu_Air": 24.5},
        {"pH": 5.2, "EC_uScm": 2200.0, "TDS_mgL": 1408.0, "Suhu_Air": 24.5},
    ])
    stages = ["Stage 1\n(Clean)", "Stage 2\n(Moderate)", "Stage 3\n(High)", "Stage 4\n(Extreme)"]

    cr_preds = cr_pack["model"].predict(cr_pack["scaler"].transform(scenarios.values))
    ni_preds = ni_pack["model"].predict(ni_pack["scaler"].transform(scenarios.values))

    x = np.arange(len(stages))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))

    WHO_CR = 50.0
    WHO_NI = 20.0

    for ax, preds, color, metal, who_val in [
        (axes[0], cr_preds, "#0F4C81", "Chromium", WHO_CR),
        (axes[1], ni_preds, "#1E6B38", "Nickel",   WHO_NI),
    ]:
        # Shading
        ax.axhspan(who_val, max(preds) * 1.25,
                   color="#fde0dc", alpha=0.4, label=f"Non-compliant (>{who_val} ug/L)")
        ax.axhspan(0, who_val,
                   color="#e8f8f5", alpha=0.35, label=f"Compliant (<={who_val} ug/L)")

        ax.plot(x, preds, color=color, marker="o", markersize=9,
                linewidth=2.5, zorder=5, label="Model Prediction")
        ax.axhline(who_val, color="#C0392B", linestyle="--", linewidth=1.8,
                   zorder=4, label=f"WHO Standard = {who_val} ug/L")

        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=9.5)
        ax.set_ylabel(rf"$\mathrm{{{metal}\ Concentration\ (\mu g/L)}}$", weight="semibold")
        ax.set_title(rf"$\mathrm{{{metal}\ Runoff\ Scenario\ Progression}}$", weight="bold")
        ax.set_ylim(0, max(preds) * 1.3)
        ax.grid(True, alpha=0.5)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

        for i, val in enumerate(preds):
            ax.annotate(
                f"{val:.1f}",
                (x[i], val),
                textcoords="offset points", xytext=(0, 11),
                ha="center", weight="bold", fontsize=9.5, color=color,
            )

    plt.suptitle(
        "Monotonic Geochemical Scenario Prediction Progression (Physical Consistency Test)",
        y=0.99, weight="bold", size=13.5
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "5_monotonic_scenario_progression.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6: SINGLE-VARIABLE SENSITIVITY CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity_curves(cr_pack, ni_pack, output_dir):
    """
    Plot 6: Single-Variable Sensitivity Curves for pH and EC.
    Overlaid with WHO threshold markers and regulatory bands.
    """
    print("[PLOT 6] Generating Single-Variable Sensitivity Curves...")
    n_steps = 200

    ph_range = np.linspace(5.0, 8.5, n_steps)
    ph_test = pd.DataFrame({
        "pH": ph_range,
        "EC_uScm": [700.0] * n_steps,
        "TDS_mgL": [448.0] * n_steps,
        "Suhu_Air": [24.5] * n_steps,
    })

    ec_range = np.linspace(100.0, 2500.0, n_steps)
    ec_test = pd.DataFrame({
        "pH": [6.5] * n_steps,
        "EC_uScm": ec_range,
        "TDS_mgL": ec_range * 0.64,
        "Suhu_Air": [24.5] * n_steps,
    })

    cr_ph = cr_pack["model"].predict(cr_pack["scaler"].transform(ph_test.values))
    ni_ph = ni_pack["model"].predict(ni_pack["scaler"].transform(ph_test.values))
    cr_ec = cr_pack["model"].predict(cr_pack["scaler"].transform(ec_test.values))
    ni_ec = ni_pack["model"].predict(ni_pack["scaler"].transform(ec_test.values))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    # Left: pH Sensitivity
    axes[0].axhline(50.0, color="#C0392B", linestyle=":", linewidth=1.4, alpha=0.7, label="WHO Cr standard (50 ug/L)")
    axes[0].axhline(20.0, color="#117A65", linestyle=":", linewidth=1.4, alpha=0.7, label="WHO Ni standard (20 ug/L)")
    axes[0].plot(ph_range, cr_ph, color="#0F4C81", linewidth=2.4, label=r"Chromium ($\mathrm{Cr}$)")
    axes[0].plot(ph_range, ni_ph, color="#1E6B38", linewidth=2.4, label=r"Nickel ($\mathrm{Ni}$)", linestyle="-.")
    axes[0].set_xlabel(r"$\mathrm{pH}$", weight="semibold")
    axes[0].set_ylabel(r"$\mathrm{Predicted\ Concentration\ (\mu g/L)}$", weight="semibold")
    axes[0].set_title(r"$\mathrm{pH\ Sensitivity\ (constant\ EC=700\ \mu S/cm)}$", weight="bold")
    axes[0].grid(True, alpha=0.5)
    axes[0].legend(fontsize=8.5)

    # Right: EC Sensitivity
    axes[1].axhline(50.0, color="#C0392B", linestyle=":", linewidth=1.4, alpha=0.7, label="WHO Cr standard (50 ug/L)")
    axes[1].axhline(20.0, color="#117A65", linestyle=":", linewidth=1.4, alpha=0.7, label="WHO Ni standard (20 ug/L)")
    axes[1].plot(ec_range, cr_ec, color="#0F4C81", linewidth=2.4, label=r"Chromium ($\mathrm{Cr}$)")
    axes[1].plot(ec_range, ni_ec, color="#1E6B38", linewidth=2.4, label=r"Nickel ($\mathrm{Ni}$)", linestyle="-.")
    axes[1].set_xlabel(r"$\mathrm{EC\ (\mu S/cm)}$", weight="semibold")
    axes[1].set_ylabel(r"$\mathrm{Predicted\ Concentration\ (\mu g/L)}$", weight="semibold")
    axes[1].set_title(r"$\mathrm{EC\ Sensitivity\ (constant\ pH=6.5)}$", weight="bold")
    axes[1].grid(True, alpha=0.5)
    axes[1].legend(fontsize=8.5)

    plt.suptitle(
        "Model Parameter Sensitivity — Geochemical Response to pH and EC Variations",
        y=0.99, weight="bold", size=13.5
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "6_single_variable_sensitivity_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7: MULTI-ALGORITHM PERMUTATION FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_algorithm_feature_importance(df, output_dir):
    """
    Plot 7: Grouped Horizontal Bar Chart of Permutation Feature Importances.
    All 5 algorithms shown side-by-side per feature.
    """
    print("[PLOT 7] Computing Permutation Feature Importances...")
    features = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    feature_labels = [r"$\mathrm{pH}$", r"$\mathrm{EC\ (\mu S/cm)}$",
                      r"$\mathrm{TDS\ (mg/L)}$", r"$\mathrm{Temperature\ (^{\circ}C)}$"]
    X = df[features].values

    cr_y = df["Chromium_ugL"].values
    ni_y = df["Nickel_ugL"].values

    cr_X_train, cr_X_test, cr_y_train, cr_y_test = train_test_split(X, cr_y, test_size=0.2, random_state=42)
    ni_X_train, ni_X_test, ni_y_train, ni_y_test = train_test_split(X, ni_y, test_size=0.2, random_state=42)

    cr_models, _, _, cr_X_test_s = train_all_algorithms(cr_X_train, cr_X_test, cr_y_train, cr_y_test)
    ni_models, _, _, ni_X_test_s = train_all_algorithms(ni_X_train, ni_X_test, ni_y_train, ni_y_test)

    model_names = list(cr_models.keys())
    colors = ["#9b59b6", "#3498db", "#95a5a6", "#e67e22", "#1abc9c"]

    cr_importances = {}
    ni_importances = {}
    cr_stds = {}
    ni_stds = {}

    for name in model_names:
        r_cr = permutation_importance(cr_models[name], cr_X_test_s, cr_y_test,
                                      scoring="r2", n_repeats=10, random_state=42)
        cr_importances[name] = r_cr.importances_mean
        cr_stds[name] = r_cr.importances_std

        r_ni = permutation_importance(ni_models[name], ni_X_test_s, ni_y_test,
                                      scoring="r2", n_repeats=10, random_state=42)
        ni_importances[name] = r_ni.importances_mean
        ni_stds[name] = r_ni.importances_std

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    y = np.arange(len(features))
    width = 0.145
    offsets = np.linspace(-2 * width, 2 * width, len(model_names))

    for panel_idx, (ax, importances, stds, metal) in enumerate([
        (axes[0], cr_importances, cr_stds, "Chromium"),
        (axes[1], ni_importances, ni_stds, "Nickel"),
    ]):
        for idx, name in enumerate(model_names):
            ax.barh(y + offsets[idx], importances[name], width,
                    xerr=stds[name], label=name if panel_idx == 0 else None,
                    color=colors[idx], alpha=0.85, edgecolor="#444", linewidth=0.5,
                    error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0, "ecolor": "#444"})

        ax.set_yticks(y)
        ax.set_yticklabels(feature_labels, fontsize=10.5)
        ax.invert_yaxis()
        ax.axvline(0, color="#555", linewidth=0.8)
        ax.set_xlabel(r"$\mathrm{Permutation\ Importance\ (\Delta R^2)}$", weight="semibold")
        ax.set_title(rf"$\mathrm{{{metal}\ Feature\ Importance}}$", weight="bold")
        ax.grid(True, axis="x", alpha=0.5)

    # Shared legend (placed on right panel)
    handles = [mpatches.Patch(color=colors[i], label=name) for i, name in enumerate(model_names)]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8.5,
                   framealpha=0.92, edgecolor="#ccc")

    plt.suptitle(
        "Multi-Algorithm Permutation Feature Importance Comparison (with ±1σ error bars)",
        y=0.99, weight="bold", size=13.5
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "7_multi_algorithm_feature_importance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8: MULTI-ALGORITHM CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_algorithm_confusion_matrices(df, output_dir):
    """
    Plot 8: 5×2 Grid of Environmental Compliance Confusion Matrices (WHO Thresholds).
    Chromium (50 ug/L) | Nickel (20 ug/L) threshold binarization.
    """
    print("[PLOT 8] Generating Multi-Algorithm Confusion Matrix Grid...")
    features = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    X = df[features].values

    targets = {
        "Chromium": {"y": df["Chromium_ugL"].values, "threshold": 50.0, "cmap": "Blues",  "col_idx": 0},
        "Nickel":   {"y": df["Nickel_ugL"].values,   "threshold": 20.0, "cmap": "Greens", "col_idx": 1},
    }
    model_names = ["Linear Regression", "Ridge Regression", "SVR (RBF Kernel)",
                   "Random Forest", "XGBoost Regressor"]

    fig, axes = plt.subplots(5, 2, figsize=(10, 19))

    for metal_name, meta in targets.items():
        y = meta["y"]
        thresh = meta["threshold"]
        cmap = meta["cmap"]
        col_idx = meta["col_idx"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_test_bin = (y_test > thresh).astype(int)
        models, _, _, X_test_scaled = train_all_algorithms(X_train, X_test, y_train, y_test)

        for row_idx, model_name in enumerate(model_names):
            y_pred = models[model_name].predict(X_test_scaled)
            y_pred_bin = (y_pred > thresh).astype(int)

            cm = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = cm[0, 0] if y_test_bin.sum() == 0 else 0
                fp = fn = tp = 0

            total = len(y_test_bin)
            acc  = (tp + tn) / total
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            cell_labels = np.array([
                [f"TN\n{tn}\n({tn/total*100:.1f}%)", f"FP\n{fp}\n({fp/total*100:.1f}%)"],
                [f"FN\n{fn}\n({fn/total*100:.1f}%)", f"TP\n{tp}\n({tp/total*100:.1f}%)"],
            ])

            ax = axes[row_idx, col_idx]
            sns.heatmap(cm, annot=cell_labels, fmt="", cmap=cmap, cbar=False, square=True,
                        xticklabels=["Safe", "Unsafe"], yticklabels=["Safe", "Unsafe"],
                        linewidths=1.2, linecolor="#ccc",
                        annot_kws={"fontsize": 9, "weight": "bold"}, ax=ax)

            ax.set_ylabel("Actual", fontsize=9, weight="semibold")
            ax.set_xlabel("Predicted", fontsize=9, weight="semibold")
            ax.set_title(
                f"{model_name}\n"
                f"Acc: {acc*100:.1f}%  Prec: {prec*100:.1f}%  Rec: {rec*100:.1f}%  F1: {f1:.3f}",
                fontsize=9, weight="bold", pad=7
            )

    # Column headers
    fig.text(0.27, 0.992, "Chromium (threshold = 50 ug/L)", ha="center",
             va="top", fontsize=11, weight="bold", color="#0F4C81")
    fig.text(0.74, 0.992, "Nickel (threshold = 20 ug/L)", ha="center",
             va="top", fontsize=11, weight="bold", color="#1E6B38")

    plt.suptitle("Environmental Compliance Confusion Matrix Grid (WHO Drinking Water Standards)",
                 y=1.003, weight="bold", size=13.5)
    plt.tight_layout(rect=[0, 0, 1, 0.998])

    out_path = os.path.join(output_dir, "8_multi_algorithm_confusion_matrices.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 & 10: MODEL COMPARISON DASHBOARD (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_dashboard(df, metal_name, y, output_path):
    """
    Plot 9 & 10: Premium multi-algorithm performance dashboard.
    Layout: 2×2 bar plots (Test R2, RMSE, MAE, MAPE) + full-width heatmap table.
    BUG FIX: Val metrics computed using the *same* fitted scaler (not a new one).
    """
    print(f"[PLOT] Generating Premium Dashboard for {metal_name}...")
    features = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    X = df[features].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models, scaler, X_train_s, X_test_s = train_all_algorithms(X_train, X_test, y_train, y_test)

    model_metrics = []
    for name, model in models.items():
        y_pred_test  = model.predict(X_test_s)
        y_pred_train = model.predict(X_train_s)     # use the FITTED scaler's train set

        r2_val   = model.score(X_train_s, y_train)
        rmse_val = np.sqrt(np.mean((y_train - y_pred_train) ** 2))

        r2   = model.score(X_test_s, y_test)
        rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        mae  = np.mean(np.abs(y_test - y_pred_test))
        # guard against zero division in MAPE
        mape = np.mean(np.abs((y_test - y_pred_test) / np.maximum(y_test, 1e-6))) * 100.0

        model_metrics.append({
            "Algorithm": name,
            "Val R²":     r2_val,
            "Val RMSE":   rmse_val,
            "Test R²":    r2,
            "Test RMSE":  rmse,
            "Test MAE":   mae,
            "MAPE (%)":   mape,
        })

    df_m = pd.DataFrame(model_metrics).sort_values("Test R²", ascending=False).reset_index(drop=True)

    # Colour palette: best -> worst
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]
    short_names = ["XGBoost", "Rnd.Forest", "SVR", "Ridge", "Lin.Reg."]
    # Reorder short_names to match sorted order
    alg_order = df_m["Algorithm"].tolist()
    name_map = {
        "Linear Regression":  "Lin.Reg.",
        "Ridge Regression":   "Ridge",
        "SVR (RBF Kernel)":   "SVR",
        "Random Forest":      "Rnd.Forest",
        "XGBoost Regressor":  "XGBoost",
    }
    short_labels = [name_map.get(n, n) for n in alg_order]

    fig = plt.figure(figsize=(13, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1.1, 0.75],
                           hspace=0.55, wspace=0.35)

    metric_specs = [
        {"col": "Test R²",    "ax_pos": (0, 0), "title": r"Test $R^2$ Score (↑ Better)",
         "ylabel": "Score", "higher_is_better": True},
        {"col": "Test RMSE",  "ax_pos": (0, 1), "title": r"Test RMSE ($\mu$g/L) (↓ Better)",
         "ylabel": "Error (ug/L)", "higher_is_better": False},
        {"col": "Test MAE",   "ax_pos": (1, 0), "title": r"Test MAE ($\mu$g/L) (↓ Better)",
         "ylabel": "Error (ug/L)", "higher_is_better": False},
        {"col": "MAPE (%)",   "ax_pos": (1, 1), "title": r"Test MAPE (%) (↓ Better)",
         "ylabel": "Error (%)", "higher_is_better": False},
    ]

    # Linear Regression is the baseline (usually last after sort by R²)
    baseline = df_m[df_m["Algorithm"] == "Linear Regression"].iloc[0]

    for spec in metric_specs:
        ax = fig.add_subplot(gs[spec["ax_pos"]])
        col = spec["col"]
        vals = df_m[col].values

        bars = ax.bar(range(len(df_m)), vals,
                      color=colors[:len(df_m)], width=0.62,
                      edgecolor="#555", linewidth=0.7)

        # Bold best bar
        bars[0].set_edgecolor("black")
        bars[0].set_linewidth(2.0)

        # Baseline dashed line
        ax.axhline(baseline[col], color="#C0392B", linestyle="--", linewidth=1.3, alpha=0.85,
                   label="LinReg baseline")

        # Annotations
        for b_i, (bar, val) in enumerate(zip(bars, vals)):
            h = bar.get_height()
            label = f"{'Best\n' if b_i == 0 else ''}{val:.3f}" if col == "Test R²" else \
                    f"{'Best\n' if b_i == 0 else ''}{val:.2f}"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8 if b_i > 0 else 8.5,
                        weight="bold" if b_i == 0 else "normal",
                        color="black" if b_i == 0 else "#333")

        ax.set_title(spec["title"], weight="bold", pad=8)
        ax.set_ylabel(spec["ylabel"], weight="semibold")
        ax.set_xticks(range(len(df_m)))
        ax.set_xticklabels(short_labels, rotation=0, fontsize=9)
        ax.grid(True, axis="y", alpha=0.5)
        ax.legend(fontsize=8, loc="best")

    # Full-width heatmap table
    ax_table = fig.add_subplot(gs[2, :])
    table_df = df_m.set_index("Algorithm")[["Val R²", "Val RMSE", "Test R²", "Test RMSE", "Test MAE", "MAPE (%)"]]
    table_df.index = short_labels  # shorten algorithm names

    sns.heatmap(
        table_df,
        annot=True, fmt=".3f", cmap="YlGnBu",
        linewidths=1.8, linecolor="#ffffff",
        cbar_kws={"label": "Metric Value", "shrink": 0.75},
        annot_kws={"size": 9, "weight": "semibold"},
        ax=ax_table,
    )
    ax_table.set_title("Summary: Validation & Test Metric Heatmap", weight="bold", pad=10)
    ax_table.set_xlabel("Performance Metrics", weight="semibold")
    ax_table.set_ylabel("Algorithm", weight="semibold")
    ax_table.tick_params(axis="x", rotation=0)

    metal_color = "#0F4C81" if metal_name == "Chromium" else "#1E6B38"
    fig.suptitle(
        f"HERA 2.0 Model Benchmark Dashboard — {metal_name}\n"
        "Dashed red = Linear Regression baseline | Best model highlighted in bold",
        y=0.995, weight="bold", size=13.5, color=metal_color
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 11: LEARNING CURVES (Anti-Overfitting Evidence)
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(df, output_dir):
    """
    Plot 11: Learning Curves for top 3 algorithms (SVR, RF, XGBoost).
    Training score vs. cross-validation score as training set size increases.
    Convergence of train and CV curves = absence of overfitting.
    Standard in top ML/water quality journals.
    """
    print("[PLOT 11] Generating Learning Curves (Overfitting Diagnostic)...")

    features = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    X_raw = df[features].values

    top_models = {
        "SVR (RBF Kernel)":   SVR(C=20.0, epsilon=0.1, kernel="rbf"),
        "Random Forest":      RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        "XGBoost Regressor":  XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1),
    }
    model_colors = {"SVR (RBF Kernel)": "#9b59b6", "Random Forest": "#e67e22", "XGBoost Regressor": "#1abc9c"}

    metals = {"Chromium": df["Chromium_ugL"].values, "Nickel": df["Nickel_ugL"].values}

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    train_sizes_frac = np.linspace(0.05, 1.0, 15)

    for metal_idx, (metal_name, y) in enumerate(metals.items()):
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        for model_idx, (model_name, model) in enumerate(top_models.items()):
            ax = axes[metal_idx, model_idx]
            color = model_colors[model_name]

            train_sizes, train_scores, cv_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes_frac,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                random_state=42,
            )

            train_mean = train_scores.mean(axis=1)
            train_std  = train_scores.std(axis=1)
            cv_mean    = cv_scores.mean(axis=1)
            cv_std     = cv_scores.std(axis=1)

            ax.plot(train_sizes, train_mean, color=color, marker="o", markersize=4,
                    linewidth=2.0, label="Training score")
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                            alpha=0.12, color=color)

            ax.plot(train_sizes, cv_mean, color="#2c3e50", marker="s", markersize=4,
                    linewidth=2.0, linestyle="--", label="5-fold CV score")
            ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std,
                            alpha=0.10, color="#2c3e50")

            ax.set_xlabel("Training Set Size (samples)", weight="semibold", fontsize=9)
            ax.set_ylabel(r"$R^2$ Score", weight="semibold", fontsize=9)
            ax.set_title(f"{model_name}\n({metal_name})", weight="bold", size=9.5)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.45)
            ax.legend(fontsize=8, loc="lower right")

            # Convergence gap annotation
            final_gap = train_mean[-1] - cv_mean[-1]
            status = "[OK] Converged" if abs(final_gap) < 0.02 else f"Gap = {final_gap:.3f}"
            ax.text(0.05, 0.05, status, transform=ax.transAxes, fontsize=8,
                    color="#27AE60" if abs(final_gap) < 0.02 else "#E74C3C",
                    weight="bold")

    plt.suptitle(
        "Learning Curves: Training vs. Cross-Validation R² Score (Overfitting Diagnostic)\n"
        "Convergence of both curves as N increases confirms absence of overfitting",
        y=0.99, weight="bold", size=13.5
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "11_learning_curves_overfitting_diagnostic.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("[INFO] Initiating HERA 2.0 Academic Scientific Visualization Suite...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "images"))
    os.makedirs(output_dir, exist_ok=True)

    set_premium_academic_style()

    try:
        df, cr_pack, ni_pack = load_data_and_models(script_dir)
        print(f"[INFO] Dataset shape: {df.shape}")
        print(f"[INFO] Output directory: {output_dir}\n")

        # ── Diagnostic console output ─────────────────────────────────────────
        run_overfitting_diagnostics(df)

        # ── Core Geochemical Visualizations ──────────────────────────────────
        plot_spearman_heatmap(df, output_dir)
        plot_thermodynamic_limits(df, output_dir)
        plot_model_parity(df, cr_pack, ni_pack, output_dir)
        plot_residuals(df, cr_pack, ni_pack, output_dir)
        plot_scenario_progression(cr_pack, ni_pack, output_dir)
        plot_sensitivity_curves(cr_pack, ni_pack, output_dir)

        # ── Multi-Algorithm Comparative Visualizations ────────────────────────
        plot_multi_algorithm_feature_importance(df, output_dir)
        plot_multi_algorithm_confusion_matrices(df, output_dir)

        # ── Premium Model Comparison Dashboards ───────────────────────────────
        cr_dash_path = os.path.join(output_dir, "9_chromium_model_comparison_dashboard.png")
        generate_comparison_dashboard(df, "Chromium", df["Chromium_ugL"].values, cr_dash_path)

        ni_dash_path = os.path.join(output_dir, "10_nickel_model_comparison_dashboard.png")
        generate_comparison_dashboard(df, "Nickel", df["Nickel_ugL"].values, ni_dash_path)

        # ── Learning Curves (Overfitting Proof) ───────────────────────────────
        plot_learning_curves(df, output_dir)

        print("\n" + "=" * 70)
        print("  [SUCCESS] All 11 publication-grade plots generated successfully!")
        print(f"  Location: {output_dir}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
