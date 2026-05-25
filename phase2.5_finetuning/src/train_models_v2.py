"""
train_models_v2.py  —  HERA 2.0  Phase 2.5  Area 3
====================================================
Retrain ALL 5 algorithms on dataset v2 (15,000 samples, 9 features)
using best hyperparameters from Area 2 (Optuna TPE).

Evaluation:
  - 80/20 holdout split  (final test metric)
  - 5-Fold Cross-Validation  (mean +/- std, journal-grade reporting)

Output:
  - phase2.5_finetuning/models/best_model_chromium_v2.pkl
  - phase2.5_finetuning/models/best_model_nickel_v2.pkl
  - phase2.5_finetuning/results/reports/benchmark_v2.csv
  - phase2.5_finetuning/results/reports/overfitting_diagnostics_v2.txt
"""

import os
import sys
import json
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset",
                                             "dataset_heavy_metal_grounded_v2.csv"))
params_path  = os.path.abspath(os.path.join(script_dir, "..", "results", "reports",
                                             "best_params.json"))
report_dir   = os.path.abspath(os.path.join(script_dir, "..", "results", "reports"))
models_dir   = os.path.abspath(os.path.join(script_dir, "..", "models"))
os.makedirs(report_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

SEED     = 42
N_FOLDS  = 5

RAW_FEATURES     = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
DERIVED_FEATURES = ["pH_squared", "pH_EC_interact", "log_EC", "pOH_proxy", "pH_temp_interact"]
ALL_FEATURES     = RAW_FEATURES + DERIVED_FEATURES

# ── Load data & best params ────────────────────────────────────────────────
print("=" * 72)
print("  HERA 2.0  Phase 2.5 — Area 3: Model Retraining with Best Params")
print("=" * 72)

if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset v2 not found: {dataset_path}")
    sys.exit(1)
if not os.path.exists(params_path):
    print(f"[ERROR] best_params.json not found: {params_path}")
    print("        Run tune_hyperparams.py (Area 2) first.")
    sys.exit(1)

df = pd.read_csv(dataset_path)
print(f"\n[INFO] Dataset v2 loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

with open(params_path) as f:
    best_params = json.load(f)
print(f"[INFO] Best hyperparameters loaded from: {params_path}\n")

# ── Build model definitions with Optuna best params ────────────────────────
def get_models(metal: str) -> dict:
    """Return 5-algorithm dict with best hyperparameters for the given metal."""
    rf_key  = f"Random_Forest_{metal}"
    xgb_key = f"XGBoost_{metal}"

    rf_params  = best_params[rf_key]["best_params"]
    xgb_params = best_params[xgb_key]["best_params"]

    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=10.0),
        "SVR (RBF Kernel)":  SVR(C=20.0, epsilon=0.1, kernel="rbf"),
        "Random Forest":     RandomForestRegressor(
                                 **rf_params,
                                 random_state=SEED, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(
                                 **xgb_params,
                                 random_state=SEED, n_jobs=-1, verbosity=0),
    }

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN & EVALUATE ONE TARGET
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(metal: str, y: np.ndarray, X_raw: np.ndarray) -> list:
    """
    Full training pipeline for one metal target:
      1. StandardScaler fit on train set
      2. 80/20 holdout evaluation (Train R2, Test R2, RMSE, MAE, MAPE, gap)
      3. 5-Fold CV on full dataset (mean +/- std)
    Returns list of metric dicts (one per model).
    """
    print(f"\n{'='*72}")
    print(f"  Training pipeline: {metal}")
    print(f"{'='*72}")

    X_train_r, X_test_r, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_r)
    X_test  = scaler.transform(X_test_r)
    X_full  = scaler.transform(X_raw)          # for 5-Fold CV

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    models = get_models(metal)

    results = []
    best_test_r2  = -np.inf
    best_model_obj = None
    best_model_name = ""

    print(f"\n  {'Model':<22} {'Train R2':>9} {'Test R2':>9} {'Gap':>8} "
          f"{'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'CV R2 mean':>11} {'CV R2 std':>10}")
    print("  " + "-" * 97)

    for name, model in models.items():
        t0 = time.time()

        # ── Holdout ──────────────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        r2_train  = r2_score(y_train, y_train_pred)
        r2_test   = r2_score(y_test,  y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test  = mean_absolute_error(y_test, y_test_pred)
        mape_test = np.mean(np.abs((y_test - y_test_pred) /
                                   np.maximum(y_test, 1e-6))) * 100.0
        gap       = (r2_train - r2_test) * 100.0

        # ── 5-Fold CV ─────────────────────────────────────────────────────
        # Refit a fresh model instance for CV (same hyperparams)
        cv_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        cv_scores = cross_val_score(cv_model, X_full, y,
                                    cv=kf, scoring="r2", n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()

        elapsed = time.time() - t0

        print(f"  {name:<22} {r2_train:>9.5f} {r2_test:>9.5f} {gap:>+7.2f}%"
              f" {rmse_test:>8.3f} {mae_test:>8.3f} {mape_test:>7.2f}%"
              f" {cv_mean:>11.5f} {cv_std:>10.5f}")

        verdict = "Safe" if abs(gap) < 2.0 else ("Caution" if abs(gap) < 5.0 else "Overfit")

        results.append({
            "Metal":         metal,
            "Model":         name,
            "Train R2":      round(r2_train, 6),
            "Test R2":       round(r2_test,  6),
            "R2 Gap (%)":    round(gap, 4),
            "Test RMSE":     round(rmse_test, 4),
            "Test MAE":      round(mae_test,  4),
            "Test MAPE (%)": round(mape_test, 4),
            "CV R2 Mean":    round(cv_mean, 6),
            "CV R2 Std":     round(cv_std,  6),
            "CV R2 Folds":   [round(s, 6) for s in cv_scores],
            "Verdict":       verdict,
            "Elapsed (s)":   round(elapsed, 2),
        })

        if r2_test > best_test_r2:
            best_test_r2   = r2_test
            best_model_obj  = model
            best_model_name = name
            best_scaler     = scaler

    # ── Serialize best model ──────────────────────────────────────────────
    model_pack = {
        "model":       best_model_obj,
        "scaler":      best_scaler,
        "features":    ALL_FEATURES,
        "metal":       metal,
        "best_model":  best_model_name,
        "test_r2":     round(best_test_r2, 6),
        "version":     "v2",
    }
    pkl_name = f"best_model_{'chromium' if metal == 'Chromium' else 'nickel'}_v2.pkl"
    pkl_path = os.path.join(models_dir, pkl_name)
    with open(pkl_path, "wb") as f:
        pickle.dump(model_pack, f)

    print(f"\n  [BEST] {best_model_name}  (Test R2 = {best_test_r2:.5f})")
    print(f"  [OK] Serialized -> {pkl_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    X_raw = df[ALL_FEATURES].values
    y_cr  = df["Chromium_ugL"].values
    y_ni  = df["Nickel_ugL"].values

    all_results = []
    all_results += train_and_evaluate("Chromium", y_cr, X_raw)
    all_results += train_and_evaluate("Nickel",   y_ni, X_raw)

    # ── Save benchmark CSV ────────────────────────────────────────────────
    df_out = pd.DataFrame(all_results)
    # Drop CV folds column for CSV readability
    df_csv = df_out.drop(columns=["CV R2 Folds"])
    csv_path = os.path.join(report_dir, "benchmark_v2.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"\n[OK] Benchmark CSV saved -> {csv_path}")

    # ── Overfitting Diagnostic Report ────────────────────────────────────
    lines = []
    lines.append("=" * 72)
    lines.append("  HERA 2.0  Phase 2.5 — Overfitting Diagnostic Report (v2)")
    lines.append(f"  Dataset   : v2 (15,000 samples, 9 features)")
    lines.append(f"  Eval      : 80/20 holdout + 5-Fold CV")
    lines.append("=" * 72)

    for metal in ["Chromium", "Nickel"]:
        rows = [r for r in all_results if r["Metal"] == metal]
        lines.append(f"\n  {metal} ({('50' if metal == 'Chromium' else '20')} ug/L WHO limit)")
        lines.append("  " + "-" * 70)
        lines.append(f"  {'Model':<22} {'Train R2':>9} {'Test R2':>9} {'Gap':>8}"
                     f" {'CV mean':>9} {'CV std':>9} {'Verdict':>10}")
        lines.append("  " + "-" * 70)
        for r in rows:
            lines.append(
                f"  {r['Model']:<22} {r['Train R2']:>9.5f} {r['Test R2']:>9.5f}"
                f" {r['R2 Gap (%)']:>+7.2f}%"
                f" {r['CV R2 Mean']:>9.5f} {r['CV R2 Std']:>9.5f}"
                f" {r['Verdict']:>10}"
            )

    lines.append("\n  Comparison vs Phase 2 Baseline:")
    lines.append("  " + "-" * 70)
    baseline = {
        ("Chromium", "Random Forest"):     (0.9587, 2.50),
        ("Chromium", "XGBoost Regressor"): (0.9650, 1.75),
        ("Nickel",   "Random Forest"):     (0.9446, 2.50),
        ("Nickel",   "XGBoost Regressor"): (0.9479, 3.21),
    }
    for r in all_results:
        key = (r["Metal"], r["Model"])
        if key in baseline:
            v1_r2, v1_gap = baseline[key]
            delta_r2  = (r["Test R2"]     - v1_r2)  * 100
            delta_gap = r["R2 Gap (%)"] - v1_gap
            lines.append(
                f"  {r['Metal']:<10} {r['Model']:<22}"
                f"  v1 R2={v1_r2:.4f} gap={v1_gap:.2f}%"
                f"  ->  v2 R2={r['Test R2']:.5f} gap={r['R2 Gap (%)']:+.2f}%"
                f"  (dR2={delta_r2:+.2f}%  dgap={delta_gap:+.2f}%)"
            )

    lines.append("\n" + "=" * 72)
    report_text = "\n".join(lines)
    print("\n" + report_text)

    diag_path = os.path.join(report_dir, "overfitting_diagnostics_v2.txt")
    with open(diag_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\n[OK] Diagnostics saved -> {diag_path}")

    print("\n" + "=" * 72)
    print("  [SUCCESS] Area 3 complete! All models retrained & serialized.")
    print(f"  Models saved in: {models_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
