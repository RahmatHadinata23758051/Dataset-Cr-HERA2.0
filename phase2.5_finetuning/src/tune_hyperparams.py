"""
tune_hyperparams.py  —  HERA 2.0  Phase 2.5 Area 2
=====================================================
Bayesian Hyperparameter Optimization using Optuna (TPE Sampler)
with 5-Fold Cross-Validation as the objective function.

Targets :  Random Forest  &  XGBoost  for BOTH Chromium and Nickel
Dataset  :  dataset_heavy_metal_grounded_v2.csv  (15,000 samples, 9 features)
Trials   :  100 per model (4 tuning jobs total)
Output   :  phase2.5_finetuning/results/reports/best_params.json
            phase2.5_finetuning/results/reports/tuning_summary.txt
"""

import os
import sys
import json
import time
import warnings
import logging
warnings.filterwarnings("ignore")

# Suppress Optuna info logs — only show warnings+
logging.getLogger("optuna").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("[ERROR] Optuna not found. Run: pip install optuna")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset",
                                             "dataset_heavy_metal_grounded_v2.csv"))
report_dir   = os.path.abspath(os.path.join(script_dir, "..", "results", "reports"))
os.makedirs(report_dir, exist_ok=True)

JSON_OUT = os.path.join(report_dir, "best_params.json")
TXT_OUT  = os.path.join(report_dir, "tuning_summary.txt")

N_TRIALS  = 100
N_FOLDS   = 5
SEED      = 42

# Feature sets for v2 dataset
RAW_FEATURES     = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
DERIVED_FEATURES = ["pH_squared", "pH_EC_interact", "log_EC", "pOH_proxy", "pH_temp_interact"]
ALL_FEATURES     = RAW_FEATURES + DERIVED_FEATURES

# ── Load Data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("  HERA 2.0  Phase 2.5 — Area 2: Bayesian Hyperparameter Tuning")
print(f"  Optuna TPE Sampler  |  {N_TRIALS} trials  |  {N_FOLDS}-Fold CV")
print("=" * 70)

if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset v2 not found at: {dataset_path}")
    print("        Run generate_dataset.py (Area 1) first.")
    sys.exit(1)

df = pd.read_csv(dataset_path)
print(f"\n[INFO] Loaded dataset v2: {df.shape[0]:,} rows x {df.shape[1]} cols")

X_raw = df[ALL_FEATURES].values
y_cr  = df["Chromium_ugL"].values
y_ni  = df["Nickel_ugL"].values

# Scale features once (5-Fold CV done on scaled data)
scaler  = StandardScaler()
X       = scaler.fit_transform(X_raw)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def objective_rf(trial, X, y):
    """Random Forest objective — maximise mean 5-Fold CV R²."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 300),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "max_features":      trial.suggest_categorical("max_features",
                                                        ["sqrt", "log2", 0.5, 0.7]),
        "ccp_alpha":         trial.suggest_float("ccp_alpha", 0.0, 0.05),
        "random_state":      SEED,
        "n_jobs":            -1,
    }
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    return scores.mean()


def objective_xgb(trial, X, y):
    """XGBoost objective — maximise mean 5-Fold CV R²."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
        "max_depth":         trial.suggest_int("max_depth", 2, 6),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 30),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "random_state":      SEED,
        "n_jobs":            -1,
        "verbosity":         0,
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    return scores.mean()


# ══════════════════════════════════════════════════════════════════════════════
# TUNING RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tuning(model_type: str, metal: str, X, y) -> dict:
    """Run N_TRIALS Bayesian optimisation trials, return best params + stats."""
    label = f"{model_type} — {metal}"
    print(f"\n{'─'*70}")
    print(f"  Tuning: {label}")
    print(f"  Features: {X.shape[1]}  |  Samples: {X.shape[0]:,}  |  Target: {metal}")
    print(f"{'─'*70}")

    objective = objective_rf if model_type == "Random Forest" else objective_xgb

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        study_name=f"{model_type}_{metal}",
    )

    t_start = time.time()

    # Progress callback — print every 20 trials
    def callback(study, trial):
        if (trial.number + 1) % 20 == 0 or trial.number == 0:
            elapsed = time.time() - t_start
            print(f"    Trial {trial.number+1:>3}/{N_TRIALS}"
                  f"  |  Best CV R2 = {study.best_value:.5f}"
                  f"  |  Elapsed: {elapsed:.0f}s")

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=N_TRIALS,
        callbacks=[callback],
        show_progress_bar=False,
    )

    elapsed = time.time() - t_start

    # Compute final 5-Fold CV stats with best params
    if model_type == "Random Forest":
        best_model = RandomForestRegressor(**study.best_params, random_state=SEED, n_jobs=-1)
    else:
        best_model = XGBRegressor(**study.best_params, random_state=SEED, n_jobs=-1, verbosity=0)

    cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring="r2", n_jobs=-1)

    result = {
        "model":        model_type,
        "metal":        metal,
        "best_cv_r2":   round(study.best_value, 6),
        "cv_r2_mean":   round(float(cv_scores.mean()), 6),
        "cv_r2_std":    round(float(cv_scores.std()), 6),
        "cv_r2_folds":  [round(float(s), 6) for s in cv_scores],
        "n_trials":     N_TRIALS,
        "elapsed_sec":  round(elapsed, 1),
        "best_params":  study.best_params,
    }

    print(f"\n  [RESULT] {label}")
    print(f"    Best CV R2  : {result['cv_r2_mean']:.5f} +/- {result['cv_r2_std']:.5f}")
    print(f"    Folds       : {[round(s,4) for s in cv_scores]}")
    print(f"    Elapsed     : {elapsed:.1f}s")
    print(f"    Best params :")
    for k, v in study.best_params.items():
        print(f"      {k:<22} = {v}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    all_results  = {}
    summary_lines = []

    jobs = [
        ("Random Forest", "Chromium", y_cr),
        ("Random Forest", "Nickel",   y_ni),
        ("XGBoost",       "Chromium", y_cr),
        ("XGBoost",       "Nickel",   y_ni),
    ]

    t_total = time.time()

    for model_type, metal, y in jobs:
        res = run_tuning(model_type, metal, X, y)
        key = f"{model_type.replace(' ', '_')}_{metal}"
        all_results[key] = res

        line = (f"  {model_type:<18} | {metal:<10} | "
                f"CV R2 = {res['cv_r2_mean']:.5f} +/- {res['cv_r2_std']:.5f}  "
                f"({res['elapsed_sec']:.0f}s)")
        summary_lines.append(line)

    total_elapsed = time.time() - t_total

    # ── Save JSON ──────────────────────────────────────────────────────────
    with open(JSON_OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Best params saved -> {JSON_OUT}")

    # ── Save Text Summary ──────────────────────────────────────────────────
    summary = []
    summary.append("=" * 70)
    summary.append("  HERA 2.0 Phase 2.5 — Hyperparameter Tuning Summary")
    summary.append(f"  Optuna TPE | {N_TRIALS} trials | {N_FOLDS}-Fold CV | Seed={SEED}")
    summary.append(f"  Dataset    : dataset_heavy_metal_grounded_v2.csv (15,000 samples)")
    summary.append(f"  Features   : {len(ALL_FEATURES)} ({', '.join(ALL_FEATURES)})")
    summary.append(f"  Total time : {total_elapsed/60:.1f} minutes")
    summary.append("=" * 70)
    summary.append("")
    summary.append("  Model              | Metal      | CV R2 (mean +/- std)")
    summary.append("  " + "-" * 66)
    summary.extend(summary_lines)
    summary.append("")

    # Improvement analysis
    summary.append("  Improvement vs Baseline (Phase 2 original):")
    summary.append("  " + "-" * 66)
    baseline = {
        "Random Forest_Chromium": 0.9587,
        "Random Forest_Nickel":   0.9446,
        "XGBoost_Chromium":       0.9650,
        "XGBoost_Nickel":         0.9479,
    }
    for key, bl in baseline.items():
        model_type, metal = key.split("_", 1)
        res_key = f"{model_type.replace(' ', '_')}_{metal}"
        if res_key in all_results:
            new_r2 = all_results[res_key]["cv_r2_mean"]
            delta  = (new_r2 - bl) * 100
            arrow  = "+" if delta >= 0 else ""
            summary.append(f"    {model_type:<18} | {metal:<10} | "
                           f"v1 Test R2={bl:.4f}  ->  v2 CV R2={new_r2:.5f}  "
                           f"({arrow}{delta:+.2f}%)")

    summary.append("")
    summary.append("  Next step: Run train_models_v2.py (Area 3) to retrain with best params")
    summary.append("=" * 70)

    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    with open(TXT_OUT, "w") as f:
        f.write(summary_text + "\n")
    print(f"\n[OK] Summary saved  -> {TXT_OUT}")

    print("\n" + "=" * 70)
    print("  [SUCCESS] Area 2 complete! Best hyperparameters found.")
    print(f"  Total elapsed: {total_elapsed/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
