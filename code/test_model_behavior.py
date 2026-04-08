#!/usr/bin/env python3
"""
Behavior Validation Test for Chromium Soft Sensor Model.

Tests whether the trained ML model exhibits consistent and physically-reasonable
behavior when subjected to scenario tests and parameter sensitivity sweeps.

Focus: Consistency, monotonic trends, stability - NOT accuracy metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_model(model_path: Path, scaler_path: Path | None = None) -> tuple:
    """Load trained model and optional scaler."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    scaler = None
    if scaler_path and scaler_path.exists():
        print(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
    
    return model, scaler


def load_data(
    scenario_path: Path,
    sensitivity_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load scenario and sensitivity test datasets."""
    print(f"Loading scenario test from {scenario_path}...")
    scenario_df = pd.read_csv(scenario_path)
    
    print(f"Loading sensitivity test from {sensitivity_path}...")
    sensitivity_df = pd.read_csv(sensitivity_path)
    
    return scenario_df, sensitivity_df


def run_inference(
    model,
    scaler,
    data: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """Run inference on test data and add predictions to dataframe."""
    print(f"Running inference on {len(data)} samples...")
    
    x = data[feature_cols].values
    
    # Scale if scaler exists
    if scaler:
        x = scaler.transform(x)
    
    predictions = model.predict(x)
    data = data.copy()
    data["Cr_predicted"] = predictions
    
    return data


def analyze_scenarios(scenario_df: pd.DataFrame) -> dict:
    """
    Analyze scenario test results.
    
    Validates that Cr predictions increase monotonically from Clean → Extreme.
    """
    print("\n" + "=" * 70)
    print("SCENARIO TEST ANALYSIS")
    print("=" * 70)
    
    scenario_order = ["Clean Water", "Moderately Polluted", "Highly Polluted", "Extreme Industrial-like"]
    results = []
    predictions_per_scenario = {}
    
    for scenario in scenario_order:
        subset = scenario_df[scenario_df["Scenario"] == scenario]
        cr_pred = subset["Cr_predicted"]
        
        stats = {
            "Scenario": scenario,
            "Count": len(subset),
            "Cr_Min": float(cr_pred.min()),
            "Cr_Max": float(cr_pred.max()),
            "Cr_Mean": float(cr_pred.mean()),
            "Cr_Std": float(cr_pred.std()),
        }
        results.append(stats)
        predictions_per_scenario[scenario] = float(cr_pred.mean())
        
        print(f"\n{scenario:25s}: Mean={stats['Cr_Mean']:7.2f} μg/L, "
              f"Min={stats['Cr_Min']:7.2f}, Max={stats['Cr_Max']:7.2f}")
    
    # Validate monotonic order
    means = list(predictions_per_scenario.values())
    is_monotonic = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    
    status = "✓ PASS" if is_monotonic else "✗ FAIL"
    print(f"\nMonotonic Order Check: {status}")
    if not is_monotonic:
        print("  Expected: Clean < Moderately < Highly < Extreme")
        print(f"  Got: {[f'{m:.1f}' for m in means]}")
    
    return {
        "results_df": pd.DataFrame(results),
        "is_monotonic": is_monotonic,
        "predictions": predictions_per_scenario,
    }


def analyze_sensitivity(sensitivity_df: pd.DataFrame) -> dict:
    """
    Analyze sensitivity sweep results.
    
    For each sweep variable (EC, TDS, pH), compute Spearman correlation
    with predicted Cr and classify trend strength.
    """
    print("\n" + "=" * 70)
    print("SENSITIVITY TEST ANALYSIS")
    print("=" * 70)
    
    sweep_variables = ["EC", "TDS", "pH"]
    results = []
    
    for sweep_var in sweep_variables:
        subset = sensitivity_df[sensitivity_df["Sweep_Variable"] == sweep_var].sort_values(sweep_var)
        
        if len(subset) < 3:
            print(f"\n{sweep_var:10s}: Insufficient data (n<3)")
            continue
        
        # Compute Spearman correlation
        corr, p_value = spearmanr(subset[sweep_var], subset["Cr_predicted"])
        
        # Expected direction
        expected_direction = "positive" if sweep_var in ["EC", "TDS"] else "negative"
        expected_sign = 1.0 if sweep_var in ["EC", "TDS"] else -1.0
        
        # Determine trend strength and direction
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            strength = "Strong"
        elif abs_corr > 0.4:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        actual_direction = "positive" if corr > 0 else "negative"
        is_correct = (corr * expected_sign > 0)
        
        trend_status = "✓ OK" if is_correct else "✗ NOT OK"
        
        results.append({
            "Sweep_Variable": sweep_var,
            "N_Points": len(subset),
            "Spearman_r": float(corr),
            "P_Value": float(p_value),
            "Expected": expected_direction,
            "Actual": actual_direction,
            "Strength": strength,
            "Status": trend_status,
            "Cr_Min": float(subset["Cr_predicted"].min()),
            "Cr_Max": float(subset["Cr_predicted"].max()),
        })
        
        print(f"\n{sweep_var:10s}: r={corr:+.4f} ({strength:8s}) - Expected: {expected_direction:8s}")
        print(f"  {' ' * 10} Actual: {actual_direction:8s} {trend_status}")
    
    return {
        "results_df": pd.DataFrame(results),
        "all_valid": all(r["Status"] == "✓ OK" for r in results),
    }


def check_stability(sensitivity_df: pd.DataFrame) -> dict:
    """
    Check for unexpected spikes or instability in predictions.
    
    Analyzes whether prediction changes are proportional to input changes.
    """
    print("\n" + "=" * 70)
    print("STABILITY CHECK")
    print("=" * 70)
    
    stability_issues = []
    
    for sweep_var in ["EC", "TDS", "pH"]:
        subset = sensitivity_df[sensitivity_df["Sweep_Variable"] == sweep_var].sort_values(sweep_var).reset_index(drop=True)
        
        if len(subset) < 2:
            continue
        
        # Calculate deltas
        input_delta = subset[sweep_var].diff().abs()
        output_delta = subset["Cr_predicted"].diff().abs()
        
        # Normalize by mean to detect disproportionate changes
        input_mean = subset[sweep_var].mean()
        output_mean = subset["Cr_predicted"].mean()
        
        if input_mean > 0 and output_mean > 0:
            relative_change_ratio = (output_delta / output_mean) / (input_delta / input_mean + 1e-6)
            
            # Flag if change ratio is very large (>10x)
            spike_indices = relative_change_ratio > 10.0
            
            if spike_indices.any():
                for idx in spike_indices[spike_indices].index:
                    stability_issues.append({
                        "Sweep_Variable": sweep_var,
                        "Index": int(idx),
                        "Input_Delta": float(input_delta[idx]),
                        "Output_Delta": float(output_delta[idx]),
                        "Change_Ratio": float(relative_change_ratio[idx]),
                        "Status": "SPIKE",
                    })
        
        # Print summary
        spike_count = spike_indices.sum()
        status = f"✓ STABLE (no spikes)" if spike_count == 0 else f"✗ {spike_count} spikes detected"
        print(f"{sweep_var:10s}: {status}")
    
    has_issues = len(stability_issues) > 0
    
    return {
        "has_issues": has_issues,
        "issues_df": pd.DataFrame(stability_issues) if stability_issues else None,
        "issue_count": len(stability_issues),
    }


def plot_results(
    scenario_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots...")
    
    # Plot 1: Scenario Test - Boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    scenario_order = ["Clean Water", "Moderately Polluted", "Highly Polluted", "Extreme Industrial-like"]
    scenario_data = [scenario_df[scenario_df["Scenario"] == s]["Cr_predicted"].values for s in scenario_order]
    
    bp = ax.boxplot(scenario_data, labels=scenario_order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    
    ax.set_ylabel("Predicted Cr (μg/L)", fontsize=11)
    ax.set_title("Model Behavior: Predicted Cr by Scenario", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "scenario_predictions_boxplot.png", dpi=150)
    plt.close()
    
    # Plot 2: EC Sweep
    ec_subset = sensitivity_df[sensitivity_df["Sweep_Variable"] == "EC"].sort_values("EC")
    if len(ec_subset) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ec_subset["EC"], ec_subset["Cr_predicted"], marker="o", linewidth=2, markersize=5, label="Predicted")
        ax.set_xlabel("EC (μS/cm)", fontsize=11)
        ax.set_ylabel("Predicted Cr (μg/L)", fontsize=11)
        ax.set_title("Sensitivity Test: EC Sweep (Expected: Monotonic Increase)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_ec_sweep.png", dpi=150)
        plt.close()
    
    # Plot 3: TDS Sweep
    tds_subset = sensitivity_df[sensitivity_df["Sweep_Variable"] == "TDS"].sort_values("TDS")
    if len(tds_subset) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tds_subset["TDS"], tds_subset["Cr_predicted"], marker="s", linewidth=2, markersize=5, label="Predicted")
        ax.set_xlabel("TDS (mg/L)", fontsize=11)
        ax.set_ylabel("Predicted Cr (μg/L)", fontsize=11)
        ax.set_title("Sensitivity Test: TDS Sweep (Expected: Monotonic Increase)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_tds_sweep.png", dpi=150)
        plt.close()
    
    # Plot 4: pH Sweep (combined acidic + alkaline)
    ph_subset = sensitivity_df[sensitivity_df["Sweep_Variable"] == "pH"].sort_values("pH")
    if len(ph_subset) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ph_subset["pH"], ph_subset["Cr_predicted"], marker="^", linewidth=2, markersize=5, label="Predicted")
        ax.axvline(x=7.0, color="gray", linestyle="--", alpha=0.5, label="pH=7 (neutral)")
        ax.set_xlabel("pH", fontsize=11)
        ax.set_ylabel("Predicted Cr (μg/L)", fontsize=11)
        ax.set_title("Sensitivity Test: pH Sweep (Expected: Negative Correlation)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_ph_sweep.png", dpi=150)
        plt.close()
    
    print("✓ Plots saved to results/testing/plots/")


def write_outputs(
    output_dir: Path,
    scenario_results: dict,
    sensitivity_results: dict,
    stability_results: dict,
    scenario_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    """Save all analysis results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scenario analysis
    scenario_results["results_df"].to_csv(
        output_dir / "test_scenario_analysis.csv",
        index=False
    )
    
    # Save sensitivity analysis
    sensitivity_results["results_df"].to_csv(
        output_dir / "test_sensitivity_analysis.csv",
        index=False
    )
    
    # Save stability report
    if stability_results["issues_df"] is not None:
        stability_results["issues_df"].to_csv(
            output_dir / "stability_issues.csv",
            index=False
        )
    else:
        # Create empty report
        pd.DataFrame([{"Status": "No stability issues detected"}]).to_csv(
            output_dir / "stability_issues.csv",
            index=False
        )
    
    # Save full predictions
    scenario_df.to_csv(output_dir / "scenario_predictions_full.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_predictions_full.csv", index=False)
    
    print(f"✓ Results saved to {output_dir}/")


def print_summary(
    scenario_results: dict,
    sensitivity_results: dict,
    stability_results: dict,
) -> None:
    """Print final validation summary."""
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    
    scenario_pass = scenario_results["is_monotonic"]
    sensitivity_pass = sensitivity_results["all_valid"]
    stability_pass = not stability_results["has_issues"]
    
    print(f"\n[1] Scenario Test (Monotonic Order)")
    print(f"    Status: {'✓ PASS' if scenario_pass else '✗ FAIL'}")
    print(f"    Clean < Moderately < Highly < Extreme: {scenario_pass}")
    
    print(f"\n[2] Sensitivity Test (Trend Direction)")
    for _, row in sensitivity_results["results_df"].iterrows():
        status = row["Status"]
        trend = f"{row['Spearman_r']:+.4f}"
        print(f"    {row['Sweep_Variable']:6s}: {status:8s} ({trend})")
    print(f"    All Correct: {sensitivity_pass}")
    
    print(f"\n[3] Stability Check")
    print(f"    Status: {'✓ PASS' if stability_pass else '✗ FAIL'}")
    if stability_results["has_issues"]:
        print(f"    Issues: {stability_results['issue_count']} spike(s) detected")
    else:
        print(f"    No spikes or unstable jumps detected")
    
    all_valid = scenario_pass and sensitivity_pass and stability_pass
    print(f"\n[4] Overall Model Behavior")
    print(f"    Status: {'✓✓✓ VALID' if all_valid else '✗✗✗ NOT VALID'}")
    
    if all_valid:
        print("\n    Model demonstrates:")
        print("    • Consistent scenario separation (Clean → Extreme)")
        print("    • Correct sensitivity trends (EC+, TDS+, pH-)")
        print("    • Stable predictions without anomalies")
        print("\n    ➜ Soft sensor model is READY for deployment")
    else:
        print("\n    Model exhibits issues:")
        if not scenario_pass:
            print("    • Scenario separation not monotonic")
        if not sensitivity_pass:
            print("    • Incorrect sensitivity trends")
        if not stability_pass:
            print("    • Unstable predictions detected")
        print("\n    ➜ Review model training / feature engineering")
    
    print("=" * 70 + "\n")


def main() -> None:
    """Main orchestration."""
    parser = argparse.ArgumentParser(
        description="Test behavior of trained Chromium soft sensor model."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model_rf_full.pkl",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="models/best_model_scaler_full.pkl",
        help="Path to scaler file (optional)",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    model_path = base_dir / args.model_path
    scaler_path = base_dir / args.scaler_path
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Paths
    scenario_path = base_dir / "Dataset" / "TestScenarios" / "synthetic_cr_scenario_test.csv"
    sensitivity_path = base_dir / "Dataset" / "TestScenarios" / "synthetic_cr_sensitivity_test.csv"
    output_dir = base_dir / "results" / "testing"
    plot_dir = output_dir / "plots"
    
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario test not found: {scenario_path}")
    if not sensitivity_path.exists():
        raise FileNotFoundError(f"Sensitivity test not found: {sensitivity_path}")
    
    # Load
    model, scaler = load_model(model_path, scaler_path)
    scenario_df, sensitivity_df = load_data(scenario_path, sensitivity_path)
    
    # Features used by model
    feature_cols = [
        "EC", "TDS", "pH",
        "Suhu Air (°C)", "Suhu Lingkungan (°C)",
        "Kelembapan Lingkungan (%)", "Tegangan (V)"
    ]
    
    # Run inference
    scenario_df = run_inference(model, scaler, scenario_df, feature_cols)
    sensitivity_df = run_inference(model, scaler, sensitivity_df, feature_cols)
    
    # Analyze
    scenario_results = analyze_scenarios(scenario_df)
    sensitivity_results = analyze_sensitivity(sensitivity_df)
    stability_results = check_stability(sensitivity_df)
    
    # Visualize
    plot_results(scenario_df, sensitivity_df, plot_dir)
    
    # Save outputs
    write_outputs(output_dir, scenario_results, sensitivity_results, stability_results, scenario_df, sensitivity_df)
    
    # Print summary
    print_summary(scenario_results, sensitivity_results, stability_results)


if __name__ == "__main__":
    main()
