#!/usr/bin/env python3
"""
Generate synthetic Chromium (Cr) test datasets for model validation and stress testing.

This module generates two types of test datasets:
1. Scenario Testing: Clean Water, Moderately Polluted, Highly Polluted, Extreme Industrial-like
2. Sensitivity Testing: EC sweep, pH sweep, TDS sweep to validate model robustness

Outputs:
1) Dataset/TestScenarios/synthetic_cr_scenario_test.csv
2) Dataset/TestScenarios/synthetic_cr_sensitivity_test.csv
3) Dataset/TestScenarios/qa_test_scenarios.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def clip(value: float, low: float, high: float) -> float:
    """Clip value to [low, high] range."""
    return max(low, min(high, value))


def compute_cr_geochemical(
    ec: float,
    tds: float,
    ph: float,
    temperature: float = 25.0,
) -> float:
    """
    Compute Cr concentration using hydrogeochemical model.
    
    Parameters:
    - ec: Electrical conductance (μS/cm)
    - tds: Total dissolved solids (mg/L)
    - ph: pH value
    - temperature: Water temperature (°C), default 25°C
    
    Returns:
    - cr: Chromium concentration (μg/L)
    
    Model basis:
    - Mechanism 1: Log-normal base from ionic strength (EC, TDS)
    - Mechanism 2: pH-dependent solubility (Cr(III) exponential inverse)
    - Mechanism 3: Physical constraint (Cr <= 5% of TDS)
    - Mechanism 4: Measurement noise (~0.8 μg/L)
    """
    # Mechanism 1: Base Cr from ionic strength using log-normal distribution
    log_ec = np.log1p(ec)
    log_tds = np.log1p(tds)
    
    a0, a1, a2 = 1.2, 0.35, 0.45
    cr_base_log = a0 + a1 * log_ec + a2 * log_tds
    cr_base = np.exp(cr_base_log)
    
    # Mechanism 2: pH effect - exponential inverse (Cr solubility ↑ in acidic)
    # Cr(III) precipitates at pH > 6.5, highly soluble at pH < 5.5
    b_acidic = 0.8
    b_neutral = 0.15
    
    if ph < 7.0:
        pH_factor = np.exp(b_acidic * (7.0 - ph))
    else:
        pH_factor = 1.0 - b_neutral * (ph - 7.0)
        pH_factor = max(0.1, pH_factor)
    
    cr = cr_base * pH_factor
    
    # Mechanism 3: Physical constraint - Cr <= 5% of TDS
    cr_max_physical = 0.05 * tds
    cr = min(cr, cr_max_physical)
    
    # Mechanism 4: Add measurement uncertainty
    sigma_noise = 0.8
    cr += np.random.normal(0, sigma_noise)
    
    return max(0.1, cr)  # Ensure positive value


def generate_scenario_dataset(
    n_per_scenario: int = 75,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic dataset with 4 distinct test scenarios.
    
    Scenarios:
    1. Clean Water: Low EC/TDS, neutral pH
    2. Moderately Polluted: Medium EC/TDS, slightly acidic pH
    3. Highly Polluted: High EC/TDS, acidic pH
    4. Extreme Industrial-like: Very high EC/TDS, extreme pH (very acidic or basic)
    """
    np.random.seed(seed)
    
    scenarios = {
        "Clean Water": {
            "ec_range": (80.0, 250.0),
            "tds_range": (40.0, 130.0),
            "ph_range": (6.8, 7.8),
            "wt_range": (18.0, 28.0),
        },
        "Moderately Polluted": {
            "ec_range": (350.0, 800.0),
            "tds_range": (180.0, 420.0),
            "ph_range": (6.0, 7.2),
            "wt_range": (15.0, 32.0),
        },
        "Highly Polluted": {
            "ec_range": (1000.0, 2500.0),
            "tds_range": (500.0, 1300.0),
            "ph_range": (4.8, 6.5),
            "wt_range": (12.0, 35.0),
        },
        "Extreme Industrial-like": {
            "ec_range": (3500.0, 8000.0),
            "tds_range": (1800.0, 4200.0),
            "ph_range": (2.5, 3.5),  # Very acidic (mine water, acid industrial waste)
            "wt_range": (8.0, 40.0),
        },
    }
    
    rows = []
    start_date = datetime(2026, 1, 1)
    time_slots = ["06:00:00", "09:00:00", "12:00:00", "15:00:00", "18:00:00"]
    
    for scenario_name, params in scenarios.items():
        ec_lo, ec_hi = params["ec_range"]
        tds_lo, tds_hi = params["tds_range"]
        ph_lo, ph_hi = params["ph_range"]
        wt_lo, wt_hi = params["wt_range"]
        
        for _ in range(n_per_scenario):
            day = int(np.random.randint(0, 365))
            d = start_date + timedelta(days=day)
            hm = np.random.choice(time_slots)
            ts = datetime.strptime(
                d.strftime("%Y-%m-%d") + " " + hm,
                "%Y-%m-%d %H:%M:%S"
            )
            
            # EC: log-uniform within scenario range
            ec = float(np.exp(np.random.uniform(np.log(ec_lo), np.log(ec_hi))))
            
            # TDS: proportional to EC with factor + noise
            k = np.random.uniform(0.50, 0.78)
            tds = ec * k * np.random.normal(1.0, 0.06)
            tds = clip(tds, tds_lo, tds_hi)
            
            # pH: within scenario range
            ph = np.random.normal((ph_lo + ph_hi) / 2.0, (ph_hi - ph_lo) / 5.0)
            ph = clip(ph, ph_lo, ph_hi)
            
            # Water temperature
            wt = np.random.uniform(wt_lo, wt_hi)
            
            # Air temperature (higher than water)
            at = clip(wt + np.random.uniform(1.0, 5.0), wt_lo - 5, wt_hi + 5)
            
            # Humidity (synthetic)
            hum = clip(np.random.normal(70.0, 12.0) - 0.7 * (at - 25.0), 30.0, 95.0)
            
            # Voltage (sensor/battery realistic)
            volt = clip(3.55 + 0.65 * np.random.rand(), 3.45, 4.25)
            
            # Chromium estimate using geochemical model
            cr = compute_cr_geochemical(ec, tds, ph, wt)
            
            rows.append({
                "Scenario": scenario_name,
                "Tanggal": ts.strftime("%Y-%m-%d"),
                "Waktu": ts.strftime("%H:%M:%S"),
                "Tegangan (V)": round(volt, 3),
                "Suhu Air (°C)": round(wt, 2),
                "Suhu Lingkungan (°C)": round(at, 2),
                "Kelembapan Lingkungan (%)": round(hum, 1),
                "TDS": round(tds, 2),
                "EC": round(ec, 2),
                "pH": round(ph, 2),
                "Cr": round(cr, 2),
                "Test_Type": "scenario_test",
                "Sweep_Variable": "",
            })
    
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_sensitivity_dataset(
    base_scenario: str = "Moderately Polluted",
    n_per_sweep: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate sensitivity test datasets: EC sweep, pH sweep, TDS sweep.
    
    Each sweep holds other variables relatively constant while varying one parameter
    to validate model monotonicity and robustness.
    
    Parameters:
    - base_scenario: Reference scenario to build sensitivities from
    - n_per_sweep: Number of points per sweep (EC, pH, TDS)
    - seed: Random seed
    """
    np.random.seed(seed)
    
    # Base parameters for Moderately Polluted scenario
    base_params = {
        "ec_mid": 575.0,
        "tds_mid": 300.0,
        "ph_mid": 6.6,
        "wt_mid": 24.0,
    }
    
    sweep_configs = {
        "EC_Sweep": {
            "variable": "EC",
            "values": np.linspace(150.0, 2000.0, n_per_sweep),
            "tds_factor": 0.60,
            "ph_offset": 0.0,
            "description": "EC from low (150) to high (2000), TDS/pH/T relatively constant",
        },
        "pH_Sweep_Acidic": {
            "variable": "pH",
            "values": np.linspace(3.0, 7.0, n_per_sweep),
            "ec_fixed": base_params["ec_mid"],
            "tds_fixed": base_params["tds_mid"],
            "description": "pH from very acidic (3.0) to neutral (7.0), EC/TDS/T constant",
        },
        "pH_Sweep_Alkaline": {
            "variable": "pH",
            "values": np.linspace(7.0, 9.0, n_per_sweep),
            "ec_fixed": base_params["ec_mid"],
            "tds_fixed": base_params["tds_mid"],
            "description": "pH from neutral (7.0) to alkaline (9.0), EC/TDS/T constant",
        },
        "TDS_Sweep": {
            "variable": "TDS",
            "values": np.linspace(50.0, 1500.0, n_per_sweep),
            "ec_factor": 1.8,  # Inverse of TDS-EC relationship
            "ph_offset": 0.0,
            "description": "TDS from low (50) to high (1500), EC/pH/T relatively constant",
        },
    }
    
    rows = []
    start_date = datetime(2026, 1, 1)
    time_slots = ["12:00:00"]  # Single time for sensitivity tests
    
    for sweep_name, config in sweep_configs.items():
        sweep_values = config["values"]
        
        for idx, sweep_value in enumerate(sweep_values):
            ts = start_date + timedelta(days=idx)
            hm = time_slots[0]
            ts = datetime.strptime(
                ts.strftime("%Y-%m-%d") + " " + hm,
                "%Y-%m-%d %H:%M:%S"
            )
            
            # Determine EC, TDS, pH based on sweep type
            if config["variable"] == "EC":
                ec = sweep_value
                tds = ec * config["tds_factor"] * np.random.normal(1.0, 0.02)
                ph = clip(base_params["ph_mid"] + config["ph_offset"], 3.0, 9.5)
            
            elif config["variable"] == "pH":
                ph = sweep_value
                ec = config["ec_fixed"]
                tds = config["tds_fixed"]
            
            elif config["variable"] == "TDS":
                tds = sweep_value
                ec = tds * config["ec_factor"] * np.random.normal(1.0, 0.02)
                ph = clip(base_params["ph_mid"] + config["ph_offset"], 3.0, 9.5)
            
            # Clip to valid ranges
            ec = clip(ec, 50.0, 5000.0)
            tds = clip(tds, 25.0, 2600.0)
            ph = clip(ph, 2.5, 9.5)
            
            # Temperature (relatively constant with small noise)
            wt = base_params["wt_mid"] + np.random.normal(0, 0.5)
            at = clip(wt + np.random.uniform(1.0, 3.0), 10.0, 40.0)
            hum = clip(70.0 + np.random.normal(0, 3.0), 30.0, 95.0)
            volt = clip(3.8 + np.random.normal(0, 0.05), 3.45, 4.25)
            
            # Chromium
            cr = compute_cr_geochemical(ec, tds, ph, wt)
            
            rows.append({
                "Scenario": base_scenario,
                "Tanggal": ts.strftime("%Y-%m-%d"),
                "Waktu": ts.strftime("%H:%M:%S"),
                "Tegangan (V)": round(volt, 3),
                "Suhu Air (°C)": round(wt, 2),
                "Suhu Lingkungan (°C)": round(at, 2),
                "Kelembapan Lingkungan (%)": round(hum, 1),
                "TDS": round(tds, 2),
                "EC": round(ec, 2),
                "pH": round(ph, 2),
                "Cr": round(cr, 2),
                "Test_Type": "sensitivity_test",
                "Sweep_Variable": config["variable"],
            })
    
    df = pd.DataFrame(rows)
    return df


def compute_qa_statistics(
    scenario_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> dict:
    """
    Compute QA statistics for test datasets.
    
    Returns:
    - Dictionary with scenario statistics, correlations, and trend analysis
    """
    qa_stats = {}
    
    # 1. Scenario test statistics
    qa_stats["scenario_counts"] = scenario_df["Scenario"].value_counts().to_dict()
    
    scenario_summary = []
    for scenario in scenario_df["Scenario"].unique():
        subset = scenario_df[scenario_df["Scenario"] == scenario]
        cr_values = subset["Cr"]
        scenario_summary.append({
            "Scenario": scenario,
            "Count": len(subset),
            "Cr_Min": float(cr_values.min()),
            "Cr_Max": float(cr_values.max()),
            "Cr_Mean": float(cr_values.mean()),
            "Cr_Std": float(cr_values.std()),
        })
    qa_stats["scenario_summary"] = pd.DataFrame(scenario_summary)
    
    # 2. Correlation analysis (combined scenario + sensitivity)
    all_df = pd.concat([scenario_df, sensitivity_df], ignore_index=True)
    corr_data = all_df[["EC", "TDS", "pH", "Cr"]].corr()["Cr"]
    qa_stats["correlations"] = {
        "corr_Cr_EC": float(corr_data["EC"]),
        "corr_Cr_TDS": float(corr_data["TDS"]),
        "corr_Cr_pH": float(corr_data["pH"]),
    }
    
    # 3. Sensitivity sweep trend analysis
    trend_summary = []
    for sweep_var in sensitivity_df["Sweep_Variable"].unique():
        if sweep_var == "":
            continue
        
        subset = sensitivity_df[sensitivity_df["Sweep_Variable"] == sweep_var].sort_values(sweep_var)
        
        # Calculate monotonic trend (Spearman correlation)
        if len(subset) > 2:
            from scipy.stats import spearmanr
            corr, p_value = spearmanr(subset[sweep_var], subset["Cr"])
        else:
            corr, p_value = np.nan, np.nan
        
        trend_summary.append({
            "Sweep_Variable": sweep_var,
            "N_Points": len(subset),
            f"{sweep_var}_Min": float(subset[sweep_var].min()),
            f"{sweep_var}_Max": float(subset[sweep_var].max()),
            "Cr_Min": float(subset["Cr"].min()),
            "Cr_Max": float(subset["Cr"].max()),
            "Spearman_Correlation": corr if not np.isnan(corr) else 0.0,
            "Trend_Direction": (
                "Monotonic_Increase" if corr > 0.5 else
                "Monotonic_Decrease" if corr < -0.5 else
                "Weak_Trend"
            ),
        })
    qa_stats["sensitivity_trends"] = pd.DataFrame(trend_summary)
    
    return qa_stats


def write_outputs(
    base_dir: Path,
    scenario_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    qa_stats: dict,
) -> dict:
    """
    Write test datasets and QA summary to CSV files.
    """
    out_dir = base_dir / "Dataset" / "TestScenarios"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    main_cols = [
        "Scenario",
        "Tanggal",
        "Waktu",
        "Tegangan (V)",
        "Suhu Air (°C)",
        "Suhu Lingkungan (°C)",
        "Kelembapan Lingkungan (%)",
        "TDS",
        "EC",
        "pH",
        "Cr",
    ]
    
    # Write scenario test
    f_scenario = out_dir / "synthetic_cr_scenario_test.csv"
    scenario_df[main_cols + ["Test_Type", "Sweep_Variable"]].to_csv(f_scenario, index=False)
    
    # Write sensitivity test
    f_sensitivity = out_dir / "synthetic_cr_sensitivity_test.csv"
    sensitivity_df[main_cols + ["Test_Type", "Sweep_Variable"]].to_csv(f_sensitivity, index=False)
    
    # Combine scenario summary and sensitivity trends for QA export
    f_qa = out_dir / "qa_test_scenarios.csv"
    
    # Write scenario summary separately since it has different structure
    qa_stats["scenario_summary"].to_csv(out_dir / "qa_scenario_summary.csv", index=False)
    qa_stats["sensitivity_trends"].to_csv(out_dir / "qa_sensitivity_trends.csv", index=False)
    
    # Write correlations summary
    corr_df = pd.DataFrame([qa_stats["correlations"]])
    corr_df.to_csv(out_dir / "qa_correlations.csv", index=False)
    
    return {
        "scenario": f_scenario,
        "sensitivity": f_sensitivity,
        "qa_scenario": out_dir / "qa_scenario_summary.csv",
        "qa_sensitivity": out_dir / "qa_sensitivity_trends.csv",
        "qa_correlations": out_dir / "qa_correlations.csv",
    }


def summarize_qa(qa_stats: dict) -> None:
    """Print QA summary for visual validation."""
    print("\n" + "=" * 70)
    print("QA SUMMARY: TEST SCENARIOS")
    print("=" * 70)
    
    print("\n[1] SCENARIO COUNTS & STATISTICS")
    print("-" * 70)
    print(qa_stats["scenario_summary"].to_string(index=False))
    
    print("\n[2] CORRELATION ANALYSIS (All Test Data)")
    print("-" * 70)
    for k, v in qa_stats["correlations"].items():
        print(f"  {k}: {v:.6f}")
    
    print("\n[3] SENSITIVITY SWEEP TRENDS")
    print("-" * 70)
    print(qa_stats["sensitivity_trends"].to_string(index=False))
    
    print("\n[4] TREND INTERPRETATION")
    print("-" * 70)
    for _, row in qa_stats["sensitivity_trends"].iterrows():
        sweep_var = row["Sweep_Variable"]
        trend = row["Trend_Direction"]
        corr = row["Spearman_Correlation"]
        print(f"  {sweep_var:20s}: {trend:25s} (Spearman r = {corr:+.4f})")
    
    print("\n[5] EXPECTED PHYSICAL BEHAVIOUR")
    print("-" * 70)
    print("  ✓ EC increase   → Cr should increase (positive correlation)")
    print("  ✓ TDS increase  → Cr should increase (positive correlation)")
    print("  ✓ pH decrease   → Cr should increase (more acidic = more Cr solubility)")
    print("=" * 70 + "\n")


def main() -> None:
    """
    Main orchestration: generate scenarios, sensitivities, compute QA, write outputs.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic Cr test datasets for model validation."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--scenario-rows",
        type=int,
        default=75,
        help="Rows per scenario (default: 75, total ~300)",
    )
    parser.add_argument(
        "--sweep-points",
        type=int,
        default=30,
        help="Points per sensitivity sweep (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    
    print(f"Generating test scenarios...")
    scenario_df = generate_scenario_dataset(
        n_per_scenario=args.scenario_rows,
        seed=args.seed,
    )
    
    print(f"Generating sensitivity sweeps...")
    sensitivity_df = generate_sensitivity_dataset(
        n_per_sweep=args.sweep_points,
        seed=args.seed + 1,
    )
    
    print(f"Computing QA statistics...")
    qa_stats = compute_qa_statistics(scenario_df, sensitivity_df)
    
    print(f"Writing outputs...")
    outputs = write_outputs(base_dir, scenario_df, sensitivity_df, qa_stats)
    
    # Print console summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Scenario test rows: {len(scenario_df)}")
    print(f"Sensitivity test rows: {len(sensitivity_df)}")
    print(f"Total test rows: {len(scenario_df) + len(sensitivity_df)}")
    
    print("\nRows per scenario (Scenario Test):")
    for scenario, count in qa_stats["scenario_counts"].items():
        print(f"  - {scenario}: {count}")
    
    summarize_qa(qa_stats)
    
    print("Output files written:")
    for key, path in outputs.items():
        print(f"  - {path}")


if __name__ == "__main__":
    main()
