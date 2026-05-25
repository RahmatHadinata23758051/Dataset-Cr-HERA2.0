#!/usr/bin/env python3
"""
Real-world Water Quality Dataset - Chromium Prediction Inference.

Load real MWQ (Multi-Variable Water Quality) datasets and apply trained
Soft Sensor model to predict Chromium concentration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_model(model_path: Path, scaler_path: Path | None = None) -> tuple:
    """Load trained model and optional scaler."""
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    scaler = None
    if scaler_path and scaler_path.exists():
        print(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
    
    metadata = load_metadata_for_model(model_path)
    return model, scaler, metadata


def load_metadata_for_model(model_path: Path) -> dict:
    """Load metadata associated with the selected model artifact."""
    metadata_candidates = [
        model_path.with_suffix(".metadata.json"),
        model_path.parent / "best_model_metadata.json",
    ]
    for metadata_path in metadata_candidates:
        if metadata_path.exists():
            print(f"Loading metadata from {metadata_path}...")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata_path.name == "best_model_metadata.json":
                expected_stem = f"best_model_{metadata.get('model_name', metadata.get('model_type', 'rf'))}_{metadata.get('scenario', 'full')}"
                if model_path.stem != expected_stem and model_path.stem != f"model_{metadata.get('model_name', metadata.get('model_type', 'rf'))}_{metadata.get('scenario', 'full')}":
                    raise ValueError(
                        f"Metadata-model mismatch: {metadata_path} indicates {expected_stem}, "
                        f"but selected model is {model_path.name}."
                    )
            return metadata
    raise FileNotFoundError(
        f"No metadata found for model {model_path}. "
        f"Expected one of: {[str(p) for p in metadata_candidates]}"
    )


def process_mwq_dataset(
    csv_path: Path,
    dataset_name: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess MWQ real dataset.
    
    Steps:
    1. Load CSV
    2. Rename Conductivity → EC
    3. Calculate TDS from EC (TDS = EC * 0.64)
    4. Validate required columns (pH, EC, TDS)
    5. Drop missing values
    """
    print(f"\nProcessing {dataset_name}...")
    
    # Load
    df = pd.read_csv(csv_path)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Rename: Handle multiple conductivity column name variants
    if "Conductivity (μS/cm)" in df.columns:
        df = df.rename(columns={"Conductivity (μS/cm)": "EC"})
    elif "Conductivity (μS)" in df.columns:
        df = df.rename(columns={"Conductivity (μS)": "EC"})
    
    if "EC" not in df.columns:
        raise ValueError(
            "Column EC is required for this model but was not found. "
            "Provide EC directly or a recognized conductivity column."
        )

    # Build per-row imputation audit trail
    imputation_flags = pd.DataFrame(index=df.index)

    if "TDS" not in df.columns:
        df["TDS"] = df["EC"] * 0.64
        imputation_flags["imputed_TDS_from_EC_x_0_64"] = True
    else:
        imputation_flags["imputed_TDS_from_EC_x_0_64"] = False
    
    # Handle turbidity column variations
    if "Turbidity (FNU)" not in df.columns and "Turbidity (NTU)" in df.columns:
        df = df.rename(columns={"Turbidity (NTU)": "Turbidity (FNU)"})
    elif "Turbidity (FNU)" not in df.columns and "Turbidity (NTU)" not in df.columns:
        # Default synthetic turbidity if not present
        df["Turbidity (FNU)"] = 0.5
    
    # Handle fDOM column (may be missing in some datasets)
    if "fDOM (QSU)" not in df.columns:
        df["fDOM (QSU)"] = 3.0  # Default synthetic value
    
    if "Suhu Air (°C)" in feature_cols and "Suhu Air (°C)" not in df.columns:
        if "Temperature (°C)" in df.columns:
            df["Suhu Air (°C)"] = df["Temperature (°C)"]
            imputation_flags["imputed_Suhu_Air_from_Temperature"] = True
        else:
            raise ValueError(
                "Feature 'Suhu Air (°C)' is required by model but missing and cannot be derived "
                "because 'Temperature (°C)' is not present."
            )
    elif "Suhu Air (°C)" in feature_cols:
        imputation_flags["imputed_Suhu_Air_from_Temperature"] = False

    if "Suhu Lingkungan (°C)" in feature_cols and "Suhu Lingkungan (°C)" not in df.columns:
        if "Temperature (°C)" in df.columns:
            df["Suhu Lingkungan (°C)"] = df["Temperature (°C)"] + 1.5
            imputation_flags["imputed_Suhu_Lingkungan_default_offset"] = True
        else:
            raise ValueError(
                "Feature 'Suhu Lingkungan (°C)' is required by model but missing and cannot be derived "
                "because 'Temperature (°C)' is not present."
            )
    elif "Suhu Lingkungan (°C)" in feature_cols:
        imputation_flags["imputed_Suhu_Lingkungan_default_offset"] = False

    if "Kelembapan Lingkungan (%)" in feature_cols and "Kelembapan Lingkungan (%)" not in df.columns:
        df["Kelembapan Lingkungan (%)"] = 70.0
        imputation_flags["imputed_Kelembapan_default_70"] = True
    elif "Kelembapan Lingkungan (%)" in feature_cols:
        imputation_flags["imputed_Kelembapan_default_70"] = False

    if "Tegangan (V)" in feature_cols and "Tegangan (V)" not in df.columns:
        df["Tegangan (V)"] = 3.8
        imputation_flags["imputed_Tegangan_default_3_8"] = True
    elif "Tegangan (V)" in feature_cols:
        imputation_flags["imputed_Tegangan_default_3_8"] = False

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required model features: {missing_cols}. "
            f"Available columns after preprocessing: {list(df.columns)}"
        )

    # Drop missing values in required model columns
    initial_rows = len(df)
    df = df.dropna(subset=feature_cols)
    imputation_flags = imputation_flags.loc[df.index].copy()
    dropped_rows = initial_rows - len(df)
    
    print(f"  Rows after validation: {len(df)} (dropped {dropped_rows})")
    print(f"  EC range: {df['EC'].min():.2f} - {df['EC'].max():.2f} uS/cm")
    print(f"  pH range: {df['pH'].min():.2f} - {df['pH'].max():.2f}")
    print(f"  TDS range: {df['TDS'].min():.2f} - {df['TDS'].max():.2f} mg/L")
    
    # Track if any fallback path was used for each row
    imputation_flags["any_default_or_imputation"] = imputation_flags.any(axis=1)

    print(f"  Rows with default/imputed feature values: {int(imputation_flags['any_default_or_imputation'].sum())}")

    return df, imputation_flags


def run_inference(
    model,
    scaler,
    data: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """Run model inference on processed dataset."""
    print(f"  Running inference on {len(data)} samples...")
    
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns: {missing_cols}. "
            f"Required by model metadata: {feature_cols}"
        )

    x = data[feature_cols].copy()
    
    # Scale if scaler exists
    if scaler:
        x = scaler.transform(x)
    
    predictions = model.predict(x)
    data = data.copy()
    data["Cr_predicted"] = predictions
    
    return data


def analyze_results(df: pd.DataFrame) -> dict:
    """Compute basic statistics on predictions."""
    cr_pred = df["Cr_predicted"]
    
    stats = {
        "Count": len(df),
        "Mean": float(cr_pred.mean()),
        "Std": float(cr_pred.std()),
        "Min": float(cr_pred.min()),
        "25%": float(cr_pred.quantile(0.25)),
        "Median": float(cr_pred.median()),
        "75%": float(cr_pred.quantile(0.75)),
        "Max": float(cr_pred.max()),
    }
    
    print(f"\n  Cr Prediction Statistics:")
    for key, val in stats.items():
        if key != "Count":
            print(f"    {key:10s}: {val:10.2f} ug/L")
        else:
            print(f"    {key:10s}: {val:10d}")
    
    return stats


def plot_analysis(
    data_list: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Generate enhanced analysis plots for all datasets combined."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # Combine all data for overview plots
    all_data = pd.concat([df for _, df in data_list], ignore_index=True)
    
    # Plot 1: Histogram of Cr predictions (all datasets)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(all_data["Cr_predicted"], bins=50, color="steelblue", edgecolor="darkblue", alpha=0.7, density=False)
    ax.axvline(all_data["Cr_predicted"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {all_data['Cr_predicted'].mean():.2f} ug/L")
    ax.axvline(all_data["Cr_predicted"].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {all_data['Cr_predicted'].median():.2f} ug/L")
    ax.set_xlabel("Predicted Cr (ug/L)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Predicted Chromium Concentration (All Datasets - 163,959 samples)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "01_cr_distribution_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 2: EC vs Cr_predicted (scatter by dataset)
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (dataset_name, df) in enumerate(data_list):
        ax.scatter(df["EC"], df["Cr_predicted"], alpha=0.5, s=15, label=dataset_name, color=colors[idx])
    ax.set_xlabel("Electrical Conductivity (uS/cm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Cr (ug/L)", fontsize=12, fontweight="bold")
    ax.set_title("EC vs Predicted Chromium (by Dataset)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "02_ec_vs_cr_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 3: pH vs Cr_predicted (scatter by dataset)
    fig, ax = plt.subplots(figsize=(13, 7))
    for idx, (dataset_name, df) in enumerate(data_list):
        ax.scatter(df["pH"], df["Cr_predicted"], alpha=0.5, s=15, label=dataset_name, color=colors[idx])
    ax.set_xlabel("pH", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Cr (ug/L)", fontsize=12, fontweight="bold")
    ax.set_title("pH vs Predicted Chromium (by Dataset)", fontsize=13, fontweight="bold")
    ax.axvline(x=7.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(7.0, ax.get_ylim()[1]*0.95, "pH 7.0", fontsize=9, ha="center")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "03_ph_vs_cr_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 4: TDS vs Cr_predicted (scatter by dataset)
    fig, ax = plt.subplots(figsize=(13, 7))
    for idx, (dataset_name, df) in enumerate(data_list):
        ax.scatter(df["TDS"], df["Cr_predicted"], alpha=0.5, s=15, label=dataset_name, color=colors[idx])
    ax.set_xlabel("Total Dissolved Solids (mg/L)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Cr (ug/L)", fontsize=12, fontweight="bold")
    ax.set_title("TDS vs Predicted Chromium (by Dataset)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "04_tds_vs_cr_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 5: Temperature vs Cr_predicted (scatter by dataset)
    fig, ax = plt.subplots(figsize=(13, 7))
    for idx, (dataset_name, df) in enumerate(data_list):
        ax.scatter(df["Temperature (°C)"], df["Cr_predicted"], alpha=0.5, s=15, label=dataset_name, color=colors[idx])
    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Cr (ug/L)", fontsize=12, fontweight="bold")
    ax.set_title("Temperature vs Predicted Chromium (by Dataset)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "05_temperature_vs_cr_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 6: Box plots by dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    dataset_names = [name for name, _ in data_list]
    dataset_cr_values = [df["Cr_predicted"].values for _, df in data_list]
    bp = ax.boxplot(dataset_cr_values, labels=dataset_names, patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Predicted Cr (ug/L)", fontsize=12, fontweight="bold")
    ax.set_title("Predicted Cr Distribution by Dataset (Box Plot with Mean)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "06_dataset_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 7: ORP vs Cr_predicted
    fig, ax = plt.subplots(figsize=(13, 7))
    for idx, (dataset_name, df) in enumerate(data_list):
        ax.scatter(df["ORP (mV)"], df["Cr_predicted"], alpha=0.5, s=15, label=dataset_name, color=colors[idx])
    ax.set_xlabel("ORP (mV)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Cr (ug/L)", fontsize=12, fontweight="bold")
    ax.set_title("ORP vs Predicted Chromium (by Dataset)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "07_orp_vs_cr_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 8: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_cols = ["EC", "TDS", "pH", "Temperature (°C)", "ORP (mV)", "Cr_predicted"]
    corr_matrix = all_data[corr_cols].corr()
    
    im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_cols)))
    ax.set_yticks(np.arange(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    
    # Add correlation values to cells
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    
    ax.set_title("Correlation Matrix - All Parameters & Predicted Cr", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Correlation Coefficient")
    plt.tight_layout()
    plt.savefig(output_dir / "08_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 9: Multi-parameter density plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    params = ["EC", "TDS", "pH", "Temperature (°C)", "ORP (mV)", "Cr_predicted"]
    units = ["uS/cm", "mg/L", "", "°C", "mV", "ug/L"]
    
    for idx, (param, unit) in enumerate(zip(params, units)):
        ax = axes[idx]
        ax.hist(all_data[param], bins=40, color="steelblue", edgecolor="darkblue", alpha=0.7)
        ax.set_xlabel(f"{param} ({unit})" if unit else param, fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        ax.set_title(f"Distribution: {param}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Parameter Distributions (All Data)", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / "09_parameter_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Plot 10: Predicted Cr statistics by dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    dataset_names = [name for name, _ in data_list]
    dataset_means = [df["Cr_predicted"].mean() for _, df in data_list]
    dataset_stds = [df["Cr_predicted"].std() for _, df in data_list]
    
    x_pos = np.arange(len(dataset_names))
    bars = ax.bar(x_pos, dataset_means, yerr=dataset_stds, capsize=5, color=colors, alpha=0.7, edgecolor="black")
    
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Predicted Cr (ug/L)", fontsize=12, fontweight="bold")
    ax.set_title("Mean Predicted Chromium by Dataset (with Std Dev Error Bars)", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_names, rotation=10, ha="right")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, dataset_means)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + dataset_stds[i] + 0.1, 
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "10_dataset_means_barplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ {10} visualization files saved to {output_dir}/")


def write_outputs(
    output_dir: Path,
    data_list: list[tuple[str, pd.DataFrame]],
    summary_stats: dict,
    imputation_audits: list[tuple[str, pd.DataFrame]],
) -> None:
    """Save predictions and summary to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual dataset predictions
    for dataset_name, df in data_list:
        output_path = output_dir / f"predictions_{dataset_name}.csv"
        # Keep original columns + new predictions
        output_cols = list(df.columns)
        df.to_csv(output_path, index=False)
        print(f"  ✓ {output_path.name}")
    
    # Save combined predictions
    all_data = pd.concat([df for _, df in data_list], ignore_index=True)
    combined_path = output_dir / "predictions_combined.csv"
    all_data.to_csv(combined_path, index=False)
    print(f"  ✓ {combined_path.name}")
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ {summary_path.name}")

    # Save imputation/default audit
    all_audits = []
    for dataset_name, audit_df in imputation_audits:
        temp = audit_df.copy()
        temp.insert(0, "Dataset", dataset_name)
        all_audits.append(temp)
    if all_audits:
        audit_path = output_dir / "imputation_audit.csv"
        pd.concat(all_audits, ignore_index=True).to_csv(audit_path, index=False)
        print(f"  ✓ {audit_path.name}")


def print_summary(summary_stats: dict) -> None:
    """Print final summary."""
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY - REAL MWQ DATASETS")
    print("=" * 70)
    
    print("\nDataset Summary:")
    df_summary = pd.DataFrame(summary_stats)
    print(df_summary.to_string(index=False))
    
    overall_mean = np.mean([s["Mean"] for s in summary_stats])
    overall_std = np.mean([s["Std"] for s in summary_stats])
    overall_min = min([s["Min"] for s in summary_stats])
    overall_max = max([s["Max"] for s in summary_stats])
    
    print(f"\nOverall Statistics (All Datasets):")
    print(f"  Mean Cr: {overall_mean:.2f} ug/L")
    print(f"  Std:     {overall_std:.2f} ug/L")
    print(f"  Range:   {overall_min:.2f} - {overall_max:.2f} ug/L")
    
    print("=" * 70 + "\n")


def main() -> None:
    """Main orchestration."""
    parser = argparse.ArgumentParser(
        description="Apply Cr soft sensor model to real MWQ datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing MWQ CSV datasets",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (relative to Dataset-Cr-HERA2.0)",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=None,
        help="Optional path to scaler (if omitted, inferred from metadata)",
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    
    # Resolve model path from Testing-MWQ's parent (Dataset-Cr-HERA2.0)
    base_dir = data_dir.parent.parent
    if args.model_path:
        model_path = base_dir / args.model_path
    else:
        best_meta_path = base_dir / "models" / "best_model_metadata.json"
        if best_meta_path.exists():
            with open(best_meta_path, "r", encoding="utf-8") as f:
                best_meta = json.load(f)
            model_name = best_meta.get("model_name", best_meta.get("model_type", "rf"))
            scenario = best_meta.get("scenario", "full")
            model_path = base_dir / "models" / f"best_model_{model_name}_{scenario}.pkl"
            if not model_path.exists():
                model_path = base_dir / "models" / f"model_{model_name}_{scenario}.pkl"
        else:
            model_path = base_dir / "models" / "best_model_rf_full.pkl"
    scaler_path = base_dir / args.scaler_path if args.scaler_path else None
    
    output_dir = data_dir / "result"
    images_dir = data_dir / "images"
    
    # Load model
    model, scaler, metadata = load_model(model_path, scaler_path)

    feature_cols = metadata.get("features")
    if not feature_cols:
        raise ValueError(
            "Model metadata does not define 'features'. "
            "Please use model artifacts produced by the updated training pipeline."
        )
    
    # Process all datasets
    print("=" * 70)
    print("PROCESSING MWQ REAL DATASETS")
    print("=" * 70)
    
    data_list = []
    summary_stats = []
    imputation_audits = []
    
    csv_files = sorted(data_dir.glob("dataset_*.csv"))
    
    for csv_path in csv_files:
        dataset_name = csv_path.stem
        
        # Process
        df, audit_df = process_mwq_dataset(csv_path, dataset_name, feature_cols)
        
        # Inference
        df = run_inference(model, scaler, df, feature_cols)
        
        # Analyze
        stats = analyze_results(df)
        stats["Dataset"] = dataset_name
        summary_stats.append(stats)
        
        data_list.append((dataset_name, df))
        imputation_audits.append((dataset_name, audit_df))
    
    # Generate plots
    plot_analysis(data_list, images_dir)
    
    # Save outputs
    print(f"\nSaving outputs to {output_dir}/...")
    write_outputs(output_dir, data_list, summary_stats, imputation_audits)
    
    # Print summary
    print_summary(summary_stats)


if __name__ == "__main__":
    main()
