import pandas as pd
import numpy as np
import os

# =============================================================================
# HERA 2.0 — Physics-Informed Synthetic Dataset Generator  v1  (ORIGINAL)
# =============================================================================
# Menghasilkan 5.000 sampel dengan distribusi normal sekitar pH 6.5.
# File ini adalah versi ASLI yang digunakan pada Phase 2 training.
# Untuk dataset v2 (Phase 2.5 Fine-Tuning), lihat:
#   phase2.5_finetuning/src/generate_dataset.py
# =============================================================================

def main():
    print("[INFO] Starting HERA 2.0 Forward Geochemical Dataset Generation (v1)...")

    # Resolve relative paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset", "dataset_heavy_metal_grounded.csv"))

    num_samples = 5000

    # Set random seed for complete reproducibility
    np.random.seed(42)

    print(f"[INFO] Generating {num_samples} physics-informed heavy metal water samples...")

    # 1. Generate physical environmental predictors first (Independent features)
    # pH: distributed normally around 6.5, representing typical slightly acidic to neutral streams
    ph = np.random.normal(6.5, 0.7, size=num_samples)
    ph = np.clip(ph, 4.8, 8.5)

    # EC: representing fresh to industrial water
    ec_uScm = np.random.normal(800.0, 350.0, size=num_samples)
    ec_uScm = np.clip(ec_uScm, 100.0, 2500.0)

    # TDS: Empirically derived (TDS = 0.64 * EC)
    tds_mgL = ec_uScm * 0.64

    # Suhu Air (Water Temperature)
    suhu_air = np.random.normal(24.5, 3.5, size=num_samples)
    suhu_air = np.clip(suhu_air, 15.0, 35.0)

    # 2. Compute dependent heavy metal concentrations
    # Nickel target range: 6.7 - 706 ug/L
    log_ni = 1.25 - 0.35 * (ph - 7.0) + 0.0005 * (ec_uScm - 600.0) + np.random.normal(0, 0.05, size=num_samples)
    ni_ugL = 10.0 ** log_ni
    ni_ugL = np.clip(ni_ugL, 6.7, 706.0)

    # Chromium target range: 5.0 - 300 ug/L
    log_cr = 1.45 - 0.32 * (ph - 7.0) + 0.0004 * (ec_uScm - 600.0) + np.random.normal(0, 0.04, size=num_samples)
    cr_ugL = 10.0 ** log_cr
    cr_ugL = np.clip(cr_ugL, 5.0, 300.0)

    # 3. Construct and Export DataFrame
    df = pd.DataFrame({
        "pH":           ph,
        "EC_uScm":      ec_uScm,
        "TDS_mgL":      tds_mgL,
        "Suhu_Air":     suhu_air,
        "Chromium_ugL": cr_ugL,
        "Nickel_ugL":   ni_ugL,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset v1 saved to: {output_path}")
    print(f"[INFO] Shape: {df.shape}")
    print("\n--- Feature Summary Stats ---")
    print(df.describe())
    print("\n--- Spearman Rank Correlations ---")
    print(df.corr(method="spearman"))


if __name__ == "__main__":
    main()
