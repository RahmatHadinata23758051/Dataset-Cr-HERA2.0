import pandas as pd
import numpy as np
import os

# =============================================================================
# HERA 2.0 — Physics-Informed Synthetic Dataset Generator  v2
# =============================================================================
# Changes vs v1:
#   - Samples: 5,000 -> 15,000
#   - Sampling: stratified across 4 pH bands (avoids over-concentration at pH 6.5)
#   - 5 derived features added (physics-informed, computed from raw inputs):
#       pH_squared, pH_EC_interact, log_EC, pOH_proxy, pH_temp_interact
#   - Output: dataset_heavy_metal_grounded_v2.csv  (raw + derived, 9+2=11 cols)
#   - Original v1 file is KEPT untouched for backward-compatibility
# =============================================================================

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5 physics-informed derived features from raw sensor columns.

    Justification per feature:
    - pH_squared       : Ksp(Ni(OH)2) is proportional to [OH-]^2, i.e. 10^(2*(pH-14))
                         => non-linear (quadratic) pH dependence must be captured
    - pH_EC_interact   : At low pH + high EC, ionic activity of free metal cations
                         is amplified — a multiplicative interaction term captures this
    - log_EC           : EC correlates log-linearly with dissolved ionic strength;
                         log transform reduces right-skew and linearises the relationship
    - pOH_proxy        : pOH = 14 - pH is the direct driver of OH- activity controlling
                         precipitation equilibrium (Ksp), more physically meaningful than pH
    - pH_temp_interact : Van't Hoff equation: Ksp shifts with temperature; the product
                         pH x T encodes both the OH- activity and the thermal effect on Ksp
    """
    df = df.copy()
    df["pH_squared"]       = df["pH"] ** 2
    df["pH_EC_interact"]   = df["pH"] * df["EC_uScm"]
    df["log_EC"]           = np.log10(df["EC_uScm"])
    df["pOH_proxy"]        = 14.0 - df["pH"]
    df["pH_temp_interact"] = df["pH"] * df["Suhu_Air"]
    return df


def generate_stratum(n: int, ph_lo: float, ph_hi: float,
                     ph_center: float, rng: np.random.Generator) -> dict:
    """
    Generate `n` samples for a single pH stratum.

    pH is sampled uniformly within [ph_lo, ph_hi] so each stratum is equally
    represented, preventing the normal-distribution clustering around pH 6.5
    that underrepresented extreme (acidic/basic) conditions in v1.
    """
    # pH: uniform within stratum bounds
    ph = rng.uniform(ph_lo, ph_hi, size=n)

    # EC: log-normal to capture the heavy-tailed distribution of industrial waters
    # Mean and std vary by stratum: more acidic = typically higher EC (runoff)
    ec_mean = 300.0 + (7.0 - ph_center) * 350.0   # higher EC in acidic strata
    ec_mean = np.clip(ec_mean, 150.0, 2000.0)
    ec_uScm = rng.lognormal(mean=np.log(ec_mean), sigma=0.50, size=n)
    ec_uScm = np.clip(ec_uScm, 100.0, 2500.0)

    # TDS: empirical linear relationship (TDS ≈ 0.64 * EC)
    tds_mgL = ec_uScm * 0.64

    # Temperature: independent of pH stratum, representing tropical river conditions
    suhu_air = rng.normal(24.5, 3.5, size=n)
    suhu_air = np.clip(suhu_air, 15.0, 35.0)

    # ── Nickel concentration (forward geochemical model) ──────────────────────
    # log[Ni] = f(pH, EC) + noise
    # Coefficient -0.35 on (pH - 7.0) gives ~ 1 log-unit change per 2.86 pH units
    # consistent with Ksp(Ni(OH)2) = 5.48e-16 ≈ 10^(2*(pH_limit - pH))
    log_ni = (1.25
              - 0.35 * (ph - 7.0)
              + 0.0005 * (ec_uScm - 600.0)
              + rng.normal(0, 0.04, size=n))           # reduced noise vs v1 (0.05->0.04)
    ni_ugL = 10.0 ** log_ni
    ni_ugL = np.clip(ni_ugL, 6.7, 706.0)

    # ── Chromium concentration (forward geochemical model) ────────────────────
    # Cr(III): log[Cr] depends on 3*(pKsp/3 - pH) -> slope ~0.32 per pH unit
    log_cr = (1.45
              - 0.32 * (ph - 7.0)
              + 0.0004 * (ec_uScm - 600.0)
              + rng.normal(0, 0.035, size=n))
    cr_ugL = 10.0 ** log_cr
    cr_ugL = np.clip(cr_ugL, 5.0, 300.0)

    return {
        "pH":           ph,
        "EC_uScm":      ec_uScm,
        "TDS_mgL":      tds_mgL,
        "Suhu_Air":     suhu_air,
        "Chromium_ugL": cr_ugL,
        "Nickel_ugL":   ni_ugL,
    }


def main():
    print("=" * 70)
    print("  HERA 2.0 Dataset Generator v2")
    print("  Physics-Informed Stratified Synthetic Dataset (15,000 samples)")
    print("=" * 70)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset"))

    # v1 path — NOT overwritten
    output_v1 = os.path.join(dataset_dir, "dataset_heavy_metal_grounded.csv")
    # v2 path — new file
    output_v2 = os.path.join(dataset_dir, "dataset_heavy_metal_grounded_v2.csv")

    os.makedirs(dataset_dir, exist_ok=True)

    NUM_SAMPLES = 15_000
    rng = np.random.default_rng(seed=42)       # use Generator API for reproducibility

    # ── Stratified pH sampling ────────────────────────────────────────────────
    # 4 pH bands — each gets 25% of total samples (3,750 each)
    # Band centres chosen to reflect acid mine drainage, industrial runoff,
    # moderately impacted, and near-neutral river conditions.
    strata = [
        {"ph_lo": 4.8,  "ph_hi": 5.5,  "ph_center": 5.15, "label": "Acidic (industrial runoff)"},
        {"ph_lo": 5.5,  "ph_hi": 6.5,  "ph_center": 6.0,  "label": "Moderately acidic"},
        {"ph_lo": 6.5,  "ph_hi": 7.5,  "ph_center": 7.0,  "label": "Near-neutral"},
        {"ph_lo": 7.5,  "ph_hi": 8.5,  "ph_center": 8.0,  "label": "Alkaline (downstream)"},
    ]

    n_per_stratum = NUM_SAMPLES // len(strata)   # 3,750 each
    remainder     = NUM_SAMPLES - n_per_stratum * len(strata)

    all_chunks = []
    print(f"\n[INFO] Generating {NUM_SAMPLES:,} samples across {len(strata)} pH strata...\n")

    for i, stratum in enumerate(strata):
        n = n_per_stratum + (1 if i < remainder else 0)
        chunk = generate_stratum(n, stratum["ph_lo"], stratum["ph_hi"],
                                 stratum["ph_center"], rng)
        all_chunks.append(chunk)
        print(f"  Stratum {i+1}: pH [{stratum['ph_lo']:.1f} - {stratum['ph_hi']:.1f}]"
              f"  ({stratum['label']})  ->  {n:,} samples")

    # ── Assemble raw DataFrame ────────────────────────────────────────────────
    raw_df = pd.DataFrame({
        key: np.concatenate([c[key] for c in all_chunks])
        for key in all_chunks[0].keys()
    })

    # Shuffle rows so strata are intermixed (not block-ordered)
    raw_df = raw_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Add derived features ──────────────────────────────────────────────────
    df_v2 = add_derived_features(raw_df)

    # ── Export ────────────────────────────────────────────────────────────────
    df_v2.to_csv(output_v2, index=False)

    print(f"\n[OK] v2 dataset saved -> {output_v2}")
    print(f"[INFO] Shape: {df_v2.shape}  ({df_v2.shape[0]:,} rows x {df_v2.shape[1]} columns)\n")

    # ── Column summary ────────────────────────────────────────────────────────
    raw_cols     = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    derived_cols = ["pH_squared", "pH_EC_interact", "log_EC", "pOH_proxy", "pH_temp_interact"]
    target_cols  = ["Chromium_ugL", "Nickel_ugL"]

    print("Columns in v2 dataset:")
    print(f"  Raw features    : {raw_cols}")
    print(f"  Derived features: {derived_cols}")
    print(f"  Targets         : {target_cols}")

    print("\n--- Feature Summary Statistics ---")
    print(df_v2.describe().to_string())

    print("\n--- pH Distribution by Stratum (verification) ---")
    bins = [4.8, 5.5, 6.5, 7.5, 8.5]
    labels = ["[4.8-5.5)", "[5.5-6.5)", "[6.5-7.5)", "[7.5-8.5]"]
    ph_band = pd.cut(df_v2["pH"], bins=bins, labels=labels, include_lowest=True)
    print(ph_band.value_counts().sort_index().to_string())

    print("\n--- Spearman Correlations: Raw Features -> Targets ---")
    spear_cols = raw_cols + target_cols
    print(df_v2[spear_cols].corr(method="spearman").to_string())

    print("\n--- Spearman Correlations: Derived Features -> Targets ---")
    spear_cols2 = derived_cols + target_cols
    print(df_v2[spear_cols2].corr(method="spearman").to_string())

    # Sanity check: verify that v1 file is still intact
    if os.path.exists(output_v1):
        v1_rows = pd.read_csv(output_v1).shape[0]
        print(f"\n[OK] v1 dataset intact -> {output_v1}  ({v1_rows:,} rows, unchanged)")
    else:
        print(f"\n[WARN] v1 dataset not found at {output_v1}")

    print("\n" + "=" * 70)
    print("  [SUCCESS] Dataset v2 generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
