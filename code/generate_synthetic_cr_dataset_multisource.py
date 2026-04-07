#!/usr/bin/env python3
"""
Generate NEW multisource synthetic dataset for Cr estimation.

This script keeps old outputs untouched by writing to:
  Dataset/Synthetic_Multisource/

Sources used:
1) UNEP GEMSWater (GFQA_v3)
2) A Comprehensive Surface Water Quality Monitoring Dataset (1940-2023)
3) Tabel 1 dalam dokumen Zenodo Nigeria
4) Water Quality Monitoring Dataset for Tilapia (Monteria, 2024)
5) Water Quality Pollution Indices for Heavy Metal Contamination Monitoring
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ensure(dic: Dict, key: str) -> Dict:
    if key not in dic:
        dic[key] = {}
    return dic[key]


def collect_multisource_values(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """
    Collect raw values by category and variable from all real datasets.
    Returns:
      values[category][variable] = list of numeric values
      contribution table
    """
    values: Dict[str, Dict[str, List[float]]] = {}
    contrib_rows: List[dict] = []

    def push(source: str, category: str, variable: str, arr: List[float]) -> None:
        if len(arr) == 0:
            return
        cat_obj = ensure(values, category)
        if variable not in cat_obj:
            cat_obj[variable] = []
        cat_obj[variable].extend(arr)
        contrib_rows.append(
            {
                "Source": source,
                "Kategori": category,
                "Variabel": variable,
                "N": int(len(arr)),
            }
        )

    # ---------------------------------------------------------------------
    # 1) UNEP GEMSWater (core source for EC/TDS/pH/TEMP/TEMP-Air/Cr)
    # ---------------------------------------------------------------------
    gfqa = dataset_dir / "UNEP GEMSWater Global Freshwater Quality Archive" / "GFQA_v3"
    station = pd.read_csv(
        gfqa / "GEMStat_station_metadata.csv",
        usecols=["GEMS Station Number", "Water Type"],
        dtype=str,
        encoding="latin1",
    )
    water_map = {
        "River station": "Sungai",
        "Lake station": "Danau",
        "Reservoir station": "Waduk",
        "Groundwater station": "Air tanah",
    }
    station["Kategori"] = station["Water Type"].map(water_map)
    station = station.dropna(subset=["Kategori"])
    station_map = dict(zip(station["GEMS Station Number"], station["Kategori"]))

    def read_unep(file_name: str, variable: str, code: str | None = None) -> None:
        src = gfqa / file_name
        usecols = ["GEMS Station Number", "Value"] if code is None else ["GEMS Station Number", "Parameter Code", "Value"]
        for ch in pd.read_csv(src, usecols=usecols, chunksize=300000, encoding="latin1"):
            if code is not None:
                ch = ch[ch["Parameter Code"] == code]
            if ch.empty:
                continue
            ch["Kategori"] = ch["GEMS Station Number"].map(station_map)
            ch = ch.dropna(subset=["Kategori"])
            ch["Value"] = pd.to_numeric(ch["Value"], errors="coerce")
            ch = ch.dropna(subset=["Value"])
            for c, g in ch.groupby("Kategori"):
                push("UNEP", c, variable, g["Value"].tolist())

    read_unep("Electrical_Conductance.csv", "EC", "EC")
    read_unep("Water.csv", "TDS", "TDS")
    read_unep("pH.csv", "pH", "pH")
    read_unep("Temperature.csv", "Suhu Air (°C)", "TEMP")
    read_unep("Temperature.csv", "Suhu Lingkungan (°C)", "TEMP-Air")
    # Chromium.csv already represents chromium measurements, so do not filter by Parameter Code.
    read_unep("Chromium.csv", "Cr", None)

    # ---------------------------------------------------------------------
    # 2) A Comprehensive Surface Water Quality Monitoring Dataset
    # ---------------------------------------------------------------------
    ac_file = (
        dataset_dir
        / "A Comprehensive Surface Water Quality Monitoring Dataset (1940-2023)"
        / "Dataset"
        / "Combined Data"
        / "Combined_dataset.csv"
    )
    usecols = ["Waterbody Type", "pH (ph units)", "Temperature (cel)"]
    for ch in pd.read_csv(ac_file, usecols=usecols, chunksize=300000):
        ch["Kategori"] = ch["Waterbody Type"].map({"River": "Sungai", "Lake": "Danau", "Reservoir": "Waduk"})
        ch = ch.dropna(subset=["Kategori"])
        ch["pH (ph units)"] = pd.to_numeric(ch["pH (ph units)"], errors="coerce")
        ch["Temperature (cel)"] = pd.to_numeric(ch["Temperature (cel)"], errors="coerce")
        for c, g in ch.groupby("Kategori"):
            push("A Comprehensive", c, "pH", g["pH (ph units)"].dropna().tolist())
            push("A Comprehensive", c, "Suhu Air (°C)", g["Temperature (cel)"].dropna().tolist())

    # ---------------------------------------------------------------------
    # 3) Nigeria dataset (pH/TDS + matrix category)
    # ---------------------------------------------------------------------
    ng_file = dataset_dir / "Tabel 1 dalam dokumen Zenodo Nigeria" / "water_data.csv"
    ng = pd.read_csv(ng_file)
    ng["Kategori"] = ng["Matriks Air"].map({"Air Permukaan": "Sungai", "Air Tanah": "Air tanah"})
    ng = ng.dropna(subset=["Kategori"])
    ng["pH"] = pd.to_numeric(ng["pH"], errors="coerce")
    ng["TDS"] = pd.to_numeric(ng["TDS (mg/L)"], errors="coerce")
    for c, g in ng.groupby("Kategori"):
        push("Nigeria", c, "pH", g["pH"].dropna().tolist())
        push("Nigeria", c, "TDS", g["TDS"].dropna().tolist())

    # ---------------------------------------------------------------------
    # 4) Monteria aquaculture dataset (new category: Akuakultur)
    # ---------------------------------------------------------------------
    mt_file = (
        dataset_dir
        / "Water Quality Monitoring Dataset for Tilapia (Oreochromis niloticus) Aquaculture in Montería, Colombia (2024)"
        / "Monteria_Aquaculture_Data.csv"
    )
    mt = pd.read_csv(mt_file, sep=";", decimal=",")
    mt["Temperature"] = pd.to_numeric(mt["Temperature"], errors="coerce")
    mt["pH"] = pd.to_numeric(mt["pH"], errors="coerce")
    push("Monteria", "Akuakultur", "Suhu Air (°C)", mt["Temperature"].dropna().tolist())
    push("Monteria", "Akuakultur", "pH", mt["pH"].dropna().tolist())

    # ---------------------------------------------------------------------
    # 5) Heavy metal pollution indices (Cr, mapped as Sungai context)
    # ---------------------------------------------------------------------
    hm_file = dataset_dir / "Water Quality Pollution Indices for Heavy Metal Contamination Monitoring" / "Data.csv"
    hm = pd.read_csv(hm_file, sep=";", header=1, decimal=",")
    if "Cr" in hm.columns:
        cr_vals = pd.to_numeric(hm["Cr"], errors="coerce").dropna().tolist()
        push("Heavy Metal Indices", "Sungai", "Cr", cr_vals)

    contrib = pd.DataFrame(contrib_rows)
    if not contrib.empty:
        contrib = contrib.groupby(["Source", "Kategori", "Variabel"], as_index=False)["N"].sum()
    return values, contrib


def robust_ranges(values: Dict[str, Dict[str, List[float]]], p_low: float, p_high: float) -> Dict[str, Dict[str, Tuple[float, float]]]:
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for cat, vmap in values.items():
        out[cat] = {}
        for var, arr_list in vmap.items():
            arr = np.array(arr_list, dtype=float)
            if arr.size == 0:
                continue
            out[cat][var] = (float(np.percentile(arr, p_low)), float(np.percentile(arr, p_high)))
    return out


def build_effective_ranges(
    p10p90: Dict[str, Dict[str, Tuple[float, float]]],
    p25p75: Dict[str, Dict[str, Tuple[float, float]]],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Build effective ranges for generation, including fallback for missing vars.
    """
    cats = ["Sungai", "Danau", "Waduk", "Air tanah", "Akuakultur"]
    vars_need = ["Suhu Air (°C)", "Suhu Lingkungan (°C)", "TDS", "EC", "pH", "Cr"]

    eff: Dict[str, Dict[str, Tuple[float, float]]] = {c: {} for c in cats}

    # Direct values
    for c in cats:
        for v in vars_need:
            if c in p10p90 and v in p10p90[c]:
                eff[c][v] = p10p90[c][v]

    # Fallback helpers from known categories
    sungai_ec = eff.get("Sungai", {}).get("EC", (80.0, 900.0))
    sungai_tds = eff.get("Sungai", {}).get("TDS", (30.0, 800.0))
    sungai_air = eff.get("Sungai", {}).get("Suhu Lingkungan (°C)", (12.0, 34.0))
    danau_cr = eff.get("Danau", {}).get("Cr", (0.0003, 0.010))
    waduk_cr = eff.get("Waduk", {}).get("Cr", (0.0005, 0.009))

    # Akuakultur fallback for missing EC/TDS/Suhu Lingkungan/Cr
    if "Akuakultur" in eff:
        wt = eff["Akuakultur"].get("Suhu Air (°C)", (26.5, 30.0))
        if "EC" not in eff["Akuakultur"]:
            eff["Akuakultur"]["EC"] = (sungai_ec[0] * 0.8, sungai_ec[1] * 0.9)
        if "TDS" not in eff["Akuakultur"]:
            eff["Akuakultur"]["TDS"] = (sungai_tds[0] * 0.9, sungai_tds[1] * 0.85)
        if "Suhu Lingkungan (°C)" not in eff["Akuakultur"]:
            lo = max(18.0, wt[0] + 1.0)
            hi = min(38.0, wt[1] + 6.0)
            eff["Akuakultur"]["Suhu Lingkungan (°C)"] = (lo, hi)
        if "Cr" not in eff["Akuakultur"]:
            eff["Akuakultur"]["Cr"] = ((danau_cr[0] + waduk_cr[0]) / 2.0, (danau_cr[1] + waduk_cr[1]) / 2.0)

    # Fill any remaining missing with conservative global defaults
    global_defaults = {
        "Suhu Air (°C)": (8.0, 30.0),
        "Suhu Lingkungan (°C)": (18.0, 35.0),
        "TDS": (20.0, 1500.0),
        "EC": (60.0, 1500.0),
        "pH": (6.7, 8.5),
        "Cr": (0.0003, 0.01),
    }
    for c in cats:
        for v in vars_need:
            if v not in eff[c]:
                eff[c][v] = global_defaults[v]

    # Synthetic-only ranges (not from real data)
    for c in cats:
        # humidity range by category profile
        if c == "Sungai":
            eff[c]["Kelembapan Lingkungan (%)"] = (55.0, 95.0)
        elif c == "Danau":
            eff[c]["Kelembapan Lingkungan (%)"] = (50.0, 92.0)
        elif c == "Waduk":
            eff[c]["Kelembapan Lingkungan (%)"] = (48.0, 90.0)
        elif c == "Air tanah":
            eff[c]["Kelembapan Lingkungan (%)"] = (45.0, 88.0)
        else:
            eff[c]["Kelembapan Lingkungan (%)"] = (52.0, 90.0)
        eff[c]["Tegangan (V)"] = (3.45, 4.25)

    # Ensure monotonic low<high
    for c in cats:
        for v, (lo, hi) in list(eff[c].items()):
            if lo >= hi:
                lo = lo * 0.9
                hi = hi * 1.1 if hi != 0 else 1.0
                eff[c][v] = (float(lo), float(hi))

    return eff


def generate_synthetic(eff: Dict[str, Dict[str, Tuple[float, float]]], n_total: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    categories = ["Sungai", "Danau", "Waduk", "Air tanah", "Akuakultur"]

    # Balanced allocation for fair representation across categories
    base = n_total // len(categories)
    rem = n_total % len(categories)
    n_map = {c: base for c in categories}
    for i in range(rem):
        n_map[categories[i]] += 1

    rows = []
    start_date = datetime(2025, 1, 1)
    time_slots = ["06:00:00", "07:00:00", "08:00:00", "09:00:00", "12:00:00", "13:00:00", "14:00:00", "17:00:00", "18:00:00"]

    # Global ranges for cross-category Cr construction (to avoid category confounding)
    gl_ec_lo = min(eff[c]["EC"][0] for c in categories)
    gl_ec_hi = max(eff[c]["EC"][1] for c in categories)
    gl_tds_lo = min(eff[c]["TDS"][0] for c in categories)
    gl_tds_hi = max(eff[c]["TDS"][1] for c in categories)
    gl_cr_lo = min(eff[c]["Cr"][0] for c in categories)
    gl_cr_hi = max(eff[c]["Cr"][1] for c in categories)

    for cat in categories:
        r = eff[cat]
        ec_lo, ec_hi = r["EC"]
        tds_lo, tds_hi = r["TDS"]
        ph_lo, ph_hi = r["pH"]
        wt_lo, wt_hi = r["Suhu Air (°C)"]
        at_lo, at_hi = r["Suhu Lingkungan (°C)"]
        cr_lo, cr_hi = r["Cr"]
        h_lo, h_hi = r["Kelembapan Lingkungan (%)"]
        v_lo, v_hi = r["Tegangan (V)"]

        mid_wt = (wt_lo + wt_hi) / 2.0
        amp_wt = (wt_hi - wt_lo) * 0.24
        mid_at = (at_lo + at_hi) / 2.0

        for _ in range(n_map[cat]):
            day = int(np.random.randint(0, 365))
            d = start_date + timedelta(days=day)
            t = np.random.choice(time_slots)
            ts = datetime.strptime(d.strftime("%Y-%m-%d") + " " + t, "%Y-%m-%d %H:%M:%S")
            season = np.sin(2 * np.pi * (ts.timetuple().tm_yday / 365.0))

            # EC skew/log-uniform
            ec = float(np.exp(np.random.uniform(np.log(ec_lo), np.log(ec_hi))))

            # TDS from EC
            if cat == "Danau":
                k = np.random.uniform(0.55, 0.82)
            elif cat == "Akuakultur":
                k = np.random.uniform(0.58, 0.78)
            else:
                k = np.random.uniform(0.50, 0.78)
            tds = ec * k * np.random.normal(1.0, 0.06)
            tds = clip(tds, tds_lo, tds_hi)

            # pH in range
            ph = np.random.normal((ph_lo + ph_hi) / 2.0, (ph_hi - ph_lo) / 5.0)
            ph = clip(ph, ph_lo, ph_hi)

            # temperatures
            wt = clip(mid_wt + amp_wt * season + np.random.normal(0, 1.1), wt_lo, wt_hi)
            at = clip(wt + np.random.uniform(1.2, 6.2) + np.random.normal(0, 0.8), at_lo, at_hi)

            hum = clip(np.random.normal((h_lo + h_hi) / 2.0, 8.0) - 0.7 * (at - mid_at), h_lo, h_hi)
            volt = clip(3.55 + 0.65 * np.random.rand() - 0.007 * (at - mid_at) + np.random.normal(0, 0.02), v_lo, v_hi)

            # Cr equation in mg/L-scale, linked to EC/TDS/pH + small noise
            # Use GLOBAL normalization so high EC/TDS categories remain positively aligned overall.
            ec_n = (np.log1p(ec) - np.log1p(gl_ec_lo)) / (np.log1p(gl_ec_hi) - np.log1p(gl_ec_lo))
            tds_n = (np.log1p(tds) - np.log1p(gl_tds_lo)) / (np.log1p(gl_tds_hi) - np.log1p(gl_tds_lo))
            ec_n = clip(ec_n, 0.0, 1.0)
            tds_n = clip(tds_n, 0.0, 1.0)
            acid = max(0.0, 7.2 - ph)
            acid_n = clip(acid / 1.2, 0.0, 1.0)
            # keep non-perfect relationship (stronger EC/TDS contribution)
            cr = gl_cr_lo + (gl_cr_hi - gl_cr_lo) * (0.54 * ec_n + 0.34 * tds_n + 0.12 * acid_n)
            cr = cr + np.random.normal(0.0, (gl_cr_hi - gl_cr_lo) * 0.02)
            cr = clip(cr, gl_cr_lo, gl_cr_hi)

            rows.append(
                {
                    "Kategori": cat,
                    "Tanggal": ts.strftime("%Y-%m-%d"),
                    "Waktu": ts.strftime("%H:%M:%S"),
                    "Tegangan (V)": round(volt, 3),
                    "Suhu Air (°C)": round(wt, 2),
                    "Suhu Lingkungan (°C)": round(at, 2),
                    "Kelembapan Lingkungan (%)": round(hum, 1),
                    "TDS": round(tds, 3),
                    "EC": round(ec, 3),
                    "pH": round(ph, 3),
                    "Cr": round(cr, 6),
                }
            )

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def fmt_range(r: Tuple[float, float], d: int = 2) -> str:
    return f"{r[0]:.{d}f} - {r[1]:.{d}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NEW multisource synthetic Cr dataset.")
    parser.add_argument("--base-dir", default=".", help="Project root path")
    parser.add_argument("--rows", type=int, default=120, help="Total rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    dataset_dir = base_dir / "Dataset"
    out_dir = dataset_dir / "Synthetic_Multisource"
    out_dir.mkdir(parents=True, exist_ok=True)

    values, contrib = collect_multisource_values(dataset_dir)
    p10p90 = robust_ranges(values, 10, 90)
    p25p75 = robust_ranges(values, 25, 75)
    eff = build_effective_ranges(p10p90, p25p75)
    syn = generate_synthetic(eff, n_total=args.rows, seed=args.seed)

    # Output files (NEW names, old files untouched)
    main_file = out_dir / "synthetic_cr_dataset_multisource.csv"
    audit_file = out_dir / "synthetic_cr_dataset_multisource_with_category.csv"
    contrib_file = out_dir / "ringkasan_kontribusi_sumber_multisource.csv"
    range_file = out_dir / "ringkasan_rentang_multisource_p10_p90.csv"
    normal_file = out_dir / "ringkasan_nilai_normal_multisource_p25_p75.csv"
    qa_file = out_dir / "qa_korelasi_multisource.csv"
    doc_file = base_dir / "dokumentasi_pembuatan_dataset_synthetic_cr_multisource.md"

    cols_main = [
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

    syn[cols_main].to_csv(main_file, index=False)
    syn.to_csv(audit_file, index=False)
    if not contrib.empty:
        contrib.to_csv(contrib_file, index=False)

    # Range tables from effective/final range model
    cats = ["Sungai", "Danau", "Waduk", "Air tanah", "Akuakultur"]
    range_rows = []
    normal_rows = []
    for c in cats:
        rr = eff[c]
        range_rows.append(
            {
                "Kategori": c,
                "Suhu Air (°C)": fmt_range(rr["Suhu Air (°C)"], 2),
                "Suhu Lingkungan (°C)": fmt_range(rr["Suhu Lingkungan (°C)"], 2),
                "TDS": fmt_range(rr["TDS"], 2),
                "EC": fmt_range(rr["EC"], 2),
                "pH": fmt_range(rr["pH"], 2),
                "Cr": fmt_range(rr["Cr"], 6),
                "Kelembapan (%)*": fmt_range(rr["Kelembapan Lingkungan (%)"], 1),
                "Tegangan (V)**": fmt_range(rr["Tegangan (V)"], 2),
            }
        )

        # "normal" from generated synthetic output itself (p25-p75) per category
        g = syn[syn["Kategori"] == c]
        normal_rows.append(
            {
                "Kategori": c,
                "Suhu Air (°C)": fmt_range((g["Suhu Air (°C)"].quantile(0.25), g["Suhu Air (°C)"].quantile(0.75)), 2),
                "Suhu Lingkungan (°C)": fmt_range((g["Suhu Lingkungan (°C)"].quantile(0.25), g["Suhu Lingkungan (°C)"].quantile(0.75)), 2),
                "TDS": fmt_range((g["TDS"].quantile(0.25), g["TDS"].quantile(0.75)), 3),
                "EC": fmt_range((g["EC"].quantile(0.25), g["EC"].quantile(0.75)), 3),
                "pH": fmt_range((g["pH"].quantile(0.25), g["pH"].quantile(0.75)), 3),
                "Cr": fmt_range((g["Cr"].quantile(0.25), g["Cr"].quantile(0.75)), 6),
                "Kelembapan (%)*": fmt_range((g["Kelembapan Lingkungan (%)"].quantile(0.25), g["Kelembapan Lingkungan (%)"].quantile(0.75)), 1),
                "Tegangan (V)**": fmt_range((g["Tegangan (V)"].quantile(0.25), g["Tegangan (V)"].quantile(0.75)), 3),
            }
        )

    pd.DataFrame(range_rows).to_csv(range_file, index=False)
    pd.DataFrame(normal_rows).to_csv(normal_file, index=False)

    # QA
    qa = syn[["EC", "TDS", "pH", "Cr"]].corr()["Cr"]
    qa_row = {
        "corr(Cr, EC)": float(qa["EC"]),
        "corr(Cr, TDS)": float(qa["TDS"]),
        "corr(Cr, pH)": float(qa["pH"]),
    }
    pd.DataFrame([qa_row]).to_csv(qa_file, index=False)

    # New documentation markdown (separate file)
    doc = f"""# Dokumentasi Pembuatan Dataset Synthetic Cr (Multisource)

## 1. Tujuan
Dokumen ini menjelaskan pembuatan dataset synthetic baru untuk estimasi Cr berbasis TDS, EC, dan pH dengan pendekatan **multisource**.

## 2. Sumber Data Riil yang Digunakan
Sumber yang dipakai sebagai referensi rentang:

1. `Dataset/UNEP GEMSWater Global Freshwater Quality Archive/GFQA_v3/`
   - `GEMStat_station_metadata.csv` (kategori air)
   - `Electrical_Conductance.csv` (EC)
   - `Water.csv` (TDS)
   - `pH.csv` (pH)
   - `Temperature.csv` (Suhu Air dan Suhu Lingkungan)
   - `Chromium.csv` (Cr)
2. `Dataset/A Comprehensive Surface Water Quality Monitoring Dataset (1940-2023)/Dataset/Combined Data/Combined_dataset.csv`
   - Kontributor tambahan pH dan Suhu Air untuk kategori Sungai dan Danau.
3. `Dataset/Tabel 1 dalam dokumen Zenodo Nigeria/water_data.csv`
   - Kontributor tambahan pH dan TDS untuk kategori Sungai dan Air tanah.
4. `Dataset/Water Quality Monitoring Dataset for Tilapia (Oreochromis niloticus) Aquaculture in Montería, Colombia (2024)/Monteria_Aquaculture_Data.csv`
   - Kontributor pH dan Suhu Air untuk kategori Akuakultur.
5. `Dataset/Water Quality Pollution Indices for Heavy Metal Contamination Monitoring/Data.csv`
   - Kontributor tambahan Cr untuk konteks Sungai.

## 3. Kategori
Kategori yang dipakai:
- Sungai
- Danau
- Waduk
- Air tanah
- Akuakultur

## 4. Metode Rentang
- Rentang robust utama: **p10-p90** per kategori.
- Nilai normal operasional: **p25-p75**.
- Jika variabel kategori tidak tersedia penuh (contoh Akuakultur untuk EC/TDS), digunakan fallback konservatif berbasis kategori permukaan terdekat.

## 5. Metode Pembangkitan Variabel
- EC: log-uniform (skewed) pada rentang kategori.
- TDS: dibentuk dari EC (`TDS ~ EC x faktor konversi x noise kecil`).
- pH: normal + clipping ke rentang kategori.
- Suhu Air dan Suhu Lingkungan: variasi musiman + noise.
- Kelembapan: synthetic, terkait Suhu Lingkungan.
- Tegangan: synthetic, rentang realistis sensor/baterai.
- Cr: target berbasis EC, TDS, pH, dengan efek kondisi asam dan noise kecil.

## 6. Jumlah Data
- Total baris: `{len(syn)}`
- Distribusi kategori:
{syn["Kategori"].value_counts().to_string()}

## 7. QA Korelasi
- corr(Cr, EC) = {qa_row['corr(Cr, EC)']:.6f}
- corr(Cr, TDS) = {qa_row['corr(Cr, TDS)']:.6f}
- corr(Cr, pH) = {qa_row['corr(Cr, pH)']:.6f}

## 8. Nilai Normal Tiap Parameter
Nilai normal (p25-p75) per kategori disimpan pada:
- `Dataset/Synthetic_Multisource/ringkasan_nilai_normal_multisource_p25_p75.csv`

## 9. File Output (Baru)
- `Dataset/Synthetic_Multisource/synthetic_cr_dataset_multisource.csv`
- `Dataset/Synthetic_Multisource/synthetic_cr_dataset_multisource_with_category.csv`
- `Dataset/Synthetic_Multisource/ringkasan_kontribusi_sumber_multisource.csv`
- `Dataset/Synthetic_Multisource/ringkasan_rentang_multisource_p10_p90.csv`
- `Dataset/Synthetic_Multisource/ringkasan_nilai_normal_multisource_p25_p75.csv`
- `Dataset/Synthetic_Multisource/qa_korelasi_multisource.csv`
- `dokumentasi_pembuatan_dataset_synthetic_cr_multisource.md`

## 10. Catatan Audit
File lama dibiarkan apa adanya. Semua keluaran multisource ditulis ke folder dan nama file baru.
"""
    doc_file.write_text(doc, encoding="utf-8")

    print("DONE")
    print(f"Rows total: {len(syn)}")
    print("Rows per category:")
    print(syn["Kategori"].value_counts().to_string())
    print("QA:")
    for k, v in qa_row.items():
        print(f"{k}: {v:.6f}")
    print("Output:")
    print(main_file)
    print(audit_file)
    print(contrib_file)
    print(range_file)
    print(normal_file)
    print(qa_file)
    print(doc_file)


if __name__ == "__main__":
    main()
