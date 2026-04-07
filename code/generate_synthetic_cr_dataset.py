#!/usr/bin/env python3
"""
Generate synthetic Chromium (Cr) dataset from real-data robust ranges.

Outputs:
1) Dataset/Synthetic/synthetic_cr_dataset.csv
2) Dataset/Synthetic/synthetic_cr_dataset_with_category.csv
3) Dataset/Synthetic/ringkasan_rentang_kategori_p10_p90.csv
4) Dataset/Synthetic/qa_korelasi_synthetic_cr.csv
5) dokumentasi_pembuatan_dataset_synthetic_cr.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def extract_robust_ranges(gfqa_dir: Path) -> tuple[dict, dict]:
    """Extract p10-p90 ranges from real UNEP/GEMStat data by water category."""
    station_file = gfqa_dir / "GEMStat_station_metadata.csv"
    station = pd.read_csv(
        station_file,
        usecols=["GEMS Station Number", "Water Type"],
        dtype=str,
        encoding="latin1",
    )

    category_map = {
        "River station": "Sungai",
        "Lake station": "Danau",
        "Reservoir station": "Waduk",
        "Groundwater station": "Air tanah",
    }
    station["Kategori"] = station["Water Type"].map(category_map)
    station = station.dropna(subset=["Kategori"])
    station_map = dict(zip(station["GEMS Station Number"], station["Kategori"]))

    categories = ["Sungai", "Danau", "Waduk", "Air tanah"]
    spec = [
        ("EC", "Electrical_Conductance.csv", "EC"),
        ("TDS", "Water.csv", "TDS"),
        ("pH", "pH.csv", "pH"),
        ("Suhu Air (°C)", "Temperature.csv", "TEMP"),
        ("Suhu Lingkungan (°C)", "Temperature.csv", "TEMP-Air"),
    ]

    ranges = {c: {} for c in categories}
    counts = {c: {} for c in categories}

    for variable, filename, code in spec:
        values = {c: [] for c in categories}
        source_file = gfqa_dir / filename

        for chunk in pd.read_csv(
            source_file,
            usecols=["GEMS Station Number", "Parameter Code", "Value"],
            chunksize=300000,
            encoding="latin1",
        ):
            chunk = chunk[chunk["Parameter Code"] == code]
            if chunk.empty:
                continue

            chunk["Kategori"] = chunk["GEMS Station Number"].map(station_map)
            chunk = chunk.dropna(subset=["Kategori"])
            chunk["Value"] = pd.to_numeric(chunk["Value"], errors="coerce")
            chunk = chunk.dropna(subset=["Value"])

            for cat, grp in chunk.groupby("Kategori"):
                values[cat].extend(grp["Value"].tolist())

        for cat in categories:
            arr = np.array(values[cat], dtype=float)
            counts[cat][variable] = int(arr.size)
            if arr.size == 0:
                ranges[cat][variable] = (np.nan, np.nan)
            else:
                ranges[cat][variable] = (
                    float(np.percentile(arr, 10)),
                    float(np.percentile(arr, 90)),
                )

    # Synthetic-only variables (not available in real dataset)
    humidity_ranges = {
        "Sungai": (55.0, 95.0),
        "Danau": (50.0, 92.0),
        "Waduk": (48.0, 90.0),
        "Air tanah": (45.0, 88.0),
    }
    for cat in categories:
        ranges[cat]["Kelembapan Lingkungan (%)"] = humidity_ranges[cat]
        ranges[cat]["Tegangan (V)"] = (3.45, 4.25)

    return ranges, counts


def generate_synthetic_dataset(
    ranges: dict,
    n_per_category: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic dataset with realistic physical/statistical relations."""
    np.random.seed(seed)
    categories = ["Sungai", "Danau", "Waduk", "Air tanah"]

    cr_bounds = {
        "Sungai": (2.0, 95.0),
        "Danau": (3.0, 140.0),
        "Waduk": (2.0, 85.0),
        "Air tanah": (5.0, 120.0),
    }

    rows = []
    start_date = datetime(2025, 1, 1)
    time_slots = [
        "06:00:00",
        "07:00:00",
        "08:00:00",
        "09:00:00",
        "12:00:00",
        "13:00:00",
        "14:00:00",
        "17:00:00",
        "18:00:00",
    ]

    for cat in categories:
        ec_lo, ec_hi = ranges[cat]["EC"]
        tds_lo, tds_hi = ranges[cat]["TDS"]
        ph_lo, ph_hi = ranges[cat]["pH"]
        wt_lo, wt_hi = ranges[cat]["Suhu Air (°C)"]
        at_lo, at_hi = ranges[cat]["Suhu Lingkungan (°C)"]
        h_lo, h_hi = ranges[cat]["Kelembapan Lingkungan (%)"]
        v_lo, v_hi = ranges[cat]["Tegangan (V)"]
        cr_lo, cr_hi = cr_bounds[cat]

        mid_wt = (wt_lo + wt_hi) / 2.0
        amp_wt = (wt_hi - wt_lo) * 0.22
        mid_at = (at_lo + at_hi) / 2.0

        for _ in range(n_per_category):
            day = int(np.random.randint(0, 365))
            d = start_date + timedelta(days=day)
            hm = np.random.choice(time_slots)
            ts = datetime.strptime(d.strftime("%Y-%m-%d") + " " + hm, "%Y-%m-%d %H:%M:%S")
            season = np.sin(2 * np.pi * (ts.timetuple().tm_yday / 365.0))

            # EC: skewed/log-uniform
            ec = float(np.exp(np.random.uniform(np.log(ec_lo), np.log(ec_hi))))

            # TDS ~ EC * factor * small noise
            k = np.random.uniform(0.55, 0.82) if cat == "Danau" else np.random.uniform(0.50, 0.78)
            tds = ec * k * np.random.normal(1.0, 0.06)
            tds = clip(tds, tds_lo, tds_hi)

            # pH in category robust range
            ph = np.random.normal((ph_lo + ph_hi) / 2.0, (ph_hi - ph_lo) / 5.0)
            ph = clip(ph, ph_lo, ph_hi)

            # Water/Air temperature
            wt = clip(mid_wt + amp_wt * season + np.random.normal(0, 1.2), wt_lo, wt_hi)
            at = clip(wt + np.random.uniform(1.2, 6.5) + np.random.normal(0, 0.8), at_lo, at_hi)

            # Humidity synthetic (linked to air temp)
            hum = clip(np.random.normal((h_lo + h_hi) / 2.0, 8.0) - 0.7 * (at - mid_at), h_lo, h_hi)

            # Voltage synthetic (sensor/battery realistic range)
            volt = clip(3.55 + 0.65 * np.random.rand() - 0.007 * (at - mid_at) + np.random.normal(0, 0.02), v_lo, v_hi)

            # Chromium target:
            # +EC, +TDS, and pH effect (acidic condition tends higher dissolved Cr), with noise
            acid = max(0.0, 7.2 - ph)
            cr = 2.2 * np.log1p(ec) + 1.75 * np.log1p(tds) - 2.0 * (ph - 7.4) + 3.5 * acid + np.random.normal(0, 1.5)
            cr = clip(cr, cr_lo, cr_hi)

            rows.append(
                {
                    "Kategori": cat,
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
                }
            )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def write_outputs(base_dir: Path, ranges: dict, synthetic_df: pd.DataFrame) -> dict:
    out_dir = base_dir / "Dataset" / "Synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_cols = [
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

    f_main = out_dir / "synthetic_cr_dataset.csv"
    f_audit = out_dir / "synthetic_cr_dataset_with_category.csv"
    f_range = out_dir / "ringkasan_rentang_kategori_p10_p90.csv"
    f_qa = out_dir / "qa_korelasi_synthetic_cr.csv"
    f_doc = base_dir / "dokumentasi_pembuatan_dataset_synthetic_cr.md"

    synthetic_df[main_cols].to_csv(f_main, index=False)
    synthetic_df.to_csv(f_audit, index=False)

    summary_rows = []
    for cat in ["Sungai", "Danau", "Waduk", "Air tanah"]:
        summary_rows.append(
            {
                "Kategori": cat,
                "Suhu Air (°C)": f"{ranges[cat]['Suhu Air (°C)'][0]:.2f} - {ranges[cat]['Suhu Air (°C)'][1]:.2f}",
                "Suhu Lingkungan (°C)": f"{ranges[cat]['Suhu Lingkungan (°C)'][0]:.2f} - {ranges[cat]['Suhu Lingkungan (°C)'][1]:.2f}",
                "TDS": f"{ranges[cat]['TDS'][0]:.2f} - {ranges[cat]['TDS'][1]:.2f}",
                "EC": f"{ranges[cat]['EC'][0]:.2f} - {ranges[cat]['EC'][1]:.2f}",
                "pH": f"{ranges[cat]['pH'][0]:.2f} - {ranges[cat]['pH'][1]:.2f}",
                "Kelembapan (%)*": f"{ranges[cat]['Kelembapan Lingkungan (%)'][0]:.0f} - {ranges[cat]['Kelembapan Lingkungan (%)'][1]:.0f}",
                "Tegangan (V)**": f"{ranges[cat]['Tegangan (V)'][0]:.2f} - {ranges[cat]['Tegangan (V)'][1]:.2f}",
            }
        )
    pd.DataFrame(summary_rows).to_csv(f_range, index=False)

    corr = synthetic_df[["EC", "TDS", "pH", "Cr"]].corr()["Cr"]
    qa_row = {
        "corr(Cr, EC)": float(corr["EC"]),
        "corr(Cr, TDS)": float(corr["TDS"]),
        "corr(Cr, pH)": float(corr["pH"]),
    }
    pd.DataFrame([qa_row]).to_csv(f_qa, index=False)

    doc_text = f"""# Dokumentasi Pembuatan Dataset Synthetic Cr

## 1. Tujuan Pembuatan Dataset Synthetic
Dataset synthetic dibuat untuk estimasi kadar Cr (Kromium) dari TDS, EC, dan pH, agar dapat digunakan untuk analisis korelasi, regresi, dan machine learning tanpa menyalin data riil.

## 2. Sumber Dataset Riil
Sumber rentang berasal dari:
- `Dataset/UNEP GEMSWater Global Freshwater Quality Archive/GFQA_v3/GEMStat_station_metadata.csv` (kategori air)
- `Dataset/UNEP GEMSWater Global Freshwater Quality Archive/GFQA_v3/Electrical_Conductance.csv` (EC)
- `Dataset/UNEP GEMSWater Global Freshwater Quality Archive/GFQA_v3/Water.csv` (TDS, Parameter Code = TDS)
- `Dataset/UNEP GEMSWater Global Freshwater Quality Archive/GFQA_v3/pH.csv` (pH)
- `Dataset/UNEP GEMSWater Global Freshwater Quality Archive/GFQA_v3/Temperature.csv` (TEMP dan TEMP-Air)

## 3. Kategori yang Digunakan
Kategori ditentukan dari Water Type:
- River station -> Sungai
- Lake station -> Danau
- Reservoir station -> Waduk
- Groundwater station -> Air tanah

## 4. Metode Pengambilan Rentang
Rentang variabel diambil dengan robust p10-p90 per kategori untuk mengurangi dampak outlier ekstrem.

## 5. Metode Pembangkitan Variabel
- EC: log-uniform pada rentang p10-p90 per kategori
- TDS: diturunkan dari EC (`TDS ~ EC x faktor x noise kecil`)
- pH: normal + clip pada rentang kategori
- Suhu Air dan Suhu Lingkungan: variasi musiman + noise
- Kelembapan: synthetic, terkait suhu lingkungan
- Tegangan: synthetic, rentang realistis sensor/baterai
- Cr: fungsi dari EC, TDS, pH + noise kecil (tidak terlalu sempurna)

## 6. Asumsi
- Kelembapan dan Tegangan tidak ada di data riil, maka dibuat synthetic
- Hubungan TDS-EC dijaga konsisten secara fisik
- Cr dibentuk logis dari prediktor, bukan acak murni

## 7. Kualitas Dataset Synthetic (QA Korelasi)
- corr(Cr, EC) = {qa_row['corr(Cr, EC)']:.6f}
- corr(Cr, TDS) = {qa_row['corr(Cr, TDS)']:.6f}
- corr(Cr, pH) = {qa_row['corr(Cr, pH)']:.6f}

## 8. Daftar File Output
- `Dataset/Synthetic/synthetic_cr_dataset.csv`
- `Dataset/Synthetic/synthetic_cr_dataset_with_category.csv`
- `Dataset/Synthetic/ringkasan_rentang_kategori_p10_p90.csv`
- `Dataset/Synthetic/qa_korelasi_synthetic_cr.csv`
- `dokumentasi_pembuatan_dataset_synthetic_cr.md`

## 9. Kesimpulan
Dataset synthetic ini dibangun dari rentang data riil dan dapat diaudit secara akademik, namun bukan salinan baris data asli.
"""
    f_doc.write_text(doc_text, encoding="utf-8")

    return {
        "main": f_main,
        "audit": f_audit,
        "range": f_range,
        "qa": f_qa,
        "doc": f_doc,
        "qa_values": qa_row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Cr dataset from GFQA real ranges.")
    parser.add_argument("--base-dir", type=str, default=".", help="Project root directory (default: current directory)")
    parser.add_argument("--rows", type=int, default=120, help="Total synthetic rows (default: 120)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    gfqa_dir = base_dir / "Dataset" / "UNEP GEMSWater Global Freshwater Quality Archive" / "GFQA_v3"
    if not gfqa_dir.exists():
        raise FileNotFoundError(f"GFQA folder not found: {gfqa_dir}")

    categories = 4
    n_per_category = max(1, args.rows // categories)

    ranges, _counts = extract_robust_ranges(gfqa_dir)
    synthetic_df = generate_synthetic_dataset(ranges, n_per_category=n_per_category, seed=args.seed)
    outputs = write_outputs(base_dir, ranges, synthetic_df)

    print("Generation complete.")
    print(f"Rows total: {len(synthetic_df)}")
    print("Rows per category:")
    print(synthetic_df["Kategori"].value_counts().to_string())
    print("QA correlations:")
    for k, v in outputs["qa_values"].items():
        print(f"- {k}: {v:.6f}")
    print("Output files:")
    print(f"- {outputs['main']}")
    print(f"- {outputs['audit']}")
    print(f"- {outputs['range']}")
    print(f"- {outputs['qa']}")
    print(f"- {outputs['doc']}")


if __name__ == "__main__":
    main()

