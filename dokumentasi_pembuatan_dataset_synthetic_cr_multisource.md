# Dokumentasi Pembuatan Dataset Synthetic Cr (Multisource)

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
- Total baris: `120`
- Distribusi kategori:
Kategori
Danau         24
Sungai        24
Waduk         24
Air tanah     24
Akuakultur    24

## 7. QA Korelasi
- corr(Cr, EC) = 0.705653
- corr(Cr, TDS) = 0.692545
- corr(Cr, pH) = -0.086327

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
