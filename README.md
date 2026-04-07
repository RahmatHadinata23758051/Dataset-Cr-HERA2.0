# README Dataset Synthetic Cr

Dokumen ini menjelaskan:
1. **bagaimana dataset synthetic dibuat**, dan  
2. **dari dataset riil mana rentang tiap parameter diambil**.

## 1. Ringkasan Hasil

Project ini memiliki 2 versi hasil synthetic:

1. `Dataset/Synthetic` (single-source, referensi utama GFQA_v3)
2. `Dataset/Synthetic_Multisource` (gabungan beberapa dataset riil)

## 2. Cara Dataset Synthetic Dibuat

Semua dataset synthetic dibuat dengan alur umum berikut:

1. Identifikasi kategori perairan (mis. Sungai, Danau, Waduk, Air tanah, Akuakultur).
2. Ambil nilai parameter dari dataset riil.
3. Hitung rentang robust per kategori (utama: **p10-p90**, nilai normal: **p25-p75**).
4. Generate data synthetic baru (bukan copy data asli):
   - `EC` dibangkitkan dengan distribusi skewed/log-uniform.
   - `TDS` diturunkan dari `EC` (`TDS ~ EC x faktor konversi x noise kecil`).
   - `pH` dibangkitkan dalam rentang kategori.
   - `Suhu Air` dan `Suhu Lingkungan` dibentuk dengan variasi musiman + noise.
   - `Kelembapan` dan `Tegangan` dibentuk synthetic (karena tidak selalu ada di data riil).
   - `Cr` dibentuk sebagai target dari `EC`, `TDS`, dan `pH` dengan noise kecil.
5. Simpan file output + file QA korelasi.

## 3. Sumber Rentang Dataset Riil (Single-Source)

Folder output: `Dataset/Synthetic`

Sumber rentang:
- `Dataset/UNEP GEMSWater Global Freshwater Quality Archive/GFQA_v3`

Pemetaan variabel -> sumber riil:
- `EC` -> `Electrical_Conductance.csv`
- `TDS` -> `Water.csv` (Parameter Code: `TDS`)
- `pH` -> `pH.csv`
- `Suhu Air (°C)` -> `Temperature.csv` (Parameter Code: `TEMP`)
- `Suhu Lingkungan (°C)` -> `Temperature.csv` (Parameter Code: `TEMP-Air`)
- Kategori air -> `GEMStat_station_metadata.csv`

File hasil:
- `synthetic_cr_dataset.csv`
- `synthetic_cr_dataset_with_category.csv`
- `ringkasan_rentang_kategori_p10_p90.csv`
- `qa_korelasi_synthetic_cr.csv`

## 4. Sumber Rentang Dataset Riil (Multisource)

Folder output: `Dataset/Synthetic_Multisource`

Sumber riil yang dipakai:
1. `Dataset/UNEP GEMSWater Global Freshwater Quality Archive/GFQA_v3`
2. `Dataset/A Comprehensive Surface Water Quality Monitoring Dataset (1940-2023)/Dataset/Combined Data/Combined_dataset.csv`
3. `Dataset/Tabel 1 dalam dokumen Zenodo Nigeria/water_data.csv`
4. `Dataset/Water Quality Monitoring Dataset for Tilapia (Oreochromis niloticus) Aquaculture in Montería, Colombia (2024)/Monteria_Aquaculture_Data.csv`
5. `Dataset/Water Quality Pollution Indices for Heavy Metal Contamination Monitoring/Data.csv`

Pemetaan kontribusi rentang:
- **UNEP GFQA_v3**: sumber utama `EC`, `TDS`, `pH`, `Suhu Air`, `Suhu Lingkungan`, dan `Cr`.
- **A Comprehensive**: tambahan rentang `pH` dan `Suhu Air` (terutama Sungai dan Danau).
- **Nigeria**: tambahan rentang `pH` dan `TDS` (Air Permukaan dan Air Tanah).
- **Monteria**: tambahan kategori `Akuakultur` untuk `pH` dan `Suhu Air`.
- **Heavy Metal Indices**: tambahan referensi `Cr` (konteks sungai).

File hasil:
- `synthetic_cr_dataset_multisource.csv`
- `synthetic_cr_dataset_multisource_with_category.csv`
- `ringkasan_kontribusi_sumber_multisource.csv`
- `ringkasan_rentang_multisource_p10_p90.csv`
- `ringkasan_nilai_normal_multisource_p25_p75.csv`
- `qa_korelasi_multisource.csv`

## 5. Format Kolom Dataset Hasil

Urutan kolom utama (konsisten):

`Tanggal | Waktu | Tegangan (V) | Suhu Air (°C) | Suhu Lingkungan (°C) | Kelembapan Lingkungan (%) | TDS | EC | pH | Cr`

Catatan:
- `Cr` selalu di kolom terakhir.
- `Tegangan` dan `Kelembapan` adalah variabel synthetic yang dibuat realistis.

## 6. Dokumen Metodologi Lengkap

Detail lengkap ada di:
- `dokumentasi_pembuatan_dataset_synthetic_cr.md`
- `dokumentasi_pembuatan_dataset_synthetic_cr_multisource.md`
