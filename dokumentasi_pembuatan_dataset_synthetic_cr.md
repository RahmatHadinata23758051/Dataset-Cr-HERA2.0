# Dokumentasi Pembuatan Dataset Synthetic Cr

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

## 5. Nilai Data Normal Tiap Parameter

Pada konteks dokumen ini, **nilai data normal** didefinisikan sebagai rentang **p25-p75 (interquartile range/IQR)**.  
Alasan penggunaan p25-p75 adalah untuk merepresentasikan nilai tengah yang paling sering muncul, sehingga lebih stabil dibanding min-max.

### 5.1 Nilai Normal dari Data Riil (p25-p75, per kategori)

Sumber: file riil UNEP/GEMStat yang dijelaskan pada Bagian 2.

| Kategori | Suhu Air (°C) | Suhu Lingkungan (°C) | TDS | EC | pH |
|---|---|---|---|---|---|
| Sungai | 9.50 - 24.00 | 20.30 - 30.40 | 104.32 - 441.60 | 180.00 - 590.00 | 7.48 - 8.15 |
| Danau | 13.40 - 27.40 | 26.00 - 32.00 | 9.10 - 14128.00 | 116.68 - 452.00 | 7.30 - 8.28 |
| Waduk | 19.90 - 27.00 | 22.90 - 30.30 | 113.28 - 327.04 | 162.00 - 453.23 | 7.60 - 8.32 |
| Air tanah | 13.00 - 22.10 | 23.70 - 30.20 | 310.00 - 852.00 | 410.00 - 856.00 | 7.11 - 7.70 |

Catatan:

- Rentang danau lebih lebar terutama pada TDS karena heterogenitas badan air global (termasuk danau dengan mineralisasi tinggi).

### 5.2 Nilai Normal Parameter Synthetic (p25-p75, per kategori)

Kedua parameter ini tidak tersedia pada data riil, sehingga dibangkitkan synthetic dengan kendali fisik.

| Kategori | Kelembapan Lingkungan (%) | Tegangan (V) |
|---|---|---|
| Sungai | 69.93 - 80.28 | 3.68 - 3.97 |
| Danau | 70.22 - 82.45 | 3.66 - 4.04 |
| Waduk | 66.23 - 75.83 | 3.72 - 4.02 |
| Air tanah | 61.03 - 72.35 | 3.72 - 4.11 |

### 5.3 Nilai Normal Target Cr pada Dataset Synthetic (p25-p75)

| Kategori | Cr |
|---|---|
| Sungai | 18.98 - 23.45 |
| Danau | 16.50 - 34.76 |
| Waduk | 18.08 - 22.72 |
| Air tanah | 22.18 - 25.80 |

Interpretasi:

- Rentang normal Cr berbeda antar kategori karena perbedaan distribusi EC, TDS, dan pH.
- Nilai Cr tidak dibentuk acak penuh; nilai tersebut mengikuti pola prediktor dengan noise kecil.

## 6. Metode Pembangkitan Variabel
- EC: log-uniform pada rentang p10-p90 per kategori
- TDS: diturunkan dari EC (`TDS ~ EC x faktor x noise kecil`)
- pH: normal + clip pada rentang kategori
- Suhu Air dan Suhu Lingkungan: variasi musiman + noise
- Kelembapan: synthetic, terkait suhu lingkungan
- Tegangan: synthetic, rentang realistis sensor/baterai
- Cr: fungsi dari EC, TDS, pH + noise kecil (tidak terlalu sempurna)

## 7. Asumsi
- Kelembapan dan Tegangan tidak ada di data riil, maka dibuat synthetic
- Hubungan TDS-EC dijaga konsisten secara fisik
- Cr dibentuk logis dari prediktor, bukan acak murni

## 8. Kualitas Dataset Synthetic (QA Korelasi)
- corr(Cr, EC) = 0.701974
- corr(Cr, TDS) = 0.690278
- corr(Cr, pH) = -0.157699

## 9. Daftar File Output
- `Dataset/Synthetic/synthetic_cr_dataset.csv`
- `Dataset/Synthetic/synthetic_cr_dataset_with_category.csv`
- `Dataset/Synthetic/ringkasan_rentang_kategori_p10_p90.csv`
- `Dataset/Synthetic/qa_korelasi_synthetic_cr.csv`
- `dokumentasi_pembuatan_dataset_synthetic_cr.md`

## 10. Kesimpulan
Dataset synthetic ini dibangun dari rentang data riil dan dapat diaudit secara akademik, namun bukan salinan baris data asli.
