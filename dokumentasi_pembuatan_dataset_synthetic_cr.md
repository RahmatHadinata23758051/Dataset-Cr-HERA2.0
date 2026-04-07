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
- corr(Cr, EC) = 0.701974
- corr(Cr, TDS) = 0.690278
- corr(Cr, pH) = -0.157699

## 8. Daftar File Output
- `Dataset/Synthetic/synthetic_cr_dataset.csv`
- `Dataset/Synthetic/synthetic_cr_dataset_with_category.csv`
- `Dataset/Synthetic/ringkasan_rentang_kategori_p10_p90.csv`
- `Dataset/Synthetic/qa_korelasi_synthetic_cr.csv`
- `dokumentasi_pembuatan_dataset_synthetic_cr.md`

## 9. Kesimpulan
Dataset synthetic ini dibangun dari rentang data riil dan dapat diaudit secara akademik, namun bukan salinan baris data asli.
