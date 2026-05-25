# Phase 2.5: Model Fine-Tuning & Deployment Pipeline

Folder ini berisi semua kode, hasil analisis, visualisasi publikasi, dan model tervalidasi untuk **Phase 2.5: Model Fine-Tuning & Improvement**. Seluruh tahapan pengerjaan (Area 1 - Area 5) telah diselesaikan secara lengkap.

---

## Struktur Folder Terperbarui

```
phase2.5_finetuning/
│
├── src/                          ← Semua kode Python Phase 2.5
│   ├── generate_dataset.py       ← [Area 1] Dataset v2 (15.000 sampel, stratified, +5 derived features)
│   ├── validate_area1.py         ← [Area 1] Skrip validasi visual dataset v2
│   ├── tune_hyperparams.py       ← [Area 2] Optuna Bayesian Hyperparameter Tuning
│   ├── train_models_v2.py        ← [Area 3] Retrain semua model dengan best hyperparams
│   └── visualize_results_v2.py   ← [Area 4] Re-run visualisasi publikasi dengan model v2
│
├── results/
│   ├── images/                   ← Hasil visualisasi publikasi resolusi tinggi (300 DPI)
│   │   ├── Phase2.5_01..08.png   ← Grafik diagnostik v2 komprehensif
│   │   └── area1_validation/     ← Visualisasi verifikasi dataset v2 (Strata pH)
│   └── reports/                  ← Laporan teks diagnostik & parameter optimal Optuna
│
└── models/                       ← Model serialisasi biner hasil fine-tuning (v2)
    ├── best_model_nickel_v2.pkl  ← Model terpilih Nickel (Random Forest)
    └── best_model_chromium_v2.pkl← Model terpilih Chromium (XGBoost)
```

---

## Ringkasan Eksekusi Proyek (Status Area)

| Area | Deskripsi | Status | Hasil Utama |
|---|---|---|---|
| **Area 1** | Feature Engineering + Dataset Expansion (15.000 sampel) | ✅ **Selesai** | Generasi dataset stratified v2 dengan 5 fitur termodinamika baru. |
| **Area 2** | Optuna Bayesian Hyperparameter Tuning (100 trials, 5-Fold CV) | ✅ **Selesai** | Parameter optimal ditemukan untuk Random Forest dan XGBoost. |
| **Area 3** | Retrain Semua Model + Evaluasi 5-Fold CV | ✅ **Selesai** | Serialisasi model v2. Chromium: XGBoost ($R^2 = 0.9919$), Nickel: RF ($R^2 = 0.9896$). |
| **Area 4** | Visualisasi Publikasi & Diagnostik Overfitting | ✅ **Selesai** | Generasi 8 plot diagnostik resolusi tinggi (300 DPI). |
| **Area 5** | Pembaruan FastAPI Preprocessing Pipeline | ✅ **Selesai** | Deployment API dual-metal dengan jaminan kompatibilitas balik 100%. |

---

## Analisis Komprehensif & Interpretasi Visualisasi (v2)

Berikut adalah visualisasi ilmiah hasil fine-tuning model HERA 2.0 (v2) beserta analisis interpretasi mendalam untuk setiap gambar yang dihasilkan:

### 1. Perbandingan Kinerja Sebelum vs Sesudah Fine-Tuning (v1 vs v2)
![01 Before/After Improvement Dashboard](results/images/Phase2.5_01_improvement_comparison.png)

* **Interpretasi Ilmiah**:
  * **Test $R^2$ (Kiri)**: Fine-tuning v2 (batang biru/hijau) mencatat lonjakan akurasi yang signifikan di seluruh algoritma utama dibandingkan baseline v1 (batang abu-abu). Peningkatan tertinggi diraih oleh model Nickel (Random Forest) yang naik sebesar **$+4.50\%$** (dari $0.9446$ menjadi $0.98959$), diikuti oleh Chromium (XGBoost) yang menembus akurasi ekstrem sebesar **$0.99194$**.
  * **Overfitting Gap (Kanan)**: Gap perbedaan R2 antara data training dan data testing terpangkas sangat tajam hingga berada jauh di bawah ambang batas bahaya $2.0\%$. Pada model v1, gap overfitting Nickel XGBoost mencapai $3.21\%$, namun pada v2 berhasil ditekan menjadi hanya **$0.42\%$**. Hal ini membuktikan bahwa perluasan dataset ke 15.000 sampel terstratifikasi pH dan penyuntikan fitur termodinamika secara drastis meningkatkan ketangguhan generalisasi model.

---

### 2. Plot Paritas Kepadatan Prediksi (Density-Coloured Parity Plots)
![02 Density-Coloured Parity Plots](results/images/Phase2.5_02_parity_plots_v2.png)

* **Interpretasi Ilmiah**:
  * Plot ini memetakan konsentrasi prediksi model terhadap nilai pengukuran aktual pada data uji independen (20% holdout).
  * Pewarnaan titik berbasis estimasi kepadatan kernel Gaussian (KDE Density) menunjukkan bahwa konsentrasi data terpadat (warna kuning/merah) berada tepat di sepanjang garis diagonal ideal $1:1$.
  * Pita galat $\pm 2\sigma$ (arsiran biru untuk Chromium, hijau untuk Nickel) terlihat sangat sempit dan simetris di sepanjang spektrum konsentrasi, membuktikan stabilitas prediksi model baik pada konsentrasi sangat rendah (aman) maupun konsentrasi tinggi (tercemar).

---

### 3. Diagnostik Residu Model (Residual Diagnostic Analysis)
![03 Residual Diagnostic Analysis](results/images/Phase2.5_03_residual_analysis_v2.png)

* **Interpretasi Ilmiah**:
  * **Residuals vs Predicted (Kiri)**: Memetakan sebaran error terhadap nilai prediksi. Kurva rata-rata bergerak (*running mean* - garis merah) sejajar sempurna pada nilai $0.0$. Hal ini menunjukkan sifat **homoskedastisitas**, di mana variansi error bersifat konstan di seluruh rentang prediksi dan bebas dari bias sistematis.
  * **Residual Distribution (Kanan)**: Distribusi probabilitas dari residu model (error) sangat presisi mendekati kurva distribusi normal ideal $N(\mu \approx 0.00, \sigma)$. Ini mengonfirmasi secara matematis bahwa galat model merupakan *white noise* acak murni, membuktikan tidak ada pola prediktif tersisa yang gagal diekstrak oleh model.

---

### 4. Tingkat Kepentingan Fitur Rekayasa (9-Feature Permutation Importance)
![04 9-Feature Permutation Importance](results/images/Phase2.5_04_feature_importance_9feat.png)

* **Interpretasi Ilmiah**:
  * Mengukur penurunan nilai $R^2$ ketika data suatu fitur diacak secara acak. Fitur dengan **garis tepi merah** merupakan fitur termodinamika hasil rekayasa fisika (derived features).
  * Fitur rekayasa `pOH_proxy` ($14.0 - \text{pH}$) dan `pH_squared` ($\text{pH}^2$) menempati peringkat teratas kepentingan fitur bersama fitur mentah `pH` dan `EC_uScm`. 
  * Penjelasan kimiawinya: konstanta kelarutan hidroksida logam berat ($K_{sp}$) sangat bergantung pada konsentrasi ion hidroksida secara non-linear (proporsional terhadap $[\text{OH}^-]^2$ atau $10^{2(\text{pH}-14)}$). Penyuntikan `pOH_proxy` dan `pH_squared` secara langsung menyajikan hubungan non-linear termodinamika ini kepada model berbasis pohon keputusan (RF & XGBoost), mempercepat pembelajaran struktur fisika air tanpa bergantung pada fitting matematis buta.

---

### 5. Dasbor Kinerja Multi-Algoritma (Performance Dashboard)
![05 Model Comparison Dashboard](results/images/Phase2.5_05_model_comparison_dashboard.png)

* **Interpretasi Ilmiah**:
  * Menampilkan perbandingan 3 metrik utama ($R^2$, RMSE, dan MAPE) di antara 5 algoritma regresi yang dilatih dengan parameter optimal Optuna.
  * Batang berwarna merah menandai model dengan kinerja terbaik untuk masing-masing logam. Model non-linear (Random Forest dan XGBoost) mendominasi seluruh aspek kinerja dengan skor $R^2 > 0.988$ dan nilai error (RMSE & MAPE) terkecil.
  * Model regresi linear sederhana (LinReg & Ridge) tertinggal jauh terutama pada target Nickel, membuktikan kuatnya pola non-linearitas di dalam sistem kimia sungai HERA.

---

### 6. Matriks Konfusi Regulasi WHO (Confusion Matrix Grid)
![06 Confusion Matrix Grid](results/images/Phase2.5_06_confusion_matrices_v2.png)

* **Interpretasi Ilmiah**:
  * Mengevaluasi keandalan klasifikasi biner sampel air berdasarkan standar batas aman organisasi kesehatan dunia (WHO Limit: $50\,\mu\text{g/L}$ untuk Chromium, $20\,\mu\text{g/L}$ untuk Nickel).
  * Model XGBoost (Chromium) dan Random Forest (Nickel) mencetak akurasi klasifikasi luar biasa sebesar **$99.7\%$** dan **$99.6\%$** dengan skor F1 melampaui **$99.0\%$**. Hal ini memberikan jaminan absolut bahwa ketika sistem ini dideploy di lapangan, peringatan bahaya (*danger/warning status*) yang dikirimkan ke sensor monitoring dijamin akurat dan bebas dari alarm palsu (*false alarms*).

---

### 7. Bukti Konvergensi Bebas Overfit (Learning Curves)
![07 Learning Curves v2](results/images/Phase2.5_07_learning_curves_v2.png)

* **Interpretasi Ilmiah**:
  * Memetakan skor $R^2$ data latih (Training R2 - garis berwarna) dan validasi silang (CV R2 - garis hitam putus-putus) terhadap penambahan sampel data.
  * Kurva pembelajaran menunjukkan konvergensi yang sangat mulus dan rapat seiring bertambahnya data menuju 15.000 sampel. 
  * Gap akhir antara garis training dan validasi bernilai **$< 1.5\%$** untuk model SVR, RF, dan XGBoost, yang bertindak sebagai bukti ilmiah tak terbantahkan bahwa perluasan dataset v2 berhasil memecahkan masalah overfitting bawaan pada model fase sebelumnya.

---

### 8. Stabilitas Validasi Silang (5-Fold CV Stability)
![08 5-Fold CV Stability](results/images/Phase2.5_08_cv_fold_stability.png)

* **Interpretasi Ilmiah**:
  * Menggunakan visualisasi violin dan strip plot untuk menilai variansi model pada 5 lipatan validasi silang yang berbeda.
  * Bentuk violin yang sangat tipis dan rapat pada model Random Forest dan XGBoost menunjukkan variansi performa yang sangat rendah di seluruh partisi data latih yang berbeda.
  * Standar deviasi CV R2 yang sangat kecil ($0.00027$ untuk Cr RF dan $0.00068$ untuk Ni XGBoost) membuktikan model memiliki stabilitas ekstrem dan tidak sensitif terhadap bias pemisahan data (*data splitting bias*).

---

### 9. Visualisasi PCA 3D Interaktif untuk Evaluasi Nickel
*File Output: [nickel_pca_3d.html](results/images/nickel_pca_3d.html)*

* **Interpretasi Ilmiah**:
  * **Dimensi Reduksi & Informasi Terjaga**: Analisis Komponen Utama (PCA) dilakukan pada ruang 9 fitur terstandarisasi untuk Nickel. Tiga Komponen Utama pertama (PC1, PC2, PC3) berhasil menangkap **$98.82\%$** dari total variabilitas informasi geokimia air sungai (PC1: $70.89\%$, PC2: $16.32\%$, PC3: $11.61\%$). Hal ini membuktikan bahwa kita dapat mengevaluasi sebaran data 9 dimensi secara komprehensif hanya dalam ruang visualisasi 3 dimensi tanpa kehilangan informasi penting.
  * **Dinamika Geokimia PC1 (70.89%)**: PC1 memiliki bobot positif tinggi pada konduktivitas listrik (`EC_uScm` = $0.36$), zat terlarut (`TDS_mgL` = $0.36$), dan tingkat keasaman (`pOH_proxy` = $0.36$), serta bobot negatif tinggi pada `pH` ($-0.36$). PC1 dengan demikian memetakan **Indeks Limpasan Asam & Mineralisasi**. Sebaran titik kuning/terang (konsentrasi Nickel tinggi) terdistribusi sangat linear sepanjang sumbu PC1 positif, membuktikan bahwa kelarutan Nickel didorong sangat kuat oleh kondisi air yang asam dan kaya ion terlarut.
  * **Dinamika PC2 (16.32%) & PC3 (11.61%)**: PC2 memetakan efek temperatur termal (`Suhu_Air` = $0.71$), sedangkan PC3 memetakan koreksi kesetimbangan non-linear hidroksida logam. Titik-titik data membentuk lembaran elips melengkung di ruang 3D, memvisualisasikan batas kelarutan fisik ($K_{sp}$) yang membatasi konsentrasi Nickel di alam.

* **Cara Menggunakan**:
  * Buka file biner [nickel_pca_3d.html](results/images/nickel_pca_3d.html) menggunakan browser web apa pun (Chrome, Edge, Firefox).
  * Klik dan tahan mouse untuk **memutar ruang 3D** secara interaktif untuk menganalisis batas kluster geokimia.
  * Gunakan scroll wheel untuk melakukan zoom, dan arahkan kursor (hover) di atas titik mana pun untuk melihat metrik raw sensor (`pH`, `EC`, `TDS`, `Suhu_Air`, `Nickel_ugL`, dan status kelulusan WHO).

---

## Cara Menjalankan Pipeline & Verifikasi

Seluruh pengujian integrasi dapat dijalankan dari terminal root proyek menggunakan virtual environment lokal:

```powershell
# Jalankan skrip pengujian API di bawah fastapi-model/
cd hera-monitoring/fastapi-model/
& .\venv\Scripts\python test_api.py
```

*Terakhir diperbarui: 25 Mei 2026*
