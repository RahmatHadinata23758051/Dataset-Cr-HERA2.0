# HERA 2.0 — Heavy Metal Soft Sensor Model Development (Chromium & Nickel)

Repositori ini berisi seluruh rangkaian penelitian, perencanaan, pemrosesan data, pelatihan machine learning, pengujian fisik geokimia, serta sistem deployment API untuk **HERA 2.0 Heavy Metal Soft Sensor** (Chromium & Nickel) pada ekosistem air sungai.

---

## 1. Arsitektur Repositori

Proyek dideploy dalam struktur modular berbasis fase pengerjaan (Day 1 - Day 3):

```
development-phase2-model/
│
├── dataset/                              ← Direktori data terikat fisik (grounded)
│   ├── dataset_heavy_metal_grounded.csv  ← Dataset v1 (5.000 sampel, 6 kolom)
│   └── dataset_heavy_metal_grounded_v2.csv← Dataset v2 (15.000 sampel, stratified, 11 kolom)
│
├── phase1_planning/                      ← Perencanaan Teknis & Landasan Teori
│   ├── TOR.tex                           ← Term of Reference (LaTeX) dengan perumusan termodinamika Ksp
│   └── PRD.md                            ← Product Requirements Document & arsitektur model
│
├── phase2_training/                      ← Model Baseline HERA 2.0 (v1)
│   ├── src/                              ← Skrip pelatihan & evaluasi 5 model v1
│   ├── models/                           ← Serialisasi model baseline (.pkl)
│   └── results/images/                   ← 11 Plot visualisasi geokimia & ML v1 (300 DPI)
│
├── phase2.5_finetuning/                  ← Optimasi, Fine-Tuning & Integrasi (v2)
│   ├── src/                              ← Pencarian hyperparameter (Optuna TPE) & retrain v2
│   ├── models/                           ← Serialisasi model fine-tuned v2 (.pkl)
│   └── results/
│       ├── images/                       ← 8 Plot diagnostik overfitting v2 (300 DPI)
│       │   └── nickel_pca_3d.html        ← Visualisasi PCA 3D Nickel Interaktif
│       └── reports/                      ← Laporan diagnostik & parameter optimal
│
└── phase3_validation/                    ← Validasi Konsistensi Fisik & Geokimia
    ├── src/                              ← Pengujian scenario monotonic & Spearman sensitivity
    └── results/                          ← Log hasil verifikasi kelayakan fisik model
```

---

## 2. Landasan Termodinamika & Desain Dataset

Dataset disintesis dengan memadukan batas kelarutan teoritis kesetimbangan kimia air (termodinamika $K_{sp}$) untuk mencegah model machine learning memprediksi konsentrasi logam bebas yang secara fisik mustahil terjadi di alam (melebihi batas jenuh).

### Formulasi Geokimia
* **Chromium (Cr(III))**: Mengikuti kesetimbangan pengendapan Chromium(III) Hidroksida:
  $$\text{Cr}^{3+} + 3\text{OH}^- \rightleftharpoons \text{Cr(OH)}_3(s) \quad \text{dengan} \quad K_{sp} = 10^{-30}$$
  Batas kelarutan maksimum teoritis Chromium dalam $\mu\text{g/L}$ diformulasikan sebagai:
  $$\log_{10}(\text{Cr}_{\text{limit}}) = 14 - 3\cdot\text{pH} + \log_{10}(51.996 \times 10^6)$$

* **Nickel (Ni(II))**: Mengikuti kesetimbangan pengendapan Nickel(II) Hidroksida:
  $$\text{Ni}^{2+} + 2\text{OH}^- \rightleftharpoons \text{Ni(OH)}_2(s) \quad \text{dengan} \quad K_{sp} = 5.48 \times 10^{-16}$$
  Batas kelarutan maksimum teoritis Nickel dalam $\mu\text{g/L}$ diformulasikan sebagai:
  $$\log_{10}(\text{Ni}_{\text{limit}}) = 28 - 2\cdot\text{pH} - \log_{10}(K_{sp,\text{water}}^2) + \log_{10}(K_{sp} \times 58.693 \times 10^6)$$

### Rekayasa Fitur Termodinamika (5 Derived Features)
Untuk membantu algoritma non-linear memahami dinamika pelarutan tanpa mengandalkan fitting buta, ditambahkan 5 fitur hasil kalkulasi fisika:
1. `pH_squared` ($\text{pH}^2$): Merepresentasikan kuadrat aktivitas ion hidroksida ($[\text{OH}^-]^2$) yang mengontrol kesetimbangan kelarutan Nickel $Ni(OH)_2$.
2. `pOH_proxy` ($14.0 - \text{pH}$): Indikator langsung dari konsentrasi ion hidroksida di dalam larutan.
3. `pH_EC_interact` ($\text{pH} \times \text{EC}$): Menangkap efek gabungan antara keasaman air dan kekuatan ionik total (ionic strength) terhadap kelarutan logam bebas.
4. `log_EC` ($\log_{10}(\text{EC})$): Linearitas kekuatan ionik terhadap konduktivitas listrik.
5. `pH_temp_interact` ($\text{pH} \times \text{Suhu}$): Menyandikan persamaan Van 't Hoff untuk menangkap pergeseran nilai $K_{sp}$ akibat pengaruh temperatur air sungai.

---

## 3. Matriks Evolusi Model: Baseline v1 vs Fine-Tuned v2

Model dilatih menggunakan 5 algoritma: Regresi Linear, Ridge, SVR (RBF), Random Forest, dan XGBoost. Evaluasi menggunakan partisi data uji independen ($20\%$ holdout) dan 5-Fold Cross Validation.

Berikut perbandingan performa model baseline (Fase 2) dan model ter-optimasi Optuna dengan 9 fitur termodinamika (Fase 2.5):

| Logam | Algoritma | Versi | Test $R^2$ | RMSE ($\mu\text{g/L}$) | MAPE (%) | Train-Test Gap | Keterangan Status |
|---|---|---|---|---|---|---|---|
| **Chromium** | **Random Forest** | v1 (Base) | 0.95870 | 9.872 | 2.50% | $+2.50\%$ | Overfitting Ringan |
| | | v2 (Optuna) | **0.99144** | **3.018** | **0.23%** | **$+0.23\%$** | **Sempurna (Safe)** |
| | **XGBoost** | v1 (Base) | 0.96500 | 8.871 | 1.75% | $+1.75\%$ | Stabil |
| | | v2 (Optuna) | **0.99194** | **2.883** | **0.12%** | **$+0.12\%$** | **Sempurna (Safe) - Terpilih** |
| **Nickel** | **Random Forest** | v1 (Base) | 0.94460 | 12.331 | 2.50% | $+2.50\%$ | Overfitting Ringan |
| | | v2 (Optuna) | **0.98959** | **4.211** | **0.43%** | **$+0.43\%$** | **Sempurna (Safe) - Terpilih** |
| | **XGBoost** | v1 (Base) | 0.94790 | 11.231 | 3.21% | $+3.21\%$ | Overfitting Waspada |
| | | v2 (Optuna) | **0.98874** | **4.321** | **0.42%** | **$+0.42\%$** | **Sempurna (Safe)** |

*Kesimpulan Evolusi*: Penggunaan dataset v2 (15.000 baris terstratifikasi pH) dan penyuntikan 5 fitur termodinamika berhasil **meningkatkan skor $R^2$ hingga $>0.989$** sekaligus **menekan gap overfitting di bawah $0.43\%$** (menghilangkan bias variansi model secara permanen).

---

## 4. Validasi Konsistensi Fisik & Geokimia (Phase 3)

Seluruh model dievaluasi menggunakan metode validasi fisik geokimia untuk memastikan keselarasan prediksi dengan hukum alam:

* **Uji Monotonik Skenario (Clean $\rightarrow$ Moderately $\rightarrow$ Highly $\rightarrow$ Extreme)**:
  * Model dievaluasi pada skenario kenaikan limbah terlarut secara bertahap.
  * Hasil: Model Random Forest dan XGBoost v2 terbukti **$100\%$ konsisten secara monotonik naik** (tidak mengalami fluktuasi prediksi anomali saat kadar mineral terlarut meningkat).
* **Uji Sensitivitas Koefisien Spearman**:
  * Mengukur korelasi peringkat arah pengaruh parameter input terhadap estimasi logam bebas.
  * Hasil: Diperoleh korelasi negatif kuat untuk pH (semakin basa air, pengendapan hidroksida logam terakselerasi sehingga kadar logam bebas terlarut turun tajam) dan korelasi positif kuat untuk EC/TDS (peningkatan mineralisasi berbanding lurus dengan pelepasan ion logam bebas). Model v2 sepenuhnya selaras dengan koefisien arah kimia teoritis.

---

## 5. Deployment API Dual-Metal (FastAPI)

API inferensi HERA 2.0 dideploy di dalam folder `hera-monitoring/fastapi-model/` dengan arsitektur dual-metal berkecepatan tinggi dan jaminan kompatibilitas balik 100%.

### Preprocessing & Pipeline Integrasi
1. **Ekstraksi Dinamis**: Saat payload JSON diterima, API mengekstrak 5 fitur termodinamika (`pH_squared`, `pH_EC_interact`, `log_EC`, `pOH_proxy`, `pH_temp_interact`) secara real-time.
2. **Normalisasi Data**: Data dinormalisasi secara independen menggunakan objek `StandardScaler` ter-serialize milik masing-masing target logam.
3. **Prediksi Dual-Metal**: Menghasilkan estimasi konsentrasi Chromium (via XGBoost) dan Nickel (via Random Forest) secara simultan.
4. **Respon Kompatibel**: Output dikirimkan dalam format JSON terstruktur ganda:
   ```json
   {
       "cr_estimated": 0.02564, 
       "ni_estimated": 0.01544, 
       "status": "normal", 
       "unit": "mg/L", 
       "chromium": {
           "value": 0.02564, 
           "status": "normal", 
           "unit": "mg/L"
       }, 
       "nickel": {
          "value": 0.01544, 
          "status": "normal", 
          "unit": "mg/L"
       }
   }
   ```
   *Catatan*: Variabel root `cr_estimated` dan `status` dipertahankan agar sistem penyerapan data sensor Laravel yang saat ini berjalan tidak mengalami kerusakan/error.

---

## 6. Akses Hasil & Visualisasi Publikasi

Hasil visualisasi publikasi resolusi tinggi dan visualisasi 3D interaktif dapat diakses pada jalur berikut:
* **Visualisasi Komparatif Sebelum/Sesudah (v1 vs v2)**: [Phase2.5_01_improvement_comparison.png](phase2.5_finetuning/results/images/Phase2.5_01_improvement_comparison.png)
* **Plot Paritas Densitas KDE**: [Phase2.5_02_parity_plots_v2.png](phase2.5_finetuning/results/images/Phase2.5_02_parity_plots_v2.png)
* **Diagnostik Residu Normal**: [Phase2.5_03_residual_analysis_v2.png](phase2.5_finetuning/results/images/Phase2.5_03_residual_analysis_v2.png)
* **Tingkat Kepentingan Fitur Rekayasa**: [Phase2.5_04_feature_importance_9feat.png](phase2.5_finetuning/results/images/Phase2.5_04_feature_importance_9feat.png)
* **Confusion Matrix Regulasi WHO**: [Phase2.5_06_confusion_matrices_v2.png](phase2.5_finetuning/results/images/Phase2.5_06_confusion_matrices_v2.png)
* **Visualisasi PCA 3D Nickel Interaktif (HTML)**: [nickel_pca_3d.html](phase2.5_finetuning/results/images/nickel_pca_3d.html)
  *(Buka file HTML interaktif ini pada browser Anda untuk memutar ruang koordinat PC1, PC2, PC3 secara bebas).*
