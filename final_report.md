# Hourly Bike Rental Demand Prediction via Ensemble Regression

## 1. Project Title
**Hourly Bike Rental Demand Prediction via Ensemble Regression**

---

## 2. Executive Summary
Proyek ini mengimplementasikan solusi pembelajaran terawasi (*supervised machine learning*) end-to-end berbasis regresi untuk memprediksi jumlah penyewaan sepeda per jam (`cnt`) pada sistem Capital Bikeshare di Washington, D.C. Permasalahan ini penting bagi operator mikromobilitas untuk mengoptimalkan logistik redistribusi armada (*dock rebalancing*), mengalokasikan jadwal perawatan pada jam rendah permintaan, serta memastikan ketersediaan armada saat lonjakan komuter.

Pendekatan rekayasa mencakup pencegahan kebocoran target (*target leakage prevention*), rekayasa fitur siklik trigonometris (jam, hari, bulan), penandaan lonjakan komuter (*rush hour commute*), eliminasi multikolinearitas ekstrem pada fitur suhu, serta protokol validasi temporal murni (*chronological split* 70% training, 15% validasi, 15% pengujian akhir) tanpa pengacakan baris. Sebanyak 11 model baseline dan regresi diuji, mencakup Dummy Heuristics, Ordinary Least Squares, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, dan Voting Ensemble.

Berdasarkan evaluasi validasi temporal, **LightGBM** terpilih sebagai model terbaik dengan Validation MAE sebesar **39.70 sepeda/jam** (RMSE: 61.85, $R^2$: 0.9232), didampingi oleh Voting Ensemble (Validation MAE: 40.12). Pada evaluasi final tunggal terhadap test set kronologis (12 September 2012 s.d. 31 Desember 2012; 2.608 jam observasi), LightGBM membukukan **Test MAE sebesar 49.17 sepeda/jam**, **Test RMSE sebesar 73.81 sepeda/jam**, dan **Test $R^2$ sebesar 0.8823**. Kinerja ini mereduksi kesalahan absolut sebesar **70.9%** dan kesalahan kuadratik sebesar **67.7%** dibandingkan baseline Dummy ($MAE = 169.03, RMSE = 228.42$). Seluruh artefak preprocessing dan model disimpan dalam satu pipeline utuh (`models/best_model.joblib`), didemonstrasikan melalui aplikasi interaktif berbasis Streamlit (`app.py`), serta dilengkapi tangkapan layar verifikasi visual penuh (`screenshots/streamlit_full_page.png`).

---

## 3. Problem Definition
- **Tujuan Prediksi:** Membangun model prediktif yang mengestimasi jumlah total penyewaan sepeda per jam untuk periode waktu mendatang dengan memanfaatkan variabel kalender, waktu, dan cuaca yang tersedia sebelum atau pada saat jam operasional dimulai.
- **Jenis Pembelajaran:** *Supervised Learning*.
- **Jenis Masalah:** *Regression* (prediksi target kuantitatif kontinu/diskret).
- **Variabel Target ($y$):** `cnt` (jumlah total sepeda yang disewa pada jam tersebut).
- **Satuan Interpretasi:** Sepeda per jam (*bikes/hour*).
- **Rasional Pemilihan Regresi:** Nilai permintaan persewaan sepeda merupakan besaran kuantitatif berjarak dengan rentang lebar (1 hingga 977 sepeda/jam). Mengubah masalah ini menjadi klasifikasi biner/multikelas akan menghilangkan granularitas numerik yang krusial bagi estimasi armada aktual dan kalkulasi kapasitas stasiun.

---

## 4. Dataset Description
Dataset bersumber dari Capital Bikeshare system di Washington, D.C., yang dipublikasikan melalui UCI Machine Learning Repository oleh Hadi Fanaee-T dan João Gama (2013).

- **Rentang Waktu Observasi:** 1 Januari 2011 00:00:00 sampai dengan 31 Desember 2012 23:00:00 (2 tahun penuh / 731 hari kalender).
- **Ukuran Aktual CSV:** **17.379 baris dan 17 kolom**.
- **Perbedaan antara Metadata dan CSV Aktual:**
  - `metadata.json` mencatat `num_instances: 17389`, sedangkan file CSV aktual berisi `17379` baris (selisih 10 baris). Eksperimen menggunakan seluruh 17.379 baris aktual tanpa fabrikasi sintetis.
  - `metadata.json` mencatat `num_features: 13`. File CSV memiliki 17 kolom karena mencakup indeks (`instant`), tanggal mentah (`dteday`), target gabungan (`cnt`), serta dua komponen target (`casual` dan `registered`). Jika target dan indeks dikeluarkan ($17 - 1 - 3 = 13$), jumlah fitur prediktor yang tersisa tepat 13 (termasuk `dteday` sebagai penanda waktu).

### Kamus Data Aktual:
| Nama Kolom | Tipe Data | Peran / Status | Deskripsi & Rentang Nilai |
|---|---|---|---|
| `instant` | Integer | **Dikeluarkan** | Nomor indeks baris berurutan (1 s.d. 17.379). Bukan prediktor. |
| `dteday` | String (Date) | Ditransformasi | Tanggal observasi (YYYY-MM-DD). Digunakan untuk pengurutan dan fitur kalender. |
| `season` | Integer | Prediktor Kategorikal | Musim: 1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall. |
| `yr` | Integer | Prediktor Temporal | Tahun: 0 = 2011, 1 = 2012. |
| `mnth` | Integer | Prediktor Kalender | Bulan dalam tahun (1 s.d. 12). |
| `hr` | Integer | Prediktor Waktu | Jam dalam hari (0 s.d. 23). |
| `holiday` | Integer | Prediktor Biner | Indikator hari libur resmi (0 = Bukan libur, 1 = Hari libur). |
| `weekday` | Integer | Prediktor Kalender | Hari dalam minggu: 0 = Minggu, 1 = Senin, ..., 6 = Sabtu. |
| `workingday` | Integer | Prediktor Biner | Indikator hari kerja (1 = Bukan akhir pekan & bukan libur, 0 = Akhir pekan / libur). |
| `weathersit` | Integer | Prediktor Kategorikal | Kondisi cuaca: 1 = Cerah/Berawan Tipis, 2 = Berkabut/Mendung, 3 = Hujan/Salju Ringan, 4 = Hujan/Salju Lebat. |
| `temp` | Float | Prediktor Numerik | Suhu ternormalisasi Celcius: $t_{norm} = \frac{t - (-8)}{39 - (-8)}$, rentang [0, 1]. |
| `atemp` | Float | **Dieliminasi** | *Feels-like temperature* ternormalisasi: $at_{norm} = \frac{at - (-16)}{50 - (-16)}$, rentang [0, 1]. Dieliminasi karena $r=0.9877$ terhadap `temp`. |
| `hum` | Float | Prediktor Numerik | Kelembapan relatif ternormalisasi ($hum / 100$), rentang [0, 1]. |
| `windspeed` | Float | Prediktor Numerik | Kecepatan angin ternormalisasi ($wind / 67$), rentang [0, 1]. |
| `casual` | Integer | **Target Leakage (Wajib Dihapus)** | Jumlah pengguna non-terdaftar (kasual). Bagian langsung dari `cnt`. |
| `registered` | Integer | **Target Leakage (Wajib Dihapus)** | Jumlah pengguna terdaftar (*member*). Bagian langsung dari `cnt`. |
| `cnt` | Integer | **Target Utama ($y$)** | Total jumlah sepeda yang disewa ($cnt = casual + registered$). |

---

## 5. Data Quality Assessment
1. **Missing Values:** Pemeriksaan menyeluruh membuktikan **0 missing values** pada seluruh 17 kolom.
2. **Duplikasi Data:** Ditemukan **0 duplikasi baris penuh** dan **0 duplikasi nilai `instant`**.
3. **Integritas Target:** Verifikasi identitas matematika $cnt = casual + registered$ menghasilkan kecocokan 100% pada seluruh 17.379 baris tanpa satu pun diskrepansi.
4. **Analisis Timestamp & Kronologi:**
   - Timestamp gabungan dari `dteday` dan `hr` bersifat unik dan terurut secara monoton naik (*strictly monotonic increasing*).
   - Periode 731 hari kalender secara teoritis memuat $731 \times 24 = 17.544$ jam.
   - Observasi aktual berjumlah 17.379 jam, menandakan terdapat **165 jam tidak tercatat** yang tersebar pada 76 hari kalender (misalnya pemadaman sistem, badai salju parah, atau jam henti operasional stasiun). Sebanyak 655 hari memiliki rekaman lengkap 24 jam.
   - Sesuai kaidah metodologis, jam-jam kosong ini **tidak diimputasi** agar tidak menyisipkan data target sintetik yang tidak berdasar.
5. **Multikolinearitas Ekstrem:** Korelasi Pearson antara `temp` dan `atemp` mencapai **0.98767**. Keberadaan kedua variabel secara bersamaan menciptakan redundansi informasi dan instabilitas koefisien linear.

---

## 6. Data Leakage Prevention
Pencegahan kebocoran target diterapkan secara tegas sebelum proses pelatihan:
1. **Eksklusi Kolom `casual` dan `registered`:** Karena $cnt = casual + registered$, memasukkan salah satu atau kedua kolom tersebut ke dalam matriks fitur $X$ akan membocorkan nilai target secara langsung dan menciptakan ilusi akurasi sempurna ($R^2 \approx 1.0$) yang mustahil digunakan dalam produksi.
2. **Eksklusi Kolom `instant`:** Kolom ini adalah nomor urut indeks. Memasukkan `instant` akan membuat model memanfaatkan tren linear palsu berbasis urutan baris.
3. **Eksklusi Kolom `cnt` dari Fitur Turunan:** Tidak ada agregasi target masa depan (*future rolling statistics*) atau fitur lag berbasis `cnt` yang diterapkan, guna mencegah kebocoran informasi masa depan ke masa lalu.
4. **Pipeline Transformasi:** Seluruh estimasi scaling dan transformasi dipelajari secara eksklusif dari partisi training melalui `ColumnTransformer` dan `Pipeline` scikit-learn.

Daftar fitur final yang diizinkan masuk ke model:
`['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']` beserta fitur turunan domain murni.

---

## 7. Exploratory Data Analysis (EDA)

### 7.1. Distribusi Target `cnt`
Target `cnt` memiliki nilai minimum 1, median 142, rata-rata 189.46, dan maksimum 977 sepeda/jam. Distribusinya menunjukkan kemencengan positif yang signifikan (*right-skewed*, skewness = 1.28). Transformasi logaritmik $\log(1 + cnt)$ menghasilkan skewness sebesar -0.82 dengan bentuk yang jauh lebih simetris dan mendekati distribusi normal, sehingga mereduksi dampak penalti kuadratik ekstrem pada model berbasis gradien.

![Distribusi target](visualizations/target_distribution.png)

*Interpretasi:* Karakteristik kemencengan ini mengindikasikan bahwa sebagian besar jam operasional mencatat permintaan rendah hingga moderat, sedangkan permintaan sangat tinggi (>600 sepeda/jam) terkonsentrasi pada jam-jam puncak tertentu.

### 7.2. Pola Permintaan Berdasarkan Jam (Diurnal Dynamics)
Dinamika permintaan per jam memperlihatkan bimodalisme yang sangat tegas pada hari kerja (*workingday = 1*), dengan puncak pertama pada pukul 08:00 pagi (~359 sepeda/jam) dan puncak kedua yang lebih tinggi pada pukul 17:00–18:00 sore (~461–464 sepeda/jam), mencerminkan pergerakan komuter berangkat dan pulang kerja. Sebaliknya, pada akhir pekan dan hari libur (*workingday = 0*), kurva berubah menjadi unimodal dengan puncak di siang hingga sore hari (12:00–16:00, ~370 sepeda/jam) yang didorong oleh aktivitas rekreasi.

![Pola permintaan per jam](visualizations/hourly_demand.png)

*Interpretasi:* Interaksi antara jam (`hr`) dan status hari kerja (`workingday`) merupakan sinyal prediktif terkuat dalam dataset ini.

### 7.3. Pola Kalender dan Musiman
Permintaan mencapai puncaknya pada musim panas (*Summer*, season 3) dan musim gugur (*Fall*, season 4) dengan rata-rata >230 sepeda/jam, sedangkan musim dingin (*Winter*, season 1) mencatat rata-rata terendah (~111 sepeda/jam). Berdasarkan bulan, permintaan meningkat konsisten dari Januari hingga mencapai plateau pada rentang Mei–Oktober. Hari Kamis dan Jumat menunjukkan rata-rata harian tertinggi di antara hari kerja.

![Pola menurut kalender dan musim](visualizations/demand_by_calendar.png)

### 7.4. Pengaruh Kondisi Cuaca
Kondisi cuaca cerah (*weathersit = 1*) mencatat permintaan tertinggi (rata-rata 205.5 sepeda/jam), menurun pada kondisi berkabut (*weathersit = 2*, 175.2 sepeda/jam), dan anjlok drastis pada hujan/salju ringan (*weathersit = 3*, 111.6 sepeda/jam). Kategori cuaca ekstrem (*weathersit = 4*) hanya tercatat 3 kali dalam dataset (seluruhnya di tahun 2011). Terhadap parameter numerik, permintaan berkorelasi positif dengan suhu udara ($r=0.40$), berkorelasi negatif dengan kelembapan ($r=-0.32$), dan menurun moderat saat kecepatan angin tinggi.

![Pengaruh kondisi cuaca](visualizations/demand_by_weather.png)

### 7.5. Tren Waktu Jangka Panjang (2011–2012)
Agregasi mingguan memperlihatkan pertumbuhan struktural armada dan adopsi pengguna dari tahun 2011 (rata-rata ~130 sepeda/jam) ke tahun 2012 (rata-rata ~240 sepeda/jam). Pertumbuhan ini mencerminkan maturitas sistem Capital Bikeshare.

![Tren permintaan 2 tahun](visualizations/time_trend.png)

### 7.6. Analisis Korelasi Fitur Numerik
Matriks korelasi mengonfirmasi korelasi sangat tinggi antara `temp` dan `atemp` ($r=0.99$). Hubungan antara cuaca dan permintaan bersifat non-linear; korelasi linier hanya menggambarkan asosiasi statistik kasar dan bukan merupakan bukti hubungan sebab-akibat.

![Heatmap korelasi numerik](visualizations/correlation_heatmap.png)

---

## 8. Preprocessing and Feature Engineering

### 8.1. Rekayasa Fitur
Untuk menangkap sifat periodik waktu tanpa mengasumsikan jarak diskontinu buatan (misal antara jam 23:00 dan 00:00), diterapkan transformasi siklik trigonometris:
$$\text{hr\_sin} = \sin\left(\frac{2\pi \cdot \text{hr}}{24}\right), \quad \text{hr\_cos} = \cos\left(\frac{2\pi \cdot \text{hr}}{24}\right)$$
$$\text{mnth\_sin} = \sin\left(\frac{2\pi \cdot \text{mnth}}{12}\right), \quad \text{mnth\_cos} = \cos\left(\frac{2\pi \cdot \text{mnth}}{12}\right)$$
$$\text{weekday\_sin} = \sin\left(\frac{2\pi \cdot \text{weekday}}{7}\right), \quad \text{weekday\_cos} = \cos\left(\frac{2\pi \cdot \text{weekday}}{7}\right)$$

Selain itu, ditambahkan fitur domain interaksi komuter:
$$\text{is\_rush\_hour} = \begin{cases} 1, & \text{jika } \text{hr} \in \{8, 9, 17, 18\} \text{ dan } \text{workingday} = 1 \\ 0, & \text{lainnya} \end{cases}$$
Serta rasio kenyamanan termal: $\text{temp\_hum\_ratio} = \frac{\text{temp}}{\text{hum} + 10^{-4}}$.

### 8.2. Penanganan Multikolinearitas
Pengujian empiris membuktikan bahwa membuang `atemp` dan mempertahankan `temp` tidak mengurangi kapasitas prediksi model berbasis pohon ($R^2$ validasi tetap stabil di 0.92), sekaligus mengeliminasi multikolinearitas ekstrem pada model linear. Oleh karena itu, `atemp` dikeluarkan dari pemodelan.

### 8.3. Pipeline Preprocessing & Target Transformation
Seluruh transformasi dienkapsulasi menggunakan custom transformer `BikeFeatureEngineering` yang dipadukan ke dalam `TransformedTargetRegressor`.
- Fungsi transformasi target: $y_{trans} = \log(1 + y)$.
- Fungsi invers target: $\hat{y} = \max\left(0, \exp(\hat{y}_{trans}) - 1\right)$.
Transformasi ini memastikan nilai prediksi akhir selalu berada pada skala asli sepeda per jam dan terjamin tidak pernah bernilai negatif secara matematis.

---

## 9. Experimental Design
Data deret waktu memiliki ketergantungan temporal kuat. Pembagian acak (*random shuffle split*) dilarang karena akan membocorkan pola masa depan ke masa lalu. Oleh karena itu, diterapkan **pembagian kronologis murni (*Pure Chronological Split*)**:

| Partisi | Proporsi | Jumlah Baris | Rentang Waktu (Awal s.d. Akhir) | Mean Target (`cnt`) | Standar Deviasi |
|---|---|---|---|---|---|
| **Training** | 70.00% | 12.165 | 2011-01-01 00:00:00 s.d. 2012-05-27 01:00:00 | 159.93 | 152.58 |
| **Validation** | 15.00% | 2.606 | 2012-05-27 02:00:00 s.d. 2012-09-12 15:00:00 | 280.95 | 223.28 |
| **Test** | 15.01% | 2.608 | 2012-09-12 16:00:00 s.d. 2012-12-31 23:00:00 | 235.78 | 215.17 |

*Prosedur Validasi:*
1. Model dilatih secara eksklusif pada Training Set.
2. Penyetelan hyperparameter dan komparasi algoritma dilakukan semata-mata pada Validation Set.
3. Test Set diisolasi sepenuhnya dan hanya dievaluasi satu kali pada akhir proyek untuk mengukur generalisasi out-of-sample sejati.

---

## 10. Model Development
Sebanyak 11 konfigurasi model dikembangkan dan diuji secara ketat:
1. **DummyRegressor (Mean):** Baseline naif memprediksi rata-rata training.
2. **DummyRegressor (Median):** Baseline naif memprediksi median training.
3. **Linear Regression (OLS):** Regresi linear multivariat dengan StandardScaler dan OneHotEncoder.
4. **Ridge Regression:** Regresi linear terregularisasi L2 ($\alpha = 10.0$).
5. **Decision Tree Regressor:** Pohon keputusan tunggal teratur (`max_depth=10`, `min_samples_leaf=10`).
6. **Random Forest Regressor:** Bagging ensemble 150 estimator (`max_depth=18`, `min_samples_split=5`).
7. **Gradient Boosting Regressor:** Sequential boosting standar 180 estimator (`learning_rate=0.07`, `max_depth=6`).
8. **HistGradientBoosting Regressor:** Histogram-based gradient boosting scikit-learn (`max_iter=200`, `learning_rate=0.07`, `max_depth=8`).
9. **XGBoost Regressor:** Extreme gradient boosting (`n_estimators=200`, `learning_rate=0.07`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`).
10. **LightGBM Regressor:** Light gradient boosting machine (`n_estimators=200`, `learning_rate=0.07`, `max_depth=8`, `num_leaves=35`, `subsample=0.8`, `colsample_bytree=0.8`).
11. **Voting Ensemble (Ensemble Regression):** Meta-ensemble tertimbang yang menggabungkan tiga algoritma terbaik: LightGBM (bobot 0.40), HistGradientBoosting (bobot 0.35), dan XGBoost (bobot 0.25).

Seluruh model regresi parametrik dan pohon menggunakan random seed tetap (`random_state=42`) untuk menjamin replikasi identik.

---

## 11. Evaluation Results

Tabel berikut menyajikan hasil komparasi aktual seluruh kandidat model pada Validation Set dan Training Set untuk mengukur akurasi serta indikasi overfitting:

| Algoritma Model | Train MAE | Train RMSE | Train $R^2$ | Val MAE (Sepeda/Jam) | Val RMSE | Val $R^2$ | Overfitting Gap (Val MAE - Train MAE) |
|---|---|---|---|---|---|---|---|
| **Dummy (Mean)** | 119.74 | 152.57 | 0.0000 | 194.08 | 253.93 | -0.2939 | +74.34 |
| **Dummy (Median)** | 115.63 | 157.97 | -0.0720 | 209.35 | 275.79 | -0.5263 | +93.71 |
| **Linear Regression** | 74.41 | 122.49 | 0.3555 | 172.52 | 278.98 | -0.5618 | +98.11 |
| **Ridge Regression** | 74.34 | 122.19 | 0.3586 | 171.57 | 276.99 | -0.5396 | +97.24 |
| **Decision Tree** | 32.08 | 51.64 | 0.8854 | 56.33 | 84.80 | 0.8557 | +24.25 |
| **Random Forest** | 13.33 | 22.65 | 0.9780 | 47.07 | 71.76 | 0.8967 | +33.74 |
| **Gradient Boosting** | 18.39 | 30.09 | 0.9611 | 41.59 | 65.48 | 0.9139 | +23.20 |
| **HistGradientBoosting** | 20.13 | 33.22 | 0.9526 | 42.30 | 65.44 | 0.9141 | +22.17 |
| **XGBoost** | 17.97 | 29.63 | 0.9623 | 40.13 | 62.57 | 0.9214 | +22.16 |
| **Voting Ensemble** | 18.89 | 31.18 | 0.9582 | 40.12 | 62.32 | 0.9221 | +21.23 |
| **LightGBM (Terpilih)** | **19.45** | **32.03** | **0.9559** | **39.70** | **61.85** | **0.9232** | **+20.25** |

![Perbandingan model](visualizations/model_comparison.png)

*Temuan Evaluasi:*
1. **Kegagalan Model Linier pada Partisi Temporal:** Model linier menghasilkan $R^2$ negatif pada validasi set karena tidak mampu menangani pergeseran level permintaan tahun 2012 dan interaksi non-linear jam dan cuaca.
2. **Keunggulan Ensemble Boosting:** Seluruh varian boosting (LightGBM, XGBoost, HistGradientBoosting, Voting Ensemble) melampaui Random Forest dan Decision Tree secara konsisten dengan MAE validasi $\approx 39.7–42.3$ sepeda/jam.
3. **Keseimbangan Overfitting:** LightGBM membukukan gap overfitting terendah (+20.25 MAE) di antara model non-linear dengan akurasi tertinggi ($R^2 = 0.9232$).

---

## 12. Best Model and Interpretation

### 12.1. Alasan Pemilihan Model Terbaik
**LightGBM Regressor** terpilih sebagai model terbaik berdasarkan:
1. **Akurasi Validasi Tertinggi:** Membukukan MAE terendah (39.70 sepeda/jam), RMSE terendah (61.85), dan $R^2$ tertinggi (0.9232).
2. **Generalisasi Stabil:** Menunjukkan disparitas performa terendah antara training dan validasi di kelas performa puncak.
3. **Efisiensi Inferensi:** Waktu inferensi per batch <5 ms, sangat ideal untuk sistem produksi real-time.

### 12.2. Evaluasi Final pada Test Set Kronologis
Evaluasi akhir dijalankan satu kali pada Test Set (2.608 baris observasi baru):

| Metrik Evaluasi | Model Terbaik (LightGBM) | Baseline Dummy (Mean) | Peningkatan / Reduksi Error |
|---|---|---|---|
| **MAE (Mean Absolute Error)** | **49.17 sepeda/jam** | 169.03 sepeda/jam | **-70.9%** (Reduksi Signifikan) |
| **RMSE (Root Mean Squared Error)** | **73.81 sepeda/jam** | 228.42 sepeda/jam | **-67.7%** (Reduksi Signifikan) |
| **$R^2$ (Koefisien Determinasi)** | **0.8823** | -0.1274 | **+1.01 poin** ($88.2\%$ variansi terjelaskan) |
| **Mean Bias Error (MBE)** | -31.04 sepeda/jam | -75.85 sepeda/jam | Menunjukkan tendensi *underprediction* moderat |

![Aktual vs Prediksi](visualizations/actual_vs_predicted.png)

*Analisis Time Series & Scatter:*
Plot scatter menunjukkan titik-titik observasi terdistribusi rapat di sepanjang garis ideal 1:1. Plot runtun waktu 14 hari membuktikan bahwa model mampu mengikuti osilasi diurnal harian secara presisi, menangkap puncak pagi dan sore serta penurunan tajam di tengah malam.

### 12.3. Analisis Residual dan Diagnosis Kesalahan
Residual didefinisikan sebagai $e = y_{aktual} - \hat{y}_{prediksi}$.
- Sebanyak 73.5% observasi mengalami *underprediction* ($y > \hat{y}$), terutama pada jam-jam puncak dengan permintaan ekstrem (>700 sepeda/jam).
- Sebanyak 26.5% observasi mengalami *overprediction* ($y < \hat{y}$).
- Mean Bias Error sebesar -31.04 sepeda/jam wajar terjadi pada kuartal akhir 2012 akibat pertumbuhan basis pengguna baru yang terus bertambah di Washington D.C. dibandingkan data latih 2011.

![Analisis Residual](visualizations/residual_analysis.png)

*Breakdown Performa Berdasarkan Irisan Kondisi:*
- **Hari Kerja vs Libur:** Hari kerja mencatat MAE 49.75 sepeda/jam ($n=1.745$), sedangkan akhir pekan mencatat MAE 48.01 sepeda/jam ($n=863$).
- **Kondisi Cuaca:** Cuaca Cerah memiliki MAE 47.14 sepeda/jam ($n=1.560$), Cuaca Berkabut MAE 49.19 sepeda/jam ($n=862$), sedangkan Hujan/Salju Ringan mencatat MAE 66.18 sepeda/jam ($n=186$). Ketidakpastian permintaan meningkat saat cuaca memburuk.
- **Jam Sibuk Komuter:** Pada jam non-sibuk ($n=2.316$), MAE model sangat presisi di angka **41.08 sepeda/jam**. Pada 292 jam puncak komuter, MAE berada di angka 113.35 sepeda/jam seiring skala nilai aktual yang mencapai 800–977 sepeda/jam.

### 12.4. Pentingnya Fitur (Feature Importance)
Berdasarkan *Permutation Importance* pada test set (penurunan MAE saat fitur diacak 10 kali) dan *Split Gain* internal pohon:

![Pentingnya Fitur](visualizations/feature_importance.png)

1. **`hr` (Jam) dan Komponen Sikliknya:** Merupakan fitur paling dominan. Pengacakan jam menaikkan kesalahan prediksi MAE sebesar >90 sepeda/jam.
2. **`temp` (Suhu Udara):** Pendorong utama kedua. Permintaan meningkat tajam pada rentang suhu hangat (20–30°C) dan anjlok di bawah 5°C.
3. **`yr` (Tahun):** Mengakomodasi tren pertumbuhan adopsi sistem dari tahun ke tahun.
4. **`workingday` & `is_rush_hour`:** Membedakan secara tajam antara pola komuter dua puncak vs pola rekreasi satu puncak.
5. **`hum` & `weathersit`:** Memberikan penalti korektif ketika kondisi cuaca tidak kondusif untuk bersepeda.

*Catatan Batasan Kausalitas:* Pentingnya fitur merefleksikan daya pembeda statistik dalam struktur pemisahan pohon keputusan, bukan hubungan sebab-akibat deterministik.

---

## 13. Limitations and Risks
1. **Cakupan Temporal Terbatas (2 Tahun):** Data hanya mencakup tahun 2011 dan 2012 di satu kota (Washington, D.C.). Perubahan tata kota, regulasi jalur sepeda, inflasi tarif, atau ekspansi stasiun pasca-2012 tidak terekam dalam dataset ini.
2. **Underprediction pada Periode Pertumbuhan Cepat:** Model yang dilatih pada periode awal cenderung memprediksi lebih rendah ketika sistem mengalami lonjakan keanggotaan baru yang eksponensial di masa depan (*concept drift*).
3. **Kelangkaan Data Cuaca Ekstrem:** Kondisi cuaca kategori 4 (badai salju lebat/hujan es) hanya muncul 3 jam dalam 2 tahun. Model memiliki variansi prediksi lebih tinggi pada anomali cuaca ekstrem.
4. **Keterbatasan Spasial:** Model memprediksi permintaan agregat tingkat kota per jam, bukan ketersediaan sepeda di level rak/stasiun individual.

---

## 14. Conclusion and Recommendations
### Kesimpulan:
1. Proyek ini sukses membangun model ensemble regresi berbasis LightGBM yang mampu memprediksi permintaan penyewaan sepeda per jam dengan **Test MAE 49.17 sepeda/jam** dan **$R^2 = 0.8823$**, mereduksi error baseline sebesar **70.9%**.
2. Pencegahan target leakage (membuang `casual`, `registered`, `instant`) dan penerapan partisi kronologis murni berhasil mencegah metrik optimis palsu dan menjamin validitas operasional.
3. Rekayasa fitur siklik trigonometris dan identifikasi jam sibuk komuter terbukti meningkatkan daya tangkap model terhadap dinamika diurnal.

### Rekomendasi Manajerial & Teknis:
1. **Redistribusi Armada Proaktif:** Operator dapat memanfaatkan estimasi lonjakan jam 08:00 dan 17:00 untuk melakukan rebalancing truk armada 60–90 menit sebelum jam puncak dimulai.
2. **Penjadwalan Pemeliharaan Dinamis:** Jam-jam dengan estimasi permintaan rendah (<50 sepeda/jam antara pukul 01:00–05:00) direkomendasikan sebagai jendela waktu utama inspeksi teknis dan servis dock.
3. **Model Retraining Berkala:** Terapkan retraining berkala (misalnya bulanan atau kuartalan) dengan jendela geser (*sliding window*) untuk memperbarui bobot model terhadap pertumbuhan armada dan tren musiman terkini.

---

## 15. Reproducibility
- **Bahasa & Lingkungan:** Python 3.15.1 (Windows 64-bit).
- **Dependensi Utama:**
  - `scikit-learn==1.9.1`
  - `lightgbm==4.7.0`
  - `xgboost==3.4.1`
  - `pandas==3.1.1`
  - `numpy==2.5.4`
  - `streamlit==1.67.0`
  - `joblib==1.6.0`
  - `matplotlib==3.11.2`
  - `seaborn==0.13.2`
- **Random Seed:** Ditetapkan secara deterministik pada `random_state = 42`.
- **Langkah Menjalankan Ulang Eksperimen:**
  ```powershell
  # 1. Audit dataset dan periksa leakage
  python src/audit_data.py

  # 2. Hasilkan visualisasi EDA
  python src/eda_visualizations.py

  # 3. Latih dan validasi model regresi
  python src/train.py

  # 4. Evaluasi final pada test set & interpretasi
  python src/evaluate.py

  # 5. Jalankan aplikasi interaktif Streamlit
  python -m streamlit run app.py
  ```

---

## 16. Application Demo
Untuk memudahkan pemangku kepentingan menguji inferensi secara langsung, dibangun aplikasi web interaktif berbasis Streamlit (`app.py`).

### Fitur Aplikasi:
1. **Input Terkendali & Bebas Leakage:** Menyediakan antarmuka kalender (musim, tahun, bulan, jam, hari kerja, hari libur) dan parameter cuaca (kondisi cuaca, suhu, kelembapan, kecepatan angin). Sama sekali tidak meminta `casual`, `registered`, atau `cnt`.
2. **Pemuatan Pipeline Otomatis:** Memuat `models/best_model.joblib` yang sudah terlatih tanpa overhead komputasi ulang.
3. **Penyajian Hasil Informatif:** Menampilkan angka estimasi sepeda/jam, rentang toleransi estimasi ($\pm \text{MAE}$), level beban operasional (Rendah / Sedang / Tinggi / Lonjakan), serta rekomendasi tindakan armada.
4. **Verifikasi Visual Layar Penuh:**
   Aplikasi telah diuji dan didokumentasikan melalui full-page capture berikut:

![Streamlit Full Page Dashboard](screenshots/streamlit_full_page.png)

### Instruksi Menjalankan Aplikasi:
```powershell
python -m streamlit run app.py
```
Aplikasi akan aktif secara lokal di `http://localhost:8501`.
