# Bab 3: Data — Bahan Bakar Machine Learning

---

## Analogi Pembuka: Masakan dan Data

Bayangkan kamu seorang koki yang akan memasak hidangan mewah. Kamu punya resep terbaik di dunia, peralatan dapur termahal, dan teknik memasak yang sempurna. Tapi...

Bahan-bahannya basi. Dagingnya berbau. Sayurnya layu. Bumbunya kadaluarsa.

Apa jadinya? Hidangan yang jelek. Bukan karena resepnya salah, bukan karena kokinya tidak pintar — tapi karena **bahannya buruk**.

Machine Learning bekerja dengan prinsip yang sama. Istilahnya:

> **"Garbage In, Garbage Out" (GIGO)**

Model ML yang paling canggih sekalipun tidak bisa menghasilkan prediksi yang baik kalau datanya buruk. Data adalah bahan bakar. Data adalah fondasi. Data adalah **segalanya**.

Di bab ini, kita akan mendalami setiap aspek data — mulai dari jenis data sampai cara menyiapkannya agar model bisa belajar dengan optimal.

---

## 3.1 Data di Dunia Nyata: Seberapa Penting?

### Fakta Mengejutkan

Di proyek ML dunia nyata, pembagian waktunya kira-kira seperti ini:

```
┌────────────────────────────────────────────────┐
│                                                │
│  5-10%   ██████████████████████████  Memilih   │
│          algoritma & training                   │
│                                                │
│  10-15% ████████████████████████████████        │
│          Evaluasi & deployment                 │
│                                                │
│  75-85% ████████████████████████████████████    │
│          ████████████████████████████████       │
│          Mengumpulkan, membersihkan, dan       │
│          menyiapkan data                       │
│                                                │
└────────────────────────────────────────────────┘
```

Kamu mungkin mengira bahwa ML adalah tentang algoritma yang fancy. Kenyataannya, **data preparation** menghabiskan sebagian besar waktu.

### Mengapa Data Begitu Penting?

| Aspek | Penjelasan |
|-------|-----------|
| **Quantity** | Model perlu cukup data untuk menemukan pola |
| **Quality** | Data kotor = model kotor |
| **Diversity** | Data harus mewakili semua skenario yang mungkin |
| **Balance** | Data yang tidak seimbang membuat model bias |
| **Relevance** | Fitur yang tidak relevan menambah noise |

---

## 3.2 Jenis-Jenis Data

### Berdasarkan Struktur

```
DATA
 │
 ├── TERSTRUKTUR (Structured)
 │   ├── Tabel (CSV, SQL)
 │   ├── Spreadsheet
 │   └── Database relasional
 │
 ├── SEMI-TERSTRUKTUR (Semi-structured)
 │   ├── JSON
 │   ├── XML
 │   ├── Log files
 │   └── HTML
 │
 └── TIDAK TERSTRUKTUR (Unstructured)
     ├── Teks (email, review, artikel)
     ├── Gambar (foto, scan)
     ├── Audio (suara, musik)
     └── Video
```

### Berdasarkan Tipe

| Tipe | Penjelasan | Contoh | Pengecekan |
|------|-----------|--------|-----------|
| **Numerik Kontinu** | Angka dengan desimal, bisa diukur | Suhu: 36.5°C, Harga: Rp 150.000 | Bisa dijumlah, ada desimal |
| **Numerik Diskrit** | Angka bulat, bisa dihitung | Jumlah anak: 3, Jumlah kamar: 2 | Bulat saja |
| **Kategorikal Nominal** | Kategori tanpa urutan | Warna: merah/biru/hijau | Tidak bisa di-sort |
| **Kategorikal Ordinal** | Kategori dengan urutan | Ukuran: S < M < L < XL | Bisa di-sort, tapi jarak tidak sama |
| **Biner** | Hanya 2 pilihan | Ya/Tidak, 0/1, Sehat/Sakit | Hanya 2 nilai |
| **Datetime** | Waktu dan tanggal | 2024-01-15 14:30:00 | Format waktu |
| **Teks** | String karakter | "Produk ini bagus sekali!" | Kalimat/paragraf |

### Representasi Data untuk ML

ML hanya mengerti **angka**. Jadi kita harus mengubah semua data menjadi angka:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

df = pd.DataFrame({
    'nama':      ['Budi', 'Ani', 'Cici', 'Dedi'],
    'umur':      [25, 30, 28, 35],
    'gaji_juta': [8.5, 12.0, 9.5, 15.0],
    'kota':      ['Jakarta', 'Bandung', 'Jakarta', 'Surabaya'],
    'ukuran':    ['M', 'S', 'L', 'M'],
    'lulus':     ['Ya', 'Tidak', 'Ya', 'Ya']
})

print("Data Asli:")
print(df)
```

```
   nama  umur  gaji_juta     kota ukuran lulus
0  Budi    25       8.5  Jakarta      M    Ja
1  Ani     30      12.0  Bandung      S  Tidak
2  Cici    28       9.5  Jakarta      L    Ja
3  Dedi    35      15.0 Surabaya      M    Ja
```

Sekarang kita ubah setiap tipe:

#### Numerik Kontinu — Sudah angka, mungkin perlu scaling

```python
# Umur dan gaji sudah numerik
# Tapi scale-nya berbeda: umur 25-35, gaji 8-15
# Kita bahas scaling nanti (Section 3.6)
```

#### Kategorikal Nominal — One-Hot Encoding

```python
# "kota" adalah nominal: Jakarta, Bandung, Surabaya
# Tidak ada urutan di antaranya

# Cara 1: pandas get_dummies
kota_encoded = pd.get_dummies(df['kota'], prefix='kota')
print(kota_encoded)
```

```
   kota_Bandung  kota_Jakarta  kota_Surabaya
0         False          True          False
1          True         False          False
2         False          True          False
3         False         False           True
```

Kenapa tidak cukup Label Encoding (Jakarta=0, Bandung=1, Surabaya=2)?

```
JAKARTA    = 0
BANDUNG    = 1
SURABAYA   = 2

Model bisa mengira: Surabaya (2) > Bandung (1) > Jakarta (0)
Padahal tidak ada urutan antar kota!
One-Hot Encoding menghilangkan hubungan ordinal palsu ini.
```

#### Kategorikal Ordinal — Ordinal Encoding

```python
# "ukuran" adalah ordinal: S < M < L
# Ada urutan yang jelas

size_mapping = {'S': 1, 'M': 2, 'L': 3}
df['ukuran_enc'] = df['ukuran'].map(size_mapping)
print(df[['ukuran', 'ukuran_enc']])
```

```
  ukuran  ukuran_enc
0      M           2
1      S           1
2      L           3
3      M           2
```

Sekarang model mengerti bahwa L(3) > M(2) > S(1).

#### Biner — Label Encoding

```python
# "lulus" hanya punya 2 nilai: Ya/Tidak

df['lulus_enc'] = df['lulus'].map({'Ya': 1, 'Tidak': 0})
print(df[['lulus', 'lulus_enc']])
```

```
  lulus  lulus_enc
0    Ja          1
1  Tidak         0
2    Ja          1
3    Ja          1
```

#### Ringkasan Encoding

| Tipe Data | Metode Encoding | Kenapa? |
|-----------|----------------|---------|
| Nominal (sedikit kategori) | One-Hot Encoding | Tidak ada urutan |
| Nominal (banyak kategori) | Target Encoding / Embedding | One-Hot terlalu banyak kolom |
| Ordinal | Ordinal Encoding | Ada urutan yang bermakna |
| Biner | Label Encoding (0/1) | Hanya 2 nilai |
| Teks | TF-IDF / Word Embedding | Konversi kata menjadi vektor |

---

## 3.3 Data Quality: Membersihkan Data yang Kotor

Data di dunia nyata **selalu kotor**. Selalu. Tidak ada data yang sempurna dari sumbernya.

### Masalah-Masalah Data Kotor

#### 1. Missing Values (Data Hilang)

```python
df_dirty = pd.DataFrame({
    'umur':  [25, np.nan, 28, 35, 30],
    'gaji':  [8.5, 12.0, np.nan, 15.0, np.nan],
    'kota':  ['Jakarta', 'Bandung', np.nan, 'Surabaya', 'Jakarta']
})

print("Data dengan missing values:")
print(df_dirty)
print(f"\nJumlah missing per kolom:\n{df_dirty.isnull().sum()}")
```

```
    umur  gaji      kota
0   25.0   8.5   Jakarta
1    NaN  12.0   Bandung
2   28.0   NaN       NaN
3   35.0  15.0  Surabaya
4   30.0   NaN   Jakarta

Jumlah missing per kolom:
umur    1
gaji    2
kota    1
```

**Cara menangani missing values:**

| Metode | Cara | Kapan Cocok? | Risiko |
|--------|------|-------------|--------|
| **Hapus baris** | `df.dropna()` | Data sedikit hilang (<5%) | Kehilangan informasi |
| **Hapus kolom** | `df.drop(col, axis=1)` | Kolom >50% hilang | Kehilangan fitur |
| **Isi dengan mean** | `df.fillna(df.mean())` | Data numerik, distribusi normal | Mengurangi variance |
| **Isi dengan median** | `df.fillna(df.median())` | Data numerik, ada outlier | Lebih robust dari mean |
| **Isi dengan mode** | `df.fillna(df.mode()[0])` | Data kategorikal | Bisa bias |
| **Isi dengan nilai konstan** | `df.fillna(0)` atau `df.fillna('Unknown')` | Kasus khusus | Bisa bias |
| **Interpolasi** | `df.interpolate()` | Data time series | Asumsi trend |
| **Prediksi (ML)** | Latih model untuk prediksi nilai hilang | Data kompleks, banyak fitur | Kompleks, bisa overfit |

```python
# Metode 1: Hapus baris dengan missing
df_clean_drop = df_dirty.dropna()
print("Setelah dropna():", len(df_clean_drop), "baris")  # Mungkin sedikit!

# Metode 2: Isi numerik dengan median
df_dirty['umur'] = df_dirty['umur'].fillna(df_dirty['umur'].median())
df_dirty['gaji'] = df_dirty['gaji'].fillna(df_dirty['gaji'].median())

# Metode 3: Isi kategorikal dengan mode
df_dirty['kota'] = df_dirty['kota'].fillna(df_dirty['kota'].mode()[0])

print("Setelah imputasi:")
print(df_dirty)
```

#### 2. Outliers (Nilai yang Jauh dari Lainnya)

```python
df_outlier = pd.DataFrame({
    'gaji_juta': [8, 9, 10, 8.5, 7.5, 500, 9.5, 11, 8, 10.5]
    # 500 juta adalah outlier! Mungkin CEO, mungkin kesalahan input
})
```

**Cara mendeteksi outlier:**

```python
# Metode 1: IQR (Interquartile Range)
Q1 = df_outlier['gaji_juta'].quantile(0.25)
Q3 = df_outlier['gaji_juta'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
print(f"Bounds: [{lower_bound}, {upper_bound}]")

outliers = df_outlier[(df_outlier['gaji_juta'] < lower_bound) |
                       (df_outlier['gaji_juta'] > upper_bound)]
print(f"Outliers: {outliers.values}")
```

```
Q1=8.25, Q3=10.375, IQR=2.125
Bounds: [5.0625, 13.5625]
Outliers: [[500]]  ← Terdeteksi!
```

**Metode 2: Z-Score**

```python
from scipy import stats

z_scores = stats.zscore(df_outlier['gaji_juta'])
outliers_z = df_outlier[np.abs(z_scores) > 3]
print(f"Outliers (Z-score): {outliers_z.values}")
```

**Cara menangani outlier:**

| Metode | Penjelasan |
|--------|-----------|
| Hapus | Kalau jelas error (gaji 5 miliar) |
| Cap/Winsorize | Ganti dengan batas atas/bawah (500 → 13.5) |
| Transformasi | Log transform meredam outlier |
| Biarkan | Kalau outlier itu valid (CEO memang gajinya tinggi) |

```python
# Capping (Winsorization)
upper_cap = df_outlier['gaji_juta'].quantile(0.95)
df_outlier['gaji_capped'] = df_outlier['gaji_juta'].clip(upper=upper_cap)
```

#### 3. Data Duplikat

```python
df_dup = pd.DataFrame({
    'nama': ['Budi', 'Budi', 'Ani', 'Cici', 'Ani'],
    'umur': [25, 25, 30, 28, 30],
    'gaji': [8.5, 8.5, 12.0, 9.5, 12.0]
})

print(f"Jumlah duplikat: {df_dup.duplicated().sum()}")
df_clean = df_dup.drop_duplicates()
print(f"Setelah hapus duplikat: {len(df_clean)} baris (dari {len(df_dup)})")
```

#### 4. Tipe Data yang Salah

```python
df_wrong = pd.DataFrame({
    'gaji_text': ['8.5', '12.0', '9.5', 'lima belas'],
    'tanggal': ['2024-01-15', '2024/02/20', '15-03-2024', 'Jan 4, 2024']
})

# Konversi gaji_text ke numerik (error = NaN)
df_wrong['gaji'] = pd.to_numeric(df_wrong['gaji_text'], errors='coerce')
print(df_wrong)
# 'lima belas' jadi NaN — harus di-handle!

# Konversi tanggal ke format seragam
df_wrong['tanggal_clean'] = pd.to_datetime(df_wrong['tanggal'], format='mixed')
```

### Checklist Data Cleaning

```
□ Cek missing values (df.isnull().sum())
□ Cek tipe data (df.dtypes)
□ Cek duplikat (df.duplicated().sum())
□ Cek outlier (boxplot, IQR, z-score)
□ Cek inkonsistensi (Jakarta vs jakarta vs JKTR)
□ Cek range nilai (umur -5? gaji negatif?)
□ Cek distribusi (apakah masuk akal?)
□ Cek korelasi antar fitur
```

---

## 3.4 Feature Engineering — Seni Membuat Fitur Baru

> "Feature engineering adalah seni mengubah data mentah menjadi fitur yang membuat model ML lebih baik."

Andrew Ng pernah berkata:

> *"Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering.'"*

### Apa Itu Feature Engineering?

Bayangkan kamu mau memprediksi harga rumah. Data mentah yang kamu punya:

```
tanggal_dibangun = "1990-05-15"
luas_tanah = 120  # m²
lokasi = "Jakarta Selatan"
jumlah_kamar = 3
```

Fitur mentah ini sudah cukup. Tapi kamu bisa **membuat fitur baru** yang lebih bermakna:

```python
import pandas as pd
from datetime import datetime

df = pd.DataFrame({
    'tanggal_dibangun': ['1990-05-15', '2005-10-01', '2020-03-20'],
    'luas_tanah': [120, 80, 150],
    'lokasi': ['Jakarta Selatan', 'Bandung', 'Jakarta Pusat'],
    'jumlah_kamar': [3, 2, 4]
})

# Feature Engineering!

# 1. Fitur dari tanggal: umur rumah (lebih bermakna dari tanggal mentah)
df['tanggal_dibangun'] = pd.to_datetime(df['tanggal_dibangun'])
df['umur_rumah'] = (datetime.now() - df['tanggal_dibangun']).dt.days / 365

# 2. Fitur rasio: luas per kamar (berapa besar setiap kamar?)
df['luas_per_kamar'] = df['luas_tanah'] / df['jumlah_kamar']

# 3. Fitur biner: rumah tua? (> 20 tahun)
df['rumah_tua'] = (df['umur_rumah'] > 20).astype(int)

# 4. Fitur kategorikal ordinal: kategori lokasi
lokasi_premium = {'Jakarta Pusat': 3, 'Jakarta Selatan': 2, 'Bandung': 1}
df['lokasi_tier'] = df['lokasi'].map(lokasi_premium)

# 5. Fitur interaksi: luas × lokasi_tier
df['luas_tier_interaction'] = df['luas_tanah'] * df['lokasi_tier']

print(df[['umur_rumah', 'luas_per_kamar', 'rumah_tua', 'lokasi_tier', 'luas_tier_interaction']])
```

### Teknik Feature Engineering

| Teknik | Penjelasan | Contoh |
|--------|-----------|--------|
| **Binning** | Kelompokkan nilai kontinu jadi kategori | Umur: 0-18=anak, 19-35=dewasa muda |
| **Log Transform** | Terapkan log pada fitur miring | log(gaji) → distribusi lebih normal |
| **Rasio** | Bagi dua fitur | luas_tanah / jumlah_kamar |
| **Interaksi** | Kalikan dua fitur | luas × lokasi_premium |
| **Ekstraksi tanggal** | Tarik info dari datetime | Tahun, bulan, hari, weekend? |
| **Target Encoding** | Ganti kategori dengan rata-rata target | Kode pos → rata-rata harga rumah di pos tersebut |
| **Polynomial** | Buat fitur pangkat | x → x, x², x³, x×y |

```python
from sklearn.preprocessing import PolynomialFeatures

# Polynomial Features
X = np.array([[1, 2], [3, 4], [5, 6]])
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print("Original:", X.shape)   # (3, 2)
print("Polynomial:", X_poly.shape)  # (3, 6)
# [1, x1, x2, x1², x1×x2, x2²]
```

### Feature Selection — Memilih Fitur yang Penting

Tidak semua fitur bermanfaat. Beberapa bahkan merusak model (noise). Feature selection membantu memilih fitur yang benar-benar berguna.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Metode 1: Filter Method (berbasis statistik)
selector = SelectKBest(f_classif, k=5)  # Pilih 5 fitur terbaik
X_selected = selector.fit_transform(X, y)
print("Fitur terpilih:", selector.get_support())

# Metode 2: Feature Importance dari Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"{name:20s}: {imp:.4f}")
```

**Mengapa Feature Selection Penting?**

1. **Curse of Dimensionality**: Semakin banyak fitur → semakin banyak data yang dibutuhkan
2. **Noise**: Fitur tidak relevan menambah noise
3. **Overfitting**: Terlalu banyak fitur → model menghafal noise
4. **Waktu training**: Fitur berlebihan → training lambat

---

## 3.5 Data Splitting — Train, Validation, Test

### Analogi Ujian

Kamu seorang murid yang akan menghadapi ujian matematika.

1. **Training Set** = Buku latihan. Kamu belajar dari ini. Kamu bisa menghafal jawabannya.
2. **Validation Set** = Latihan ujian simulasi. Kamu coba, lihat skor, lalu sesuaikan cara belajar.
3. **Test Set** = Ujian sesungguhnya. Kamu hanya mengerjakan ini sekali. Ini menentukan nilai akhirmu.

```
SELURUH DATA (100%)
├── Training Set (60-80%)  → Model belajar dari sini
├── Validation Set (10-20%) → Menyetel hyperparameter
└── Test Set (10-20%)       → Evaluasi akhir, SEKALI SAJA
```

### Kenapa Tiga Split, Bukan Dua?

```
SALAH:
  Training (80%) → Model belajar
  Test (20%) → Evaluasi
  ❌ Tapi hyperparameter di-tune berdasarkan test score
  ❌ Model secara tidak langsung "melihat" test data
  ❌ Hasil test tidak bisa dipercaya

BENAR:
  Training (60%) → Model belajar
  Validation (20%) → Tuning hyperparameter
  Test (20%) → Evaluasi final, SEKALI SAJA, JANGAN DISENTUH LAGI
  ✓ Model belum pernah "melihat" test data
  ✓ Evaluasi jujur dan dapat dipercaya
```

### Kode: Data Splitting

```python
from sklearn.model_selection import train_test_split

# Load data
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Split 1: Training + Validation vs Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split 2: Training vs Validation (dari data temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
# 0.25 × 0.8 = 0.2 → Jadi: 60% train, 20% val, 20% test

print(f"Training:   {len(X_train)} ({len(X_train)/len(X):.0%})")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X):.0%})")
print(f"Test:       {len(X_test)} ({len(X_test)/len(X):.0%})")
```

### Cross-Validation — Lebih Baik dari Split Tunggal

Kalau dataset kecil, satu split mungkin tidak representatif. Cross-validation membagi data jadi K bagian:

```
5-Fold Cross-Validation:

Fold 1: [TEST   ][TRAIN  ][TRAIN  ][TRAIN  ][TRAIN  ] → Score: 0.92
Fold 2: [TRAIN  ][TEST   ][TRAIN  ][TRAIN  ][TRAIN  ] → Score: 0.88
Fold 3: [TRAIN  ][TRAIN  ][TEST   ][TRAIN  ][TRAIN  ] → Score: 0.91
Fold 4: [TRAIN  ][TRAIN  ][TRAIN  ][TEST   ][TRAIN  ] → Score: 0.87
Fold 5: [TRAIN  ][TRAIN  ][TRAIN  ][TRAIN  ][TEST   ] → Score: 0.90

Rata-rata Score: 0.896 ± 0.022
```

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {scores}")
print(f"Rata-rata: {scores.mean():.4f}")
print(f"Std Deviasi: {scores.std():.4f}")
```

### Stratified Split — Jaga Proporsi Kelas

```python
# TANPA stratify — bisa tidak seimbang
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DENGAN stratify — proporsi kelas dijaga sama
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Penting untuk dataset imbalance!
# Misal kelas A=90%, kelas B=10%
# Tanpa stratify → test mungkin 100% kelas A, 0% kelas B
# Dengan stratify → test juga 90%A, 10%B
```

---

## 3.6 Feature Scaling — Menyamakan Skala

### Mengapa Scaling Penting?

Algoritma berbasis **jarak** (K-NN, SVM, Neural Network) sangat sensitif terhadap skala fitur.

```python
# Contoh masalah skala
data = pd.DataFrame({
    'gaji_rp':    [8000000, 12000000, 15000000, 9000000],  # Range: jutaan
    'umur':       [25, 30, 35, 28],                         # Range: puluhan
})

# Jarak Euclidean antara orang 1 dan 2:
# d = √((8000000-12000000)² + (25-30)²)
#   = √(16000000000000 + 25)
#   ≈ 4000000
#
# Umur praktis TIDAK BERPENGARUH karena gaji jauh lebih besar!
```

### Metode Scaling

| Metode | Rumus | Range Output | Kapan Pakai? |
|--------|-------|-------------|-------------|
| **Min-Max** | (x - min) / (max - min) | [0, 1] | Neural Network, ketika distribusi tidak normal |
| **Standardization** | (x - mean) / std | ~[-3, 3] | SVM, K-NN, Logistic Regression, PCA |
| **Robust Scaler** | (x - median) / IQR | Varies | Data ada outlier |
| **Log Transform** | log(x) | Varies | Data sangat miring (skewed) |

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np

data = np.array([[25, 8000000],
                  [30, 12000000],
                  [35, 15000000],
                  [28, 9000000]])

# Min-Max Scaling → [0, 1]
scaler_mm = MinMaxScaler()
data_mm = scaler_mm.fit_transform(data)
print("Min-Max Scaled:\n", data_mm)
# [[0.  , 0.  ],
#  [0.5 , 0.57],
#  [1.  , 1.  ],
#  [0.3 , 0.14]]

# Standardization → mean=0, std=1
scaler_std = StandardScaler()
data_std = scaler_std.fit_transform(data)
print("Standardized:\n", data_std)
# Sekarang gaji dan umur punya kontribusi sama terhadap jarak!

# Robust Scaling → tahan outlier
scaler_rob = RobustScaler()
data_rob = scaler_rob.fit_transform(data)
print("Robust Scaled:\n", data_rob)
```

### Scaling Decision Tree

```
Pertanyaan: Apakah algoritma berbasis jarak?
│
├── YA (K-NN, SVM, Neural Network, K-Means, PCA)
│   │
│   ├── Ada outlier signifikan?
│   │   ├── YA → Robust Scaler
│   │   └── TIDAK → Standard Scaler (Z-score)
│   │
│   └── Butuh range tepat [0,1]? (contoh: Neural Network dengan sigmoid)
│       └── YA → Min-Max Scaler
│
└── TIDAK (Decision Tree, Random Forest, Naive Bayes)
    └── Scaling TIDAK PERLU (DT berbasis threshold, bukan jarak)
```

---

## 3.7 Imbalanced Data — Ketika Kelas Tidak Seimbang

### Masalah

```python
# Dataset fraud detection
# 99.8% transaksi normal, 0.2% fraud
total = 100000
normal = 99800
fraud = 200

# Model yang selalu prediksi "normal" akan punya akurasi:
accuracy = 99800 / 100000
print(f"Accuracy: {accuracy:.2%}")  # 99.80%!
# Tapi model ini SAMA SEKALI tidak berguna untuk mendeteksi fraud!
```

### Solusi

#### 1. Resampling

```python
from sklearn.utils import resample

# OVERSAMPLING: Perbanyak kelas minoritas
df_majority = df[df['label'] == 'normal']
df_minority = df[df['label'] == 'fraud']

df_minority_upsampled = resample(
    df_minority,
    replace=True,         # Sample WITH replacement
    n_samples=len(df_majority),  # Samakan jumlahnya
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# UNDERSAMPLING: Kurangi kelas mayoritas
df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_balanced_v2 = pd.concat([df_minority, df_majority_downsampled])
```

#### 2. SMOTE (Synthetic Minority Over-sampling Technique)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# SMOTE membuat data sintetis baru di antara data minoritas yang ada
# Bukan sekadar menduplikasi — ini membuat data baru yang masuk akal
```

#### 3. Class Weight

```python
from sklearn.linear_model import LogisticRegression

# Beri bobot lebih besar pada kelas minoritas
model = LogisticRegression(class_weight='balanced')
# atau secara manual:
model = LogisticRegression(class_weight={0: 1, 1: 50})  # Kelas 1 (fraud) 50× lebih penting
```

#### 4. Threshold Adjustment

```python
# Default threshold: 0.5
y_proba = model.predict_proba(X_test)[:, 1]

# Turunkan threshold untuk mendeteksi lebih banyak fraud
y_pred_custom = (y_proba >= 0.3).astype(int)  # Lebih sensitif terhadap fraud
```

### Pilihan Metode

| Metode | Kelebihan | Kekurangan | Kapan? |
|--------|-----------|-----------|--------|
| Oversampling | Tidak kehilangan data | Bisa overfit | Data sedikit |
| Undersampling | Tidak duplikasi data | Kehilangan informasi | Data banyak |
| SMOTE | Data sintetis realistis | Bisa buat noise | Kasus umum |
| Class Weight | Tanpa resampling | Tidak semua algoritma support | Solusi paling simpel |
| Threshold | Mudah diubah | Trade-off precision/recall | Saat deployment |

---

## 3.8 Data Leakage — Musuh Tersembunyi

### Apa Itu Data Leakage?

**Data leakage** terjadi ketika informasi dari luar training set secara tidak sengaja masuk ke dalam training, membuat model terlihat lebih baik dari seharusnya.

Analoginya: Kamu belajar untuk ujian dengan **menghafal soal ujian itu sendiri**. Tentu saja kamu dapat nilai sempurna — tapi itu bukan karena kamu pintar, itu karena kamu curang.

### Contoh Data Leakage

#### Contoh 1: Scaling Sebelum Split

```python
# ❌ SALAH — Data leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit di SELURUH data
X_train, X_test = train_test_split(X_scaled, ...)  # Split setelah scaling

# Kenapa salah? Test data mempengaruhi mean dan std yang digunakan scaling!
# Model "melihat" informasi dari test data melalui parameter scaling.

# ✓ BENAR — Tidak ada leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit hanya di training
X_test_scaled = scaler.transform(X_test)        # Transform test dengan parameter training
```

#### Contoh 2: Feature dari Masa Depan

```python
# ❌ SALAH — Menggunakan fitur dari masa depan
df['harga_rumah_besok'] = df['harga_rumah'].shift(-1)  # Data hari besok!
features = [..., 'harga_rumah_besok', ...]  # Model "curang" — sudah tahu jawabannya

# ✓ BENAR — Hanya gunakan fitur dari masa lalu
features = [..., 'harga_rumah_kemarin', 'harga_rumah_hari_ini', ...]
```

#### Contoh 3: Target Encoding Sebelum Split

```python
# ❌ SALAH
df['kota_target_enc'] = df.groupby('kota')['harga'].transform('mean')
X_train, X_test = train_test_split(df[features], df['harga'])

# ✓ BENAR
X_train, X_test, y_train, y_test = train_test_split(df[features], df['harga'])
# Hitung encoding HANYA dari training data
mean_by_kota = X_train.join(y_train).groupby('kota')['harga'].mean()
X_train['kota_enc'] = X_train['kota'].map(mean_by_kota)
X_test['kota_enc'] = X_test['kota'].map(mean_by_kota)  # Gunakan mapping dari TRAINING
```

### Checklist Anti-Leakage

```
□ Scaling: fit hanya di training, transform di test
□ Feature engineering: fit/learn hanya di training
□ Resampling: hanya di training data
□ Target encoding: hitung hanya dari training
□ Time-series: jangan pakai data masa depan
□ Duplikat: hapus sebelum split
□ Hold-out test: JANGAN PERNAH pakai untuk tuning
```

---

## 3.9 Pipeline — Semua Langkah dalam Satu Paket

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Definisikan kolom
numeric_features = ['umur', 'gaji', 'pengalaman']
categorical_features = ['kota', 'pendidikan']

# Pipeline untuk numerik
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline untuk kategorikal
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Gabungkan
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Sekarang bisa langsung cross-validate!
scores = cross_val_score(full_pipeline, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# FIT dan PREDICT — semua preprocessing otomatis
full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)

# TIDAK ADA DATA LEAKAGE — preprocessing dilakukan di dalam CV!
```

Kenapa Pipeline penting?
1. **Mencegah data leakage** — preprocessing di dalam CV
2. **Reproducible** — semua langkah terdokumentasi
3. **Mudah deploy** — satu objek, semua proses
4. **Grid search friendly** — bisa tune preprocessing + model sekaligus

---

## 3.10 Ringkasan Bab 3

1. **Data adalah segalanya** — Garbage In, Garbage Out. 75-85% waktu ML dipakai untuk data preparation.
2. **Data harus bersih** — Tangani missing values, outlier, duplikat, inkonsistensi.
3. **Feature engineering adalah seni** — Buat fitur baru yang lebih bermakna dari data mentah.
4. **Scaling penting** untuk algoritma berbasis jarak (K-NN, SVM, NN), tidak penting untuk tree-based.
5. **Imbalanced data** butuh penanganan khusus — jangan tertipu akurasi tinggi.
6. **Data leakage** adalah musuh tersembunyi — selalu split dulu, preprocess kemudian.
7. **Pipeline** mengamankan proses dan mencegah kesalahan.

---

**Selanjutnya → Bab 4: Overfitting vs Underfitting — Musuh Besar Machine Learning**

Di bab berikutnya, kita akan mendalami masalah yang paling fundamental di ML: model yang "terlalu pintar" (overfitting) vs model yang "terlalu bodoh" (underfitting). Ini adalah kunci memahami mengapa model tidak bekerja dan bagaimana memperbaikinya.