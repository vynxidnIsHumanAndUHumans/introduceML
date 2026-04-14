# K-Nearest Neighbors (K-NN) - Penjelasan Lengkap

## Daftar Isi
1. [Konsep Dasar](#konsep-dasar)
2. [Cara Kerja Detail (Step-by-Step)](#cara-kerja-detail)
3. [Kode Python dari Nol (Tanpa Library)](#kode-python-dari-nol)
4. [Contoh Manual: 4 Data Point](#contoh-manual-4-data)
5. [Contoh Manual: 8 Data Point](#contoh-manual-8-data)
6. [Contoh Skala Besar: 1000+ Data dengan Scikit-Learn](#contoh-skala-besar)
7. [Alur Kerja Program Lengkap](#alur-kerja-program)
8. [Semua Operasi Matematika yang Dilakukan](#operasi-matematika)
9. [Variasi K dan Pengaruhnya](#variasi-k)
10. [Pro dan Kontra](#pro-dan-kontra)

---

## Konsep Dasar

K-NN adalah algoritma **lazy learning** (instance-based learning). Artinya:
- Tidak ada fase training yang sesungguhnya
- Model hanya "menghafal" semua data training
- Semua komputasi terjadi saat **prediksi** (inference time)

Analogi sederhana:
> Kamu pindah ke lingkungan baru. Kamu melihat 5 tetangga terdekatmu. Jika 4 dari 5 tetangga adalah petani, kemungkinan besar kamu juga akan menjadi petani. Inilah K-NN dengan K=5.

---

## Cara Kerja Detail

### Algoritma K-NN Classification:

```
INPUT: Data training (X_train, y_train), data baru (x_new), nilai K
OUTPUT: Label prediksi untuk x_new

LANGKAH:
1. Untuk setiap data point di X_train, hitung jarak ke x_new
2. Urutkan semua data berdasarkan jarak (dari terkecil ke terbesar)
3. Ambil K data point dengan jarak terkecil (K tetangga terdekat)
4. Hitung jumlah suara (vote) untuk setiap kelas di antara K tetangga
5. Kembalikan kelas dengan vote terbanyak (majority vote)
```

### Metode Jarak yang Digunakan

#### 1. Euclidean Distance (paling umum)
```
d(x, y) = √( Σ (xi - yi)² )
```

Untuk 2 dimensi:
```
d(x, y) = √( (x1-y1)² + (x2-y2)² )
```

#### 2. Manhattan Distance
```
d(x, y) = Σ |xi - yi|
```

#### 3. Minkowski Distance (generalisasi)
```
d(x, y) = ( Σ |xi - yi|^p )^(1/p)
```
- p=1 → Manhattan
- p=2 → Euclidean

---

## Kode Python dari Nol

```python
import math
from collections import Counter

class KNN:
    def __init__(self, k=3, distance='euclidean'):
        self.k = k
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Menyimpan data training (lazy learning - tidak ada training)"""
        self.X_train = X_train
        self.y_train = y_train

    def _ calculate_distance(self, x1, x2):
        """Menghitung jarak antara dua titik"""
        if self.distance == 'euclidean':
            sum_squared = sum((a - b) ** 2 for a, b in zip(x1, x2))
            return math.sqrt(sum_squared)
        elif self.distance == 'manhattan':
            return sum(abs(a - b) for a, b in zip(x1, x2))
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

    def _get_neighbors(self, x_new):
        """Menemukan K tetangga terdekat"""
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x_new, x_train)
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda d: d[0])
        neighbors = distances[:self.k]
        return neighbors

    def predict(self, X_new):
        """Memprediksi label untuk data baru (bisa single atau batch)"""
        if isinstance(X_new[0], (int, float)):
            X_new = [X_new]
        predictions = []
        for x in X_new:
            neighbors = self._get_neighbors(x)
            votes = [label for _, label in neighbors]
            counter = Counter(votes)
            predicted = counter.most_common(1)[0][0]
            predictions.append(predicted)
        return predictions if len(predictions) > 1 else predictions[0]

    def predict_proba(self, x_new):
        """Mengembalikan probabilitas setiap kelas"""
        neighbors = self._get_neighbors(x_new)
        votes = [label for _, label in neighbors]
        counter = Counter(votes)
        total = len(votes)
        return {label: count / total for label, count in counter.items()}

    def score(self, X_test, y_test):
        """Menghitung akurasi"""
        predictions = self.predict(X_test)
        correct = sum(1 for p, t in zip(predictions, y_test) if p == t)
        return correct / len(y_test)
```

---

## Contoh Manual: 4 Data Point

### Data Training:

| # | Berat (kg) | Tinggi (cm) | Kelas    |
|---|-----------|-------------|----------|
| 1 | 60        | 165         | Normal   |
| 2 | 80        | 175         | Gemuk    |
| 3 | 50        | 160         | Kurus    |
| 4 | 70        | 170         | Normal   |

**Data baru yang ingin diprediksi:** Berat = 65 kg, Tinggi = 168 cm

### Langkah 1: Hitung Jarak Euclidean ke Semua Data

```
d(data_baru, data_1) = √((65-60)² + (168-165)²)
                     = √(25 + 9)
                     = √34
                     = 5.831

d(data_baru, data_2) = √((65-80)² + (168-175)²)
                     = √(225 + 49)
                     = √274
                     = 16.553

d(data_baru, data_3) = √((65-50)² + (168-160)²)
                     = √(225 + 64)
                     = √289
                     = 17.000

d(data_baru, data_4) = √((65-70)² + (168-170)²)
                     = √(25 + 4)
                     = √29
                     = 5.385
```

### Langkah 2: Urutkan Berdasarkan Jarak

| Urutan | Data # | Jarak  | Kelas  |
|--------|--------|--------|--------|
| 1      | 4      | 5.385  | Normal |
| 2      | 1      | 5.831  | Normal |
| 3      | 2      | 16.553 | Gemuk  |
| 4      | 3      | 17.000 | Kurus  |

### Langkah 3: Ambil K=3 Tetangga Terdekat

Tetangga: Data #4 (Normal), Data #1 (Normal), Data #2 (Gemuk)

### Langkah 4: Voting

| Kelas | Vote |
|-------|------|
| Normal | 2    |
| Gemuk  | 1    |

### Hasil: **Normal** (mayoritas 2 dari 3 tetangga adalah "Normal")

### Eksekusi dengan Kode:

```python
knn = KNN(k=3)
X_train = [[60, 165], [80, 175], [50, 160], [70, 170]]
y_train = ['Normal', 'Gemuk', 'Kurus', 'Normal']
knn.fit(X_train, y_train)

result = knn.predict([65, 168])
print(f"Prediksi: {result}")

proba = knn.predict_proba([65, 168])
print(f"Probabilitas: {proba}")
```

**Output:**
```
Prediksi: Normal
Probabilitas: {'Normal': 0.667, 'Gemuk': 0.333}
```

---

## Contoh Manual: 8 Data Point

### Data Training:

| # | Gula (g) | Lemak (g) | Kalori | Kelas       |
|---|----------|-----------|--------|-------------|
| 1 | 5        | 1         | 80     | Sehat       |
| 2 | 2        | 0.5       | 50     | Sehat       |
| 3 | 15       | 8         | 250    | Tidak Sehat |
| 4 | 3        | 2         | 90     | Sehat       |
| 5 | 20       | 12        | 350    | Tidak Sehat |
| 6 | 8        | 3         | 150    | Sehat       |
| 7 | 18       | 10        | 300    | Tidak Sehat |
| 8 | 1        | 0.3       | 40     | Sehat       |

**Data baru:** Gula = 10g, Lemak = 5g, Kalori = 180

### Kalkulasi Jarak:

```
d(#1) = √((10-5)²  + (5-1)²    + (180-80)²)  = √(25 + 16 + 10000) = √10041  = 100.205
d(#2) = √((10-2)²  + (5-0.5)²  + (180-50)²)  = √(64 + 20.25 + 16900) = √16984.25 = 130.324
d(#3) = √((10-15)² + (5-8)²    + (180-250)²)  = √(25 + 9 + 4900)  = √4934    = 70.242
d(#4) = √((10-3)²  + (5-2)²    + (180-90)²)   = √(49 + 9 + 8100)  = √8158    = 90.322
d(#5) = √((10-20)² + (5-12)²   + (180-350)²)  = √(100 + 49 + 28900) = √29049  = 170.438
d(#6) = √((10-8)²  + (5-3)²    + (180-150)²)  = √(4 + 4 + 900)    = √908     = 30.133
d(#7) = √((10-18)² + (5-10)²   + (180-300)²)  = √(64 + 25 + 14400) = √14489   = 120.371
d(#8) = √((10-1)²  + (5-0.3)²  + (180-40)²)   = √(81 + 22.09 + 19600) = √19703.09 = 140.368
```

### Urutan Jarak (K=3):

| Urutan | Data # | Jarak   | Kelas       |
|--------|--------|---------|-------------|
| 1      | 6      | 30.133  | Sehat       |
| 2      | 3      | 70.242  | Tidak Sehat |
| 3      | 4      | 90.322  | Sehat       |

### Voting (K=3):
- Sehat: 2 vote (data #6, #4)
- Tidak Sehat: 1 vote (data #3)

**Hasil: Sehat** (2 dari 3 tetangga terdekat)

### Kode:

```python
knn = KNN(k=3)
X_train = [
    [5, 1, 80],    [2, 0.5, 50],  [15, 8, 250],
    [3, 2, 90],    [20, 12, 350], [8, 3, 150],
    [18, 10, 300], [1, 0.3, 40]
]
y_train = ['Sehat', 'Sehat', 'Tidak Sehat', 'Sehat',
           'Tidak Sehat', 'Sehat', 'Tidak Sehat', 'Sehat']
knn.fit(X_train, y_train)

result = knn.predict([10, 5, 180])
print(f"Prediksi: {result}")
print(f"Probabilitas: {knn.predict_proba([10, 5, 180])}")
```

---

## Contoh Skala Besar: 1000+ Data dengan Scikit-Learn

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# 1. GENERATE DATA SINTETIS
X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=4,
    random_state=42
)

print(f"Shape X: {X.shape}")
print(f"Shape y: {y.shape}")
print(f"Distribusi kelas: {np.bincount(y)}")

# 2. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")

# 3. FEATURE SCALING (PENTING untuk K-NN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kenapa scaling penting?
# K-NN menggunakan jarak, jika satu fitur punya range 0-1000
# dan lainnya 0-1, fitur pertama akan mendominasi perhitungan jarak

# 4. TRAINING (sebenarnya hanya menyimpan data)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
start = time.time()
knn.fit(X_train_scaled, y_train)
train_time = time.time() - start
print(f"\nWaktu 'training': {train_time:.4f} detik (lazy learning!)")

# 5. PREDIKSI
start = time.time()
y_pred = knn.predict(X_test_scaled)
pred_time = time.time() - start
print(f"Waktu prediksi (2000 data): {pred_time:.4f} detik")

# 6. EVALUASI
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 7. EKSPERIMEN NILAI K
print("\n=== Eksperimen Nilai K ===")
for k in [1, 3, 5, 7, 11, 21, 51]:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    accuracy = knn_k.score(X_test_scaled, y_test)
    print(f"K={k:3d} → Akurasi: {accuracy:.4f}")

# 8. EKSPERIMEN JARAK
print("\n=== Eksperimen Metrik Jarak ===")
for metric in ['euclidean', 'manhattan', 'chebyshev']:
    knn_m = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_m.fit(X_train_scaled, y_train)
    accuracy = knn_m.score(X_test_scaled, y_test)
    print(f"Metric={metric:12s} → Akurasi: {accuracy:.4f}")

# 9. K-NN REGRESSION
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=5000, n_features=5, noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2)

scaler_r = StandardScaler()
X_tr_s = scaler_r.fit_transform(X_tr)
X_te_s = scaler_r.transform(X_te)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_tr_s, y_tr)
print(f"\nK-NN Regression R²: {knn_reg.score(X_te_s, y_te):.4f}")

# K-NN Regression memprediksi RATA-RATA dari K tetangga (bukan voting)
```

---

## Alur Kerja Program

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALUR KERJA K-NN                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INISIALISASI                                                 │
│     ├── Set nilai K (misal K=3)                                  │
│     ├── Set metrik jarak (misal Euclidean)                       │
│     └── Siapkan list kosong untuk data training                  │
│                                                                  │
│  2. FIT (Training Phase - Lazy Learning)                         │
│     ├── Terima X_train dan y_train                               │
│     ├── Simpan X_train di memori                                 │
│     ├── Simpan y_train di memori                                 │
│     └── TIDAK ADA komputasi lain (hanya menyimpan data)          │
│                                                                  │
│  3. PREDICT (Inference Phase - Semua komputasi di sini)          │
│     │                                                            │
│     ├── Untuk setiap data baru x_new:                           │
│     │   │                                                        │
│     │   ├── LOOP: Hitung jarak ke semua data training            │
│     │   │   │                                                    │
│     │   │   ├── data_train[0]: d = √(Σ(xi - x0i)²)              │
│     │   │   ├── data_train[1]: d = √(Σ(xi - x1i)²)              │
│     │   │   ├── data_train[2]: d = √(Σ(xi - x2i)²)              │
│     │   │   └── ...                                              │
│     │   │                                                        │
│     │   ├── SIMPAN semua (jarak, label) ke list                  │
│     │   │                                                        │
│     │   ├── SORT list berdasarkan jarak (ascending)              │
│     │   │                                                        │
│     │   ├── AMBIL K data pertama (K tetangga terdekat)            │
│     │   │                                                        │
│     │   ├── VOTING: Hitung frekuensi setiap kelas                │
│     │   │   ├── Kelas A: 2 vote                                  │
│     │   │   └── Kelas B: 1 vote                                  │
│     │   │                                                        │
│     │   └── RETURN kelas dengan vote terbanyak                   │
│     │                                                            │
│     └── Return semua prediksi                                    │
│                                                                  │
│  4. EVALUASI                                                     │
│     ├── Bandingkan prediksi dengan label sebenarnya              │
│     ├── Hitung akurasi = benar / total                           │
│     └── Optional: Confusion matrix, precision, recall, F1      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Operasi Matematika

### Setiap Prediksi Melibatkan Operasi Berikut:

Jika ada **N** data training dengan **M** fitur, dan **K** tetangga:

| # | Operasi | Jumlah Operasi | Detail |
|---|---------|---------------|--------|
| 1 | Pengurangan | N × M | `(xi - yi)` untuk setiap fitur setiap data |
| 2 | Pengkuadratan | N × M | `(xi - yi)²` |
| 3 | Penjumlahan | N × (M-1) | Penjumlahan semua squared differences |
| 4 | Akar kuadrat | N | `√(sum)` |
| 5 | Sorting | N log N | Mengurutkan N jarak |
| 6 | Pengambilan | K | Mengambil K tetangga terdekat |
| 7 | Penghitungan | K | Menghitung vote setiap kelas |
| 8 | Pembandingan | C | Menentukan kelas mayoritas (C = jumlah kelas) |

**Total kompleksitas: O(N × M + N log N)** per prediksi

### Contoh Perhitungan Detail untuk 1 Prediksi:

Data baru: `[65, 168]`, Data training pertama: `[60, 165]`

```
Langkah 1 - Pengurangan:
  65 - 60 = 5
  168 - 165 = 3

Langkah 2 - Pengkuadratan:
  5² = 25
  3² = 9

Langkah 3 - Penjumlahan:
  25 + 9 = 34

Langkah 4 - Akar Kuadrat:
  √34 = 5.83095...

Langkah 5 - (Ulangi langkah 1-4 untuk semua N data)

Langkah 6 - Sorting semua jarak

Langkah 7 - Ambil K=3 terdekat

Langkah 8 - Voting:
  Tetangga 1: Normal (jarak 5.385)
  Tetangga 2: Normal (jarak 5.831)
  Tetangga 3: Gemuk  (jarak 16.553)

  Count: Normal=2, Gemuk=1

Langkah 9 - Return: Normal
```

---

## Variasi K dan Pengaruhnya

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                           n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

k_values = range(1, 51)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    train_scores.append(knn.score(X_train_s, y_train))
    test_scores.append(knn.score(X_test_s, y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, label='Training Accuracy')
plt.plot(k_values, test_scores, label='Testing Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.title('K-NN: Pengaruh Nilai K terhadap Akurasi')
plt.savefig('knn_k_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Penjelasan Pengaruh K:

| Nilai K | Karakteristik | Risiko |
|---------|--------------|--------|
| K=1 | Sangat sensitif terhadap noise | **Overfitting** - mengikuti noise |
| K kecil (3-5) | Fleksibel, mengikuti pola lokal | Sedikit overfitting |
| K sedang (7-15) | Keseimbangan bias-variance | Biasanya optimal |
| K besar (20+) | Sangat halus, pola sederhana | **Underfitting** - terlalu general |
| K=N | Selalu prediksi kelas mayoritas | Underfitting total |

**Aturan umum:** Pilih K ganjil (untuk menghindari tie), dan K ≈ √N (N = jumlah data training)

---

## Pro dan Kontra

### Pro:
1. **Sederhana** - Mudah dipahami dan diimplementasikan
2. **Tanpa training** - Data baru bisa ditambahkan tanpa retrain
3. **Non-parametrik** - Tidak mengasumsikan distribusi data
4. **Multiclass natural** - Langsung mendukung multi-kelas
5. **Versatile** - Bisa classification dan regression

### Kontra:
1. **Lambat saat prediksi** - Harus menghitung jarak ke semua data (O(N×M))
2. **Memori besar** - Harus menyimpan semua data training
3. **Curse of dimensionality** - Performa turun drastis di dimensi tinggi
4. **Sensitif terhadap fitur irrelevant** - Fitur tidak penting mengganggu jarak
5. **Sensitif terhadap scaling** - Harus dinormalisasi/standardisasi
6. **Imbalanced data** - Kelas mayoritas bisa mendominasi voting

### Solusi untuk Kontra:

| Masalah | Solusi |
|---------|--------|
| Lambat | KD-Tree, Ball Tree, Approximate K-NN |
| Memori | Prototype selection, condensing |
| Dimensi tinggi | PCA, feature selection |
| Scaling | StandardScaler, MinMaxScaler |
| Imbalanced | Weighted K-NN (bobot berdasarkan jarak) |
| Irrelevant features | Feature selection, PCA |

### Weighted K-NN:

```python
class WeightedKNN(KNN):
    def predict(self, X_new):
        if isinstance(X_new[0], (int, float)):
            X_new = [X_new]
        predictions = []
        for x in X_new:
            neighbors = self._get_neighbors(x)
            # Bobot = 1/jarak (jarak kecil = bobot besar)
            class_weights = {}
            for dist, label in neighbors:
                weight = 1 / (dist + 1e-10)  # +epsilon agar tidak divide by zero
                class_weights[label] = class_weights.get(label, 0) + weight
            predicted = max(class_weights, key=class_weights.get)
            predictions.append(predicted)
        return predictions if len(predictions) > 1 else predictions[0]
```

Weighted K-NN memberikan bobot lebih besar pada tetangga yang lebih dekat, sehingga tetangga jauh tidak terlalu mempengaruhi hasil.