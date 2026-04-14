# Naive Bayes - Penjelasan Lengkap

## Daftar Isi
1. [Konsep Dasar](#konsep-dasar)
2. [Teorema Bayes](#teorema-bayes)
3. [Cara Kerja Detail (Step-by-Step)](#cara-kerja-detail)
4. [Kode Python dari Nol (Tanpa Library)](#kode-python-dari-nol)
5. [Contoh Manual: 4 Data Point (Kategorikal)](#contoh-manual-4-data)
6. [Contoh Manual: 8 Data Point (Gaussian)](#contoh-manual-8-data)
7. [Contoh Skala Besar: 10000+ Data dengan Scikit-Learn](#contoh-skala-besar)
8. [Alur Kerja Program Lengkap](#alur-kerja-program)
9. [Semua Operasi Matematika yang Dilakukan](#operasi-matematika)
10. [Variasi Naive Bayes](#variasi-naive-bayes)
11. [Pro dan Kontra](#pro-dan-kontra)

---

## Konsep Dasar

Naive Bayes adalah algoritma **probabilistic classifier** berdasarkan Teorema Bayes dengan asumsi **"naive"** bahwa semua fitur **independen** (tidak saling mempengaruhi).

Analogi:
> Seorang dokter mendiagnosis penyakit. Ia melihat gejala A, B, dan C secara terpisah. Ia menghitung probabilitas setiap gejala muncul pada setiap penyakit, lalu mengalikannya. Penyakit dengan probabilitas tertinggi menjadi diagnosa.

Kenapa "Naive"? Karena asumsi independensi ini hampir **tidak pernah** terpenuhi di dunia nyata. Tapi anehnya, algoritma ini **tetap bekerja dengan baik**!

---

## Teorema Bayes

### Rumus Dasar:

```
             P(A|B) × P(B)
P(B|A) = ─────────────────
                P(A)
```

### Dalam konteks Machine Learning:

```
                  P(X|Ck) × P(Ck)
P(Ck|X) = ─────────────────────────
                       P(X)
```

Di mana:
- `P(Ck|X)` = **Posterior** - probabilitas kelas Ck given data X (YANG INGIN KITA CARI)
- `P(X|Ck)` = **Likelihood** - probabilitas data X given kelas Ck
- `P(Ck)` = **Prior** - probabilitas kelas Ck sebelum melihat data
- `P(X)` = **Evidence** - probabilitas data X (sama untuk semua kelas, bisa diabaikan)

### Asumsi Naive (Independensi):

```
P(X|Ck) = P(x1|Ck) × P(x2|Ck) × ... × P(xn|Ck)
```

Sehingga:

```
P(Ck|X) ∝ P(Ck) × Π P(xi|Ck)
```

Kita tidak perlu menghitung P(X) karena sama untuk semua kelas. Cukup cari kelas dengan **P(Ck) × Π P(xi|Ck)** terbesar.

---

## Cara Kerja Detail

### Algoritma Naive Bayes (Training):

```
INPUT: Data training (X, y)
OUTPUT: Parameter model (prior, likelihood untuk setiap kelas dan fitur)

TRAINING:
1. Untuk setiap kelas Ck:
   a. Hitung prior: P(Ck) = count(Ck) / N
   b. Untuk setiap fitur xi:
      - Hitung P(xi|Ck) berdasarkan tipe data:
        * Kategorikal: P(xi=v|Ck) = count(xi=v dalam Ck) / count(Ck)
        * Numerik (Gaussian): Hitung mean dan variance fitur xi di kelas Ck
```

### Algoritma Naive Bayes (Prediction):

```
PREDICTION:
1. Untuk setiap kelas Ck:
   a. Mulai dengan log(P(Ck))
   b. Untuk setiap fitur xi:
      - Tambahkan log(P(xi|Ck))
   c. Hitung log_posterior(Ck) = log(P(Ck)) + Σ log(P(xi|Ck))
2. Return kelas dengan log_posterior terbesar
```

Kenapa menggunakan **log**? Karena perkalian banyak probabilitas kecil → **underflow** (angka terlalu kecil untuk direpresentasikan di komputer). Log mengubah perkalian jadi penjumlahan.

---

## Kode Python dari Nol

### Gaussian Naive Bayes (untuk fitur numerik kontinu)

```python
import math

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.variances = {}
        self.n_features = None

    def fit(self, X, y):
        """Training: Hitung prior, mean, dan variance setiap kelas"""
        n_samples = len(y)
        self.classes = list(set(y))
        self.n_features = len(X[0])

        for cls in self.classes:
            # Filter data untuk kelas ini
            X_cls = [X[i] for i in range(n_samples) if y[i] == cls]
            n_cls = len(X_cls)

            # Prior: P(class)
            self.priors[cls] = n_cls / n_samples

            # Mean dan Variance untuk setiap fitur
            self.means[cls] = []
            self.variances[cls] = []

            for j in range(self.n_features):
                values = [row[j] for row in X_cls]
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                # + epsilon agar tidak divide by zero
                variance = max(variance, 1e-9)
                self.means[cls].append(mean)
                self.variances[cls].append(variance)

    def _gaussian_pdf(self, x, mean, variance):
        """Fungsi Densitas Probabilitas Gaussian (Normal Distribution)"""
        exponent = -((x - mean) ** 2) / (2 * variance)
        coefficient = 1 / math.sqrt(2 * math.pi * variance)
        return coefficient * math.exp(exponent)

    def predict(self, X):
        """Prediksi: Cari kelas dengan posterior terbesar"""
        if isinstance(X[0], (int, float)):
            X = [X]
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        posteriors = {}
        for cls in self.classes:
            # log(Prior)
            log_posterior = math.log(self.priors[cls])

            # Tambahkan log(P(xi|class)) untuk setiap fitur
            for j in range(self.n_features):
                likelihood = self._gaussian_pdf(x[j], self.means[cls][j],
                                                 self.variances[cls][j])
                log_posterior += math.log(likelihood)

            posteriors[cls] = log_posterior

        return max(posteriors, key=posteriors.get)

    def predict_proba(self, X):
        """Mengembalikan probabilitas setiap kelas"""
        if isinstance(X[0], (int, float)):
            X = [X]
        results = []
        for x in X:
            log_posteriors = {}
            for cls in self.classes:
                log_posterior = math.log(self.priors[cls])
                for j in range(self.n_features):
                    likelihood = self._gaussian_pdf(x[j], self.means[cls][j],
                                                     self.variances[cls][j])
                    log_posterior += math.log(likelihood)
                log_posteriors[cls] = log_posterior

            # Konversi dari log ke probabilitas (softmax-like)
            max_log = max(log_posteriors.values())
            posteriors = {cls: math.exp(lp - max_log) for cls, lp in log_posteriors.items()}
            total = sum(posteriors.values())
            probs = {cls: p / total for cls, p in posteriors.items()}
            results.append(probs)
        return results

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = sum(1 for p, t in zip(predictions, y_test) if p == t)
        return correct / len(y_test)
```

### Multinomial Naive Bayes (untuk fitur kategorikal/count)

```python
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.classes = None
        self.priors = {}
        self.likelihoods = {}

    def fit(self, X, y):
        n_samples = len(y)
        self.classes = list(set(y))

        for cls in self.classes:
            X_cls = [X[i] for i in range(n_samples) if y[i] == cls]
            n_cls = len(X_cls)
            self.priors[cls] = n_cls / n_samples

            n_features = len(X[0])
            self.likelihoods[cls] = []

            for j in range(n_features):
                values = [row[j] for row in X_cls]
                total_count = sum(values)
                # Laplace smoothing
                vocab_size = n_features
                likelihood_for_j = (sum(values) + self.alpha) / (total_count + self.alpha * vocab_size)
                self.likelihoods[cls].append(likelihood_for_j)

    def predict(self, X):
        if isinstance(X[0], (int, float)):
            X = [X]
        results = []
        for x in X:
            best_cls = None
            best_log_prob = float('-inf')
            for cls in self.classes:
                log_prob = math.log(self.priors[cls])
                for j, val in enumerate(x):
                    log_prob += val * math.log(self.likelihoods[cls][j] + 1e-10)
                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_cls = cls
            results.append(best_cls)
        return results
```

---

## Contoh Manual: 4 Data Point

### Dataset: Prediksi Cuaca untuk Bermain Tenis (Kategorikal)

| # | Cuaca    | Suhu     | Angin   | Bermain? |
|---|----------|----------|---------|----------|
| 1 | Cerah    | Panas    | Lemah   | Tidak    |
| 2 | Mendung  | Panas    | Lemah   | Tidak    |
| 3 | Hujan    | Sedang   | Kuat    | Ya       |
| 4 | Hujan    | Dingin   | Lemah   | Ya       |

### Training Phase:

#### 1. Hitung Prior:
```
Total data = 4
P(Ya)    = 2/4 = 0.5
P(Tidak) = 2/4 = 0.5
```

#### 2. Hitung Likelihood (Conditional Probability) dengan Laplace Smoothing:

**Untuk kelas "Ya" (2 data):**

P(Cuaca|Ya):
```
P(Cerah|Ya)   = (0 + 1) / (2 + 3) = 1/5 = 0.2    ← Laplace: +1, vocab_size=3
P(Mendung|Ya) = (0 + 1) / (2 + 3) = 1/5 = 0.2
P(Hujan|Ya)   = (2 + 1) / (2 + 3) = 3/5 = 0.6
```

P(Suhu|Ya):
```
P(Panas|Ya)   = (0 + 1) / (2 + 3) = 1/5 = 0.2
P(Sedang|Ya)  = (1 + 1) / (2 + 3) = 2/5 = 0.4
P(Dingin|Ya)  = (1 + 1) / (2 + 3) = 2/5 = 0.4
```

P(Angin|Ya):
```
P(Lemah|Ya)   = (1 + 1) / (2 + 2) = 2/4 = 0.5
P(Kuat|Ya)    = (1 + 1) / (2 + 2) = 2/4 = 0.5
```

**Untuk kelas "Tidak" (2 data):**

P(Cuaca|Tidak):
```
P(Cerah|Tidak)   = (1 + 1) / (2 + 3) = 2/5 = 0.4
P(Mendung|Tidak) = (1 + 1) / (2 + 3) = 2/5 = 0.4
P(Hujan|Tidak)   = (0 + 1) / (2 + 3) = 1/5 = 0.2
```

P(Suhu|Tidak):
```
P(Panas|Tidak)   = (2 + 1) / (2 + 3) = 3/5 = 0.6
P(Sedang|Tidak)  = (0 + 1) / (2 + 3) = 1/5 = 0.2
P(Dingin|Tidak)  = (0 + 1) / (2 + 3) = 1/5 = 0.2
```

P(Angin|Tidak):
```
P(Lemah|Tidak)   = (2 + 1) / (2 + 2) = 3/4 = 0.75
P(Kuat|Tidak)   = (0 + 1) / (2 + 2) = 1/4 = 0.25
```

### Prediction Phase:

**Data baru:** Cuaca = Hujan, Suhu = Sedang, Angin = Lemah

```
P(Ya|Data) ∝ P(Ya) × P(Hujan|Ya) × P(Sedang|Ya) × P(Lemah|Ya)
           = 0.5 × 0.6 × 0.4 × 0.5
           = 0.06

P(Tidak|Data) ∝ P(Tidak) × P(Hujan|Tidak) × P(Sedang|Tidak) × P(Lemah|Tidak)
              = 0.5 × 0.2 × 0.2 × 0.75
              = 0.015
```

**Normalisasi:**
```
P(Ya|Data)    = 0.06 / (0.06 + 0.015) = 0.06 / 0.075 = 0.80
P(Tidak|Data) = 0.015 / (0.06 + 0.015) = 0.015 / 0.075 = 0.20
```

**Hasil: Ya (80% probabilitas)**

### Mengapa Menggunakan Log?

```
Tanpa log: 0.5 × 0.6 × 0.4 × 0.5 = 0.06 (masih ok)

Tapi bayangkan 100 fitur:
0.5 × 0.6 × 0.4 × 0.5 × ... × 0.3 × 0.1
= angka yang sangat kecil (underflow!)

Dengan log:
log(0.5) + log(0.6) + log(0.4) + log(0.5)
= -0.693 + (-0.511) + (-0.916) + (-0.693)
= -2.813 (tidak underflow)
```

---

## Contoh Manual: 8 Data Point

### Dataset: Klasifikasi Bunga (Gaussian - Fitur Numerik)

| # | Panjang Kelopak | Lebar Kelopak | Kelas     |
|---|----------------|---------------|-----------|
| 1 | 1.4            | 0.2           | Setosa    |
| 2 | 1.3            | 0.4           | Setosa    |
| 3 | 1.5            | 0.3           | Setosa    |
| 4 | 4.7            | 1.4           | Versicolor|
| 5 | 4.5            | 1.5           | Versicolor|
| 6 | 4.9            | 1.3           | Versicolor|
| 7 | 5.1            | 1.8           | Virginica |
| 8 | 6.0            | 2.5           | Virginica |

### Training: Hitung Mean dan Variance

**Setosa (3 data):**

Panjang Kelopak:
```
Mean = (1.4 + 1.3 + 1.5) / 3 = 4.2 / 3 = 1.4
Var  = ((1.4-1.4)² + (1.3-1.4)² + (1.5-1.4)²) / 3
     = (0 + 0.01 + 0.01) / 3 = 0.02 / 3 = 0.0067
```

Lebar Kelopak:
```
Mean = (0.2 + 0.4 + 0.3) / 3 = 0.9 / 3 = 0.3
Var  = ((0.2-0.3)² + (0.4-0.3)² + (0.3-0.3)²) / 3
     = (0.01 + 0.01 + 0) / 3 = 0.02 / 3 = 0.0067
```

**Versicolor (3 data):**

Panjang Kelopak:
```
Mean = (4.7 + 4.5 + 4.9) / 3 = 14.1 / 3 = 4.7
Var  = ((4.7-4.7)² + (4.5-4.7)² + (4.9-4.7)²) / 3
     = (0 + 0.04 + 0.04) / 3 = 0.08 / 3 = 0.0267
```

Lebar Kelopak:
```
Mean = (1.4 + 1.5 + 1.3) / 3 = 4.2 / 3 = 1.4
Var  = ((1.4-1.4)² + (1.5-1.4)² + (1.3-1.4)²) / 3
     = (0 + 0.01 + 0.01) / 3 = 0.02 / 3 = 0.0067
```

**Virginica (2 data):**

Panjang Kelopak:
```
Mean = (5.1 + 6.0) / 2 = 5.55
Var  = ((5.1-5.55)² + (6.0-5.55)²) / 2
     = (0.2025 + 0.2025) / 2 = 0.2025
```

Lebar Kelopak:
```
Mean = (1.8 + 2.5) / 2 = 2.15
Var  = ((1.8-2.15)² + (2.5-2.15)²) / 2
     = (0.1225 + 0.1225) / 2 = 0.1225
```

### Ringkasan Parameter:

| Kelas     | Prior | Mean PK | Var PK  | Mean LK | Var LK  |
|-----------|-------|---------|---------|---------|---------|
| Setosa    | 3/8   | 1.4     | 0.0067  | 0.3     | 0.0067  |
| Versicolor| 3/8   | 4.7     | 0.0267  | 1.4     | 0.0067  |
| Virginica | 2/8   | 5.55    | 0.2025  | 2.15    | 0.1225  |

### Prediction: Data baru - Panjang=3.0, Lebar=1.0

Kita perlu menghitung Gaussian PDF untuk setiap fitur di setiap kelas:

```
Gaussian PDF: f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))
```

**Setosa:**
```
P(PK=3.0|Setosa) = (1/√(2π×0.0067)) × e^(-(3.0-1.4)²/(2×0.0067))
                 = (1/0.1449) × e^(-(2.56/0.0134))
                 = 6.902 × e^(-191.04)
                 ≈ 6.902 × 2.4×10⁻⁸³
                 ≈ 1.66×10⁻⁸² (sangat kecil!)

P(LK=1.0|Setosa) = (1/√(2π×0.0067)) × e^(-(1.0-0.3)²/(2×0.0067))
                 = 6.902 × e^(-(0.49/0.0134))
                 = 6.902 × e^(-36.57)
                 ≈ 6.902 × 1.57×10⁻¹⁶
                 ≈ 1.08×10⁻¹⁵ (sangat kecil)

P(Setosa|X) ∝ (3/8) × 1.66×10⁻⁸² × 1.08×10⁻¹⁵ ≈ ~0 (praktis nol)
```

**Versicolor:**
```
P(PK=3.0|Versicolor) = (1/√(2π×0.0267)) × e^(-(3.0-4.7)²/(2×0.0267))
                     = (1/0.2892) × e^(-(2.89/0.0534))
                     = 3.458 × e^(-54.12)
                     ≈ 3.458 × 2.62×10⁻²⁴
                     ≈ 9.06×10⁻²⁴

P(LK=1.0|Versicolor) = (1/√(2π×0.0067)) × e^(-(1.0-1.4)²/(2×0.0067))
                     = 6.902 × e^(-(0.16/0.0134))
                     = 6.902 × e^(-11.94)
                     ≈ 6.902 × 6.52×10⁻⁶
                     ≈ 4.5×10⁻⁵

P(Versicolor|X) ∝ (3/8) × 9.06×10⁻²⁴ × 4.5×10⁻⁵ ≈ 1.53×10⁻²⁸
```

**Virginica:**
```
P(PK=3.0|Virginica) = (1/√(2π×0.2025)) × e^(-(3.0-5.55)²/(2×0.2025))
                   = (1/0.7982) × e^(-(6.5025/0.405))
                   = 1.253 × e^(-16.056)
                   ≈ 1.253 × 1.03×10⁻⁷
                   ≈ 1.29×10⁻⁷

P(LK=1.0|Virginica) = (1/√(2π×0.1225)) × e^(-(1.0-2.15)²/(2×0.1225))
                   = (1/0.6218) × e^(-(1.3225/0.245))
                   = 1.608 × e^(-5.398)
                   ≈ 1.608 × 4.52×10⁻³
                   ≈ 7.27×10⁻³

P(Virginica|X) ∝ (2/8) × 1.29×10⁻⁷ × 7.27×10⁻³ ≈ 2.35×10⁻¹⁰
```

**Perbandingan (menggunakan log untuk kejelasan):**
```
log P(Setosa|X)    ∝ -∞ (praktis nol)
log P(Versicolor|X) ∝ -64.5
log P(Virginica|X)  ∝ -22.2
```

**Hasil: Versicolor** (posterior terbesar, meskipun Virginica juga cukup tinggi)

> Catatan: Angka-angka di atas sangat kecil karena contoh kecil — inilah kenapa log digunakan di implementasi nyata.

### Kode untuk Contoh Ini:

```python
gnb = GaussianNaiveBayes()
X = [
    [1.4, 0.2], [1.3, 0.4], [1.5, 0.3],
    [4.7, 1.4], [4.5, 1.5], [4.9, 1.3],
    [5.1, 1.8], [6.0, 2.5]
]
y = ['Setosa', 'Setosa', 'Setosa',
     'Versicolor', 'Versicolor', 'Versicolor',
     'Virginica', 'Virginica']
gnb.fit(X, y)

result = gnb.predict([3.0, 1.0])
print(f"Prediksi: {result}")

proba = gnb.predict_proba([[3.0, 1.0]])
print(f"Probabilitas: {proba}")
```

---

## Contoh Skala Besar: 10000+ Data dengan Scikit-Learn

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.datasets import make_classification, load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import time

# ============================================
# CONTOH 1: Gaussian NB - Dataset Sintetis
# ============================================
print("=" * 60)
print("CONTOH 1: Gaussian NB - 50,000 Data Sintetis")
print("=" * 60)

X, y = make_classification(
    n_samples=50000,
    n_features=20,
    n_informative=15,
    n_classes=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()

# Training
start = time.time()
gnb.fit(X_train, y_train)
train_time = time.time() - start
print(f"Waktu training: {train_time:.4f} detik (SANGAT CEPAT!)")

# Prediction
start = time.time()
y_pred = gnb.predict(X_test)
pred_time = time.time() - start
print(f"Waktu prediksi ({len(X_test)} data): {pred_time:.4f} detik")

print(f"Train accuracy: {gnb.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {gnb.score(X_test, y_test):.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ============================================
# CONTOH 2: Iris Dataset (Real Data)
# ============================================
print("\n" + "=" * 60)
print("CONTOH 2: Gaussian NB - Iris Dataset")
print("=" * 60)

iris = load_iris()
X_ir, y_ir = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_ir, y_ir, test_size=0.3, random_state=42)

gnb_iris = GaussianNB()
gnb_iris.fit(X_tr, y_tr)

print(f"Test accuracy: {gnb_iris.score(X_te, y_te):.4f}")
print(f"Class priors: {gnb_iris.class_prior_}")
print(f"Class means:\n{gnb_iris.theta_}")  # theta = mean
print(f"Class variances:\n{gnb_iris.var_}")

# ============================================
# CONTOH 3: Text Classification (20 Newsgroups)
# ============================================
print("\n" + "=" * 60)
print("CONTOH 3: Multinomial NB - Text Classification")
print("=" * 60)

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(newsgroups.data, newsgroups.target,
                                            test_size=0.3, random_state=42)

# Pipeline: TF-IDF + Multinomial NB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('nb', MultinomialNB(alpha=0.1))
])

pipeline.fit(X_tr, y_tr)
print(f"Test accuracy: {pipeline.score(X_te, y_te):.4f}")

y_pred = pipeline.predict(X_te)
print("\n=== Classification Report ===")
print(classification_report(y_te, y_pred, target_names=categories))

# Test dengan teks baru
new_texts = [
    "God exists and Jesus is our savior",
    "My computer has a graphics card issue",
    "New medical treatment for cancer",
    "I don't believe in any religion"
]
predictions = pipeline.predict(new_texts)
for text, pred in zip(new_texts, predictions):
    print(f"  '{text[:50]}...' → {categories[pred]}")

# ============================================
# CONTOH 4: Spam Detection (Bernoulli NB)
# ============================================
print("\n" + "=" * 60)
print("CONTOH 4: Bernoulli NB - Spam Detection")
print("=" * 60)

emails = [
    "Win free money now click here",
    "Meeting at 3pm tomorrow",
    "Cheap viagra prescription online",
    "Project update attached",
    "Congratulations you won a prize",
    "Can we discuss the report?",
    "Limited offer buy now discount",
    "Lunch at noon?",
    "Free credit card offer",
    "Review the document I sent"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham

vectorizer = CountVectorizer(binary=True)  # Binary = Bernoulli
X_emails = vectorizer.fit_transform(emails)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_emails, labels)

test_emails = [
    "Free offer click now",
    "Please review the attached report",
    "Win discount prescription"
]
X_test_e = vectorizer.transform(test_emails)
preds = bnb.predict(X_test_e)
probs = bnb.predict_proba(X_test_e)

for email, pred, prob in zip(test_emails, preds, probs):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"  '{email}' → {label} (spam prob: {prob[1]:.2%})")

# ============================================
# EKSPERIMEN: Perbandingan Semua Varian NB
# ============================================
print("\n" + "=" * 60)
print("EKSPERIMEN: Perbandingan Varian Naive Bayes")
print("=" * 60)

X_exp, y_exp = make_classification(n_samples=5000, n_features=20,
                                    n_classes=3, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X_exp, y_exp, test_size=0.3)

for name, model in [('Gaussian', GaussianNB()),
                     ('Bernoulli', BernoulliNB())]:
    model.fit(Xtr, ytr)
    score = model.score(Xte, yte)
    cv = cross_val_score(model, Xtr, ytr, cv=5).mean()
    print(f"{name:12s} | Test={score:.4f} | CV={cv:.4f}")
```

---

## Alur Kerja Program

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      ALUR KERJA NAIVE BAYES                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. INISIALISASI                                                           │
│     ├── Set tipe: Gaussian / Multinomial / Bernoulli                       │
│     ├── Set alpha (Laplace smoothing parameter)                             │
│     └── Siapkan dict kosong untuk priors, means, variances                 │
│                                                                             │
│  2. FIT (Training Phase)                                                   │
│     │                                                                       │
│     ├── Hitung PRIOR untuk setiap kelas:                                    │
│     │   P(Ck) = count(Ck) / N                                              │
│     │                                                                       │
│     ├── Untuk SETIAP KELAS Ck:                                              │
│     │   ├── Filter data yang termasuk kelas Ck                             │
│     │   │                                                                   │
│     │   ├── Jika GAUSSIAN:                                                  │
│     │   │   FOR setiap fitur j:                                             │
│     │   │     ├── Hitung mean μ(Ck, j) = Σx / n                           │
│     │   │     └── Hitung variance σ²(Ck, j) = Σ(x-μ)² / n                 │
│     │   │                                                                   │
│     │   ├── Jika MULTINOMIAL:                                               │
│     │   │   FOR setiap fitur j:                                             │
│     │   │     └── P(xj|Ck) = (count(xj dalam Ck) + α) / (total + αV)       │
│     │   │                                                                   │
│     │   └── Jika BERNOULLI:                                                 │
│     │       FOR setiap fitur j:                                             │
│     │         ├── P(xj=1|Ck) = (count(xj=1 dalam Ck) + α) / (n + 2α)      │
│     │         └── P(xj=0|Ck) = 1 - P(xj=1|Ck)                              │
│     │                                                                       │
│     └── Simpan semua parameter                                             │
│                                                                             │
│  3. PREDICT (Inference Phase)                                              │
│     │                                                                       │
│     └── Untuk setiap data baru x:                                          │
│         │                                                                   │
│         ├── Untuk SETIAP KELAS Ck:                                          │
│         │   ├── Inisialisasi: log_posterior = log(P(Ck))                   │
│         │   │                                                               │
│         │   ├── FOR setiap fitur j:                                         │
│         │   │   ├── GAUSSIAN:                                               │
│         │   │   │   likelihood = Gaussian_PDF(xj, μ(Ck,j), σ²(Ck,j))      │
│         │   │   │   log_posterior += log(likelihood)                         │
│         │   │   │                                                           │
│         │   │   ├── MULTINOMIAL:                                            │
│         │   │   │   log_posterior += xj × log(P(xj|Ck))                    │
│         │   │   │                                                           │
│         │   │   └── BERNOULLI:                                             │
│         │   │       log_posterior += log(P(xj=xj_val|Ck))                   │
│         │   │                                                               │
│         │   └── Simpan log_posterior(Ck)                                    │
│         │                                                                   │
│         └── Return kelas dengan log_posterior TERBESAR                     │
│                                                                             │
│  4. EVALUASI                                                               │
│     ├── Accuracy, Precision, Recall, F1                                    │
│     ├── Confusion Matrix                                                    │
│     └── Log Loss (jika probabilitas dipakai)                                │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Operasi Matematika

### Gaussian PDF (Fungsi Densitas Probabilitas)

```
         1              (x - μ)²
f(x) = ───────── × e^(- ──────────)
       √(2πσ²)          2σ²
```

### Operasi dalam Training:

| Operasi | Jumlah | Detail |
|---------|---------|--------|
| Penjumlahan | N × F × C | Untuk hitung sum setiap fitur setiap kelas |
| Pembagian | F × C | Hitung mean |
| Pengkuadratan | N × F × C | Untuk variance |
| Pengurangan | N × F × C | x - mean |
| Perkalian | F × C | Variance |
| Prior calculation | C | Count / N |

### Operasi dalam Prediction:

| Operasi | Jumlah | Detail |
|---------|---------|--------|
| Pengurangan | F × C | x - mean |
| Pengkuadratan | F × C | (x - μ)² |
| Pembagian | F × C | / (2σ²) |
| Eksponensial | F × C | e^(-result) |
| Perkalian skalar | F × C | coefficient × exp |
| Log | F × C + C | log likelihood + log prior |
| Penjumlahan | F × C | Σ log likelihood |
| Perbandingan | C | Cari max posterior |

**Kompleksitas Training: O(N × F × C)**
**Kompleksitas Prediction: O(F × C)** → SANGAT CEPAT!

---

## Variasi Naive Bayes

| Varian | Tipe Data | Asumsi Distribusi | Contoh Penggunaan |
|--------|-----------|-------------------|-------------------|
| **Gaussian** | Numerik kontinu | Distribusi Normal (Bell Curve) | Iris, sensor data |
| **Multinomial** | Count/frekuensi | Distribusi Multinomial | Text classification, word counts |
| **Bernoulli** | Binary (0/1) | Distribusi Bernoulli | Spam filter (ada/tidaknya kata) |
| **Complement** | Count (imbalance) | Modifikasi Multinomial | Imbalanced text data |
| **Categorical** | Kategorikal | Distribusi Kategorikal | Weather prediction |

### Contoh Perbedaan:

```python
# Email: "free money win"

# Multinomial NB: Menghitung FREKUENSI kata
#   "free" muncul 1x, "money" muncul 1x, "win" muncul 1x
#   → Fitur: [1, 1, 1] (counts)

# Bernoulli NB: Menghitung ADA/TIDAKNYA kata
#   "free" ada, "money" ada, "win" ada
#   → Fitur: [1, 1, 1] (binary)

# Perbedaan muncul jika kata muncul >1 kali:
# Email: "free free free money"
# Multinomial: [3, 1, 0]
# Bernoulli:   [1, 1, 0]
```

### Laplace Smoothing

Tanpa smoothing, jika ada fitur yang **tidak pernah muncul** di kelas tertentu:
```
P(kata|spam) = 0
→ Seluruh posterior = 0 (dihilangkan!)
```

Dengan Laplace Smoothing (α=1):
```
P(kata|spam) = (0 + 1) / (N_spam + V)
```
di mana V = jumlah vocabulary (fitur unik)

---

## Pro dan Kontra

### Pro:
1. **Sangat cepat** - Training O(NFC), Prediction O(FC)
2. **Sedikit data** - Bekerja baik dengan data training kecil
3. **Skalabel** - Performa konsisten walau data besar
4. **Text champion** - Terbaik untuk text classification
5. **Probabilitas** - Output probabilitas, bukan hanya label
6. **Multi-class** - Secara natural mendukung multi-kelas
7. **Online learning** - Bisa di-update incrementally

### Kontra:
1. **Asumsi independensi** - Fitur real hampir never independen
2. **Gaussian assumption** - Data real sering tidak normal
3. **Zero frequency** - Tanpa smoothing, fitur baru = nol
4. **Sensitif fitur irrelevant** - Fitur noise menurunkan akurasi
5. **EstimasiProbability jelek** - Posterior sering terlalu ekstrem (overconfident)
6. **Tidak menangkap interaksi** - Karena asumsi independen

### Kapan Pakai Naive Bayes?

| Scenario | Rekomendasi |
|----------|-------------|
| Text classification (spam, sentiment) | Multinomial/Bernoulli NB |
| Baseline model | Gaussian NB (cepat, simple) |
| Real-time prediction | Semua varian (prediction O(FC)) |
| Sedikit data | NB lebih baik dari model kompleks |
| Feature independen (approx) | NB ideal |
| Feature dependen kuat | Hindari NB, pakai Logistic Regression/DT |