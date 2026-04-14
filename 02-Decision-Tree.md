# Decision Tree (Pohon Keputusan) - Penjelasan Lengkap

## Daftar Isi
1. [Konsep Dasar](#konsep-dasar)
2. [Cara Kerja Detail (Step-by-Step)](#cara-kerja-detail)
3. [Kode Python dari Nol (Tanpa Library)](#kode-python-dari-nol)
4. [Contoh Manual: 5 Data Point](#contoh-manual-5-data)
5. [Contoh Manual: 10 Data Point](#contoh-manual-10-data)
6. [Contoh Skala Besar: 10000+ Data dengan Scikit-Learn](#contoh-skala-besar)
7. [Alur Kerja Program Lengkap](#alur-kerja-program)
8. [Semua Operasi Matematika yang Dilakukan](#operasi-matematika)
9. [Pruning dan Mencegah Overfitting](#pruning)
10. [Pro dan Kontra](#pro-dan-kontra)

---

## Konsep Dasar

Decision Tree adalah algoritma **supervised learning** yang membangun model prediktif berbentuk pohon keputusan.

Analogi: Seperti permainan "20 Pertanyaan" - kamu bertanya ya/tidak secara berurutan sampai menemukan jawaban.

### Terminologi:

```
            [Umur > 30?]           ← Root Node (Akar)
           /            \
         Ya              Tidak     ← Branch (Cabang)
          |                |
   [Gaji > 5jt?]      [Menikah?]  ← Internal Node (Node keputusan)
    /        \         /       \
  Ya        Tidak    Ya       Tidak  ← Branch
   |          |       |        |
[SETUJUI]  [TOLAK] [SETUJUI] [TOLAK] ← Leaf Node (Daun = keputusan akhir)
```

| Term | Penjelasan |
|------|-----------|
| **Root Node** | Node paling atas, pertanyaan pertama |
| **Internal Node** | Node di tengah yang mengajukan pertanyaan |
| **Leaf Node** | Node akhir yang memberikan prediksi/kelas |
| **Branch** | Garis penghubung antar node (Ya/Tidak) |
| **Depth** | Kedalaman pohon (jumlah level) |
| **Split** | Proses membagi data berdasarkan kondisi |

---

## Cara Kerja Detail

### Algoritma Utama: Recursive Binary Splitting

```
BUILD_TREE(dataset):
1. IF semua data satu kelas → return Leaf(kelas tersebut)
2. IF dataset kosong → return Leaf(kelas mayoritas parent)
3. IF semua fitur sama → return Leaf(kelas mayoritas)
4. Cari FITUR dan THRESHOLD TERBAIK untuk split
5. Split data menjadi left_data dan right_data
6. left_child  = BUILD_TREE(left_data)
7. right_child = BUILD_TREE(right_data)
8. Return Node(fitur, threshold, left_child, right_child)
```

### Bagaimana Menemukan Split Terbaik?

Menggunakan **Impurity Measures** - mengukur seberapa "campur" data di suatu node.

#### 1. Gini Impurity (digunakan oleh CART/scikit-learn)

```
Gini = 1 - Σ (pi²)
```
di mana `pi` = proporsi kelas i dalam node

- Gini = 0 → Node murni (semua data satu kelas)
- Gini mendekati 0.5 → Node sangat campur (biner)

**Gini Gain** = Gini(parent) - [weighted_avg_Gini(children)]

#### 2. Entropy (digunakan oleh ID3/C4.5)

```
Entropy = -Σ (pi × log₂(pi))
```

- Entropy = 0 → Murni
- Entropy = 1 → Campur sempurna (biner)

**Information Gain** = Entropy(parent) - [weighted_avg_Entropy(children)]

#### 3. Perhitungan Split:

Untuk setiap kandidat split (fitur + threshold):
1. Bagi data menjadi left dan right
2. Hitung impurity left dan right
3. Hitung weighted average impurity
4. Hitung gain = impurity(parent) - weighted_impurity(children)
5. Pilih split dengan gain TERBESAR

---

## Kode Python dari Nol

```python
from collections import Counter
import math

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.root = self._build_tree(X, y, depth=0)

    def _gini(self, y):
        counts = Counter(y)
        n = len(y)
        gini = 1.0
        for count in counts.values():
            p = count / n
            gini -= p ** 2
        return gini

    def _entropy(self, y):
        counts = Counter(y)
        n = len(y)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _impurity(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)

    def _information_gain(self, y, y_left, y_right):
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        parent_impurity = self._impurity(y)
        weighted_child_impurity = (n_left / n) * self._impurity(y_left) + \
                                  (n_right / n) * self._impurity(y_right)
        return parent_impurity - weighted_child_impurity

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n = len(y)

        for feature_idx in range(self.n_features):
            thresholds = sorted(set(row[feature_idx] for row in X))

            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2

                y_left = [y[j] for j in range(n) if X[j][feature_idx] <= threshold]
                y_right = [y[j] for j in range(n) if X[j][feature_idx] > threshold]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _most_common(self, y):
        return Counter(y).most_common(1)[0][0]

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        n_classes = len(set(y))

        # Stopping conditions
        if n_classes == 1:
            return DecisionTreeNode(value=y[0])
        if n_samples < self.min_samples_split:
            return DecisionTreeNode(value=self._most_common(y))
        if depth >= self.max_depth:
            return DecisionTreeNode(value=self._most_common(y))

        # Find best split
        feature, threshold, gain = self._best_split(X, y)

        if gain <= 0:
            return DecisionTreeNode(value=self._most_common(y))

        # Split data
        left_indices = [i for i in range(n_samples) if X[i][feature] <= threshold]
        right_indices = [i for i in range(n_samples) if X[i][feature] > threshold]

        X_left = [X[i] for i in left_indices]
        y_left = [y[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_right = [y[i] for i in right_indices]

        # Recursive build
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return DecisionTreeNode(feature=feature, threshold=threshold,
                                 left=left_child, right=right_child)

    def predict(self, X):
        if isinstance(X[0], (int, float)):
            X = [X]
        return [self._traverse(x, self.root) for x in X]

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def print_tree(self, node=None, depth=0, prefix="Root"):
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}{prefix} → LEAF: Class={node.value}")
        else:
            print(f"{indent}{prefix} → Feature[{node.feature}] <= {node.threshold:.3f}?")
            self.print_tree(node.left, depth + 1, "Left (Yes)")
            self.print_tree(node.right, depth + 1, "Right (No)")

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = sum(1 for p, t in zip(predictions, y_test) if p == t)
        return correct / len(y_test)
```

---

## Contoh Manual: 5 Data Point

### Dataset: Prediksi Apakah Seseorang Akan Membeli Komputer

| # | Umur | Pendapatan | Beli? |
|---|------|-----------|-------|
| 1 | 25   | 30        | Tidak |
| 2 | 35   | 60        | Ya    |
| 3 | 45   | 80        | Ya    |
| 4 | 20   | 20        | Tidak |
| 5 | 50   | 70        | Ya    |

### Langkah 1: Hitung Impurity Root Node

**Data root:** [Tidak, Ya, Ya, Tidak, Ya] → 2 Tidak, 3 Ya

```
Gini(root) = 1 - (2/5)² - (3/5)²
           = 1 - 0.16 - 0.36
           = 0.48
```

### Langkah 2: Evaluasi Semua Possible Splits

#### Split pada Fitur "Umur":

**Coba threshold = 22.5** (rata-rata antara 20 dan 25):
```
Left  (Umur ≤ 22.5): Data #4 → [Tidak] → Gini = 0
Right (Umur > 22.5):  Data #1,2,3,5 → [Tidak, Ya, Ya, Ya] → Gini = 1 - (1/4)² - (3/4)² = 1 - 0.0625 - 0.5625 = 0.375
Weighted Gini = (1/5)(0) + (4/5)(0.375) = 0.3
Gain = 0.48 - 0.3 = 0.18
```

**Coba threshold = 30** (rata-rata antara 25 dan 35):
```
Left  (Umur ≤ 30):  Data #1,4 → [Tidak, Tidak] → Gini = 0
Right (Umur > 30):  Data #2,3,5 → [Ya, Ya, Ya] → Gini = 0
Weighted Gini = (2/5)(0) + (3/5)(0) = 0
Gain = 0.48 - 0 = 0.48 ← SEMPURNA!
```

**Coba threshold = 40** (rata-rata antara 35 dan 45):
```
Left  (Umur ≤ 40):  Data #1,2,4 → [Tidak, Ya, Tidak] → Gini = 1 - (2/3)² - (1/3)² = 0.444
Right (Umur > 40):  Data #3,5 → [Ya, Ya] → Gini = 0
Weighted Gini = (3/5)(0.444) + (2/5)(0) = 0.267
Gain = 0.48 - 0.267 = 0.213
```

**Coba threshold = 47.5** (rata-rata antara 45 dan 50):
```
Left  (Umur ≤ 47.5):  Data #1,2,3,4 → [Tidak, Ya, Ya, Tidak] → Gini = 0.5
Right (Umur > 47.5):  Data #5 → [Ya] → Gini = 0
Weighted Gini = (4/5)(0.5) + (1/5)(0) = 0.4
Gain = 0.48 - 0.4 = 0.08
```

#### Split pada Fitur "Pendapatan":

**Coba threshold = 25** (rata-rata antara 20 dan 30):
```
Left  (Pendapatan ≤ 25):  Data #4 → [Tidak] → Gini = 0
Right (Pendapatan > 25):  Data #1,2,3,5 → [Tidak, Ya, Ya, Ya] → Gini = 0.375
Weighted Gini = (1/5)(0) + (4/5)(0.375) = 0.3
Gain = 0.48 - 0.3 = 0.18
```

**Coba threshold = 45** (rata-rata antara 30 dan 60):
```
Left  (Pendapatan ≤ 45):  Data #1,4 → [Tidak, Tidak] → Gini = 0
Right (Pendapatan > 45):  Data #2,3,5 → [Ya, Ya, Ya] → Gini = 0
Weighted Gini = (2/5)(0) + (3/5)(0) = 0
Gain = 0.48 - 0 = 0.48 ← SEMPURNA JUGA!
```

**Coba threshold = 65** (rata-rata antara 60 dan 70):
```
Left  (Pendapatan ≤ 65):  Data #1,2,4 → [Tidak, Ya, Tidak] → Gini = 0.444
Right (Pendapatan > 65):  Data #3,5 → [Ya, Ya] → Gini = 0
Weighted Gini = (3/5)(0.444) + (2/5)(0) = 0.267
Gain = 0.48 - 0.267 = 0.213
```

**Coba threshold = 75** (rata-rata antara 70 dan 80):
```
Left  (Pendapatan ≤ 75):  Data #1,2,4,5 → [Tidak, Ya, Tidak, Ya] → Gini = 0.5
Right (Pendapatan > 75):  Data #3 → [Ya] → Gini = 0
Weighted Gini = (4/5)(0.5) + (1/5)(0) = 0.4
Gain = 0.48 - 0.4 = 0.08
```

### Langkah 3: Pilih Split Terbaik

| Split | Gain |
|-------|------|
| Umur ≤ 30   | **0.48** |
| Umur ≤ 22.5 | 0.18 |
| Umur ≤ 40   | 0.213 |
| Umur ≤ 47.5 | 0.08 |
| Pendapatan ≤ 45 | **0.48** |
| Pendapatan ≤ 25 | 0.18 |
| Pendapatan ≤ 65 | 0.213 |
| Pendapatan ≤ 75 | 0.08 |

**Hasil:** Umur ≤ 30 **ATAU** Pendapatan ≤ 45, keduanya perfect split (Gain = 0.48).

Pilih salah satu (biasanya yang pertama ditemukan): **Umur ≤ 30**

### Pohon Hasil:

```
[Umur ≤ 30?]
├── Ya → LEAF: Tidak (Data #1,4)
└── Tidak → LEAF: Ya (Data #2,3,5)
```

Kedua leaf sudah murni (Gini=0), jadi pohon selesai!

### Prediksi Data Baru: Umur=28, Pendapatan=40

```
Root: Umur ≤ 30? → 28 ≤ 30 → Ya → LEAF: Tidak
```

**Hasil: Tidak membeli**

### Kode untuk Contoh Ini:

```python
dt = DecisionTree(max_depth=5, criterion='gini')
X = [[25, 30], [35, 60], [45, 80], [20, 20], [50, 70]]
y = ['Tidak', 'Ya', 'Ya', 'Tidak', 'Ya']
dt.fit(X, y)
dt.print_tree()

result = dt.predict([[28, 40]])
print(f"\nPrediksi (Umur=28, Pendapatan=40): {result}")
```

---

## Contoh Manual: 10 Data Point

### Dataset: Prediksi Kesukaan Olahraga

| # | Suhu (°C) | Kelembapan (%) | Angin (km/h) | Olahraga? |
|---|-----------|----------------|---------------|-----------|
| 1  | 28        | 70             | 5             | Ya        |
| 2  | 30        | 85             | 10            | Tidak     |
| 3  | 25        | 65             | 3             | Ya        |
| 4  | 32        | 90             | 15            | Tidak     |
| 5  | 22        | 60             | 8             | Ya        |
| 6  | 20        | 75             | 12            | Tidak     |
| 7  | 27        | 68             | 2             | Ya        |
| 8  | 35        | 95             | 20            | Tidak     |
| 9  | 24        | 72             | 6             | Ya        |
| 10 | 29        | 88             | 14            | Tidak     |

### Impurity Root:

```
5 Ya, 5 Tidak
Gini = 1 - (0.5)² - (0.5)² = 0.5
Entropy = -0.5×log2(0.5) - 0.5×log2(0.5) = 1.0
```

### Evaluasi Best Splits (disederhanakan - yang signifikan):

Setelah mengevaluasi semua threshold, misal split terbaik adalah **Suhu ≤ 26**:

```
Left  (Suhu ≤ 26): Data #3,5,6,9 → [Ya, Ya, Tidak, Ya] = 3 Ya, 1 Tidak
  Gini = 1 - (3/4)² - (1/4)² = 0.375

Right (Suhu > 26): Data #1,2,4,7,8,10 → [Ya, Tidak, Tidak, Ya, Tidak, Tidak] = 2 Ya, 4 Tidak
  Gini = 1 - (2/6)² - (4/6)² = 0.444

Weighted Gini = (4/10)(0.375) + (6/10)(0.444) = 0.15 + 0.267 = 0.417
Gain = 0.5 - 0.417 = 0.083
```

Node kiri masih belum murni, jadi kita split lagi. Coba **Kelembapan ≤ 68**:

```
Data kiri: #3(65), #5(60), #6(75), #9(72)
Left-left  (Kep ≤ 68): #3,5 → [Ya, Ya] → Gini = 0
Left-right (Kep > 68): #6,9 → [Tidak, Ya] → Gini = 0.5

Gain = 0.375 - (2/4)(0) - (2/4)(0.5) = 0.375 - 0.25 = 0.125
```

Sekarang pohon:

```
[Suhu ≤ 26?]
├── Ya → [Kelembapan ≤ 68?]
│        ├── Ya → LEAF: Ya (#3,5)
│        └── Tidak → [Angin ≤ 9?]
│                 ├── Ya → LEAF: Ya (#9)
│                 └── Tidak → LEAF: Tidak (#6)
└── Tidak → [Suhu ≤ 31?]
             ├── Ya → [Kelembapan ≤ 75?]
             │        ├── Ya → LEAF: Ya (#1,7)
             │        └── Tidak → LEAF: Tidak (#2)
             └── Tidak → LEAF: Tidak (#4,8,10)
```

### Kode:

```python
dt = DecisionTree(max_depth=5, criterion='gini')
X = [
    [28, 70, 5],  [30, 85, 10], [25, 65, 3],  [32, 90, 15],
    [22, 60, 8],  [20, 75, 12], [27, 68, 2],  [35, 95, 20],
    [24, 72, 6],  [29, 88, 14]
]
y = ['Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak']
dt.fit(X, y)
dt.print_tree()

test = [[26, 66, 5]]
print(f"\nPrediksi (Suhu=26, Kep=66, Angin=5): {dt.predict(test)}")
```

---

## Contoh Skala Besar: 10000+ Data dengan Scikit-Learn

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import make_classification, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# ============================================
# CONTOH 1: Dataset Sintetis Besar
# ============================================
print("=" * 60)
print("CONTOH 1: Dataset Sintetis 50,000 Data")
print("=" * 60)

X, y = make_classification(
    n_samples=50000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree TIDAK memerlukan feature scaling!
# (kengan K-NN, scaling penting; dengan DT, tidak)
dt = DecisionTreeClassifier(max_depth=15, criterion='gini', random_state=42)

start = time.time()
dt.fit(X_train, y_train)
train_time = time.time() - start
print(f"Waktu training: {train_time:.4f} detik")

start = time.time()
y_pred = dt.predict(X_test)
pred_time = time.time() - start
print(f"Waktu prediksi ({len(X_test)} data): {pred_time:.4f} detik")

print(f"Train accuracy: {dt.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {dt.score(X_test, y_test):.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print(f"\nKedalaman pohon: {dt.get_depth()}")
print(f"Jumlah leaf: {dt.get_n_leaves()}")
print(f"Fitur paling penting: Feature {np.argmax(dt.feature_importances_)} (importance={max(dt.feature_importances_):.4f})")

# ============================================
# CONTOH 2: Real Dataset (Wine)
# ============================================
print("\n" + "=" * 60)
print("CONTOH 2: Wine Dataset (Real Data)")
print("=" * 60)

wine = load_wine()
X_w, y_w = wine.data, wine.target
X_tr, X_te, y_tr, y_te = train_test_split(X_w, y_w, test_size=0.3, random_state=42)

dt_wine = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_wine.fit(X_tr, y_tr)

print(f"Test accuracy: {dt_wine.score(X_te, y_te):.4f}")
print(f"Tree depth: {dt_wine.get_depth()}")
print(f"Number of leaves: {dt_wine.get_n_leaves()}")

# Visualisasi text tree
print("\n=== Struktur Pohon ===")
print(export_text(dt_wine, feature_names=wine.feature_names))

# Feature importance
print("\n=== Feature Importance ===")
for name, imp in sorted(zip(wine.feature_names, dt_wine.feature_importances_),
                         key=lambda x: -x[1]):
    print(f"  {name:30s}: {imp:.4f}")

# ============================================
# CONTOH 3: Regression Tree
# ============================================
print("\n" + "=" * 60)
print("CONTOH 3: Decision Tree Regression")
print("=" * 60)

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

X_reg, y_reg = make_regression(n_samples=10000, n_features=10, noise=20, random_state=42)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, y_reg, test_size=0.2)

dt_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_reg.fit(Xr_tr, yr_tr)
yr_pred = dt_reg.predict(Xr_te)

print(f"R² Score: {r2_score(yr_te, yr_pred):.4f}")
print(f"MSE: {mean_squared_error(yr_te, yr_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(yr_te, yr_pred)):.4f}")

# ============================================
# EKSPERIMEN: Max Depth vs Overfitting
# ============================================
print("\n" + "=" * 60)
print("EKSPERIMEN: Pengaruh Max Depth")
print("=" * 60)

X_d, y_d = make_classification(n_samples=5000, n_features=10, n_classes=3, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X_d, y_d, test_size=0.3)

for depth in [1, 3, 5, 10, 15, 20, None]:
    dt_exp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_exp.fit(Xtr, ytr)
    train_acc = dt_exp.score(Xtr, ytr)
    test_acc = dt_exp.score(Xte, yte)
    leaves = dt_exp.get_n_leaves()
    print(f"Depth={str(depth):5s} | Train={train_acc:.4f} | Test={test_acc:.4f} | Leaves={leaves}")

# Cross-validation
print("\n=== Cross-Validation ===")
dt_cv = DecisionTreeClassifier(max_depth=10, random_state=42)
scores = cross_val_score(dt_cv, Xtr, ytr, cv=5)
print(f"CV Scores: {scores}")
print(f"Mean CV: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

---

## Alur Kerja Program

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ALUR KERJA DECISION TREE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. INISIALISASI                                                            │
│     ├── Set max_depth, min_samples_split, criterion (gini/entropy)          │
│     └── Siapkan root = None                                                 │
│                                                                              │
│  2. FIT (Training Phase)                                                    │
│     │                                                                        │
│     └── BUILD_TREE(X, y, depth=0)                                           │
│         │                                                                    │
│         ├── CHECK STOP CONDITIONS:                                          │
│         │   ├── Semua y sama? → Return Leaf(y[0])                            │
│         │   ├── n_samples < min_samples? → Return Leaf(most_common(y))       │
│         │   └── depth >= max_depth? → Return Leaf(most_common(y))            │
│         │                                                                    │
│         ├── SEARCH BEST SPLIT:                                               │
│         │   FOR each feature:                                                │
│         │     FOR each candidate threshold:                                  │
│         │       ├── Split data: left (≤ threshold) and right (> threshold)   │
│         │       ├── Calculate impurity_left, impurity_right                  │
│         │       ├── Calculate weighted_impurity                              │
│         │       ├── Calculate gain = parent_impurity - weighted_impurity      │
│         │       └── If gain > best_gain → update best_split                  │
│         │                                                                    │
│         ├── If best_gain ≤ 0 → Return Leaf(most_common(y))                  │
│         │                                                                    │
│         ├── SPLIT DATA by best feature + threshold                           │
│         │                                                                    │
│         ├── RECURSIVE BUILD:                                                 │
│         │   left_child  = BUILD_TREE(X_left,  y_left,  depth+1)             │
│         │   right_child = BUILD_TREE(X_right, y_right, depth+1)              │
│         │                                                                    │
│         └── Return Node(feature, threshold, left, right)                    │
│                                                                              │
│  3. PREDICT (Inference Phase)                                               │
│     │                                                                        │
│     └── For each x_new:                                                      │
│         └── TRAVERSE(x, node):                                               │
│             ├── If leaf → return node.value                                  │
│             ├── If x[feature] ≤ threshold → TRAVERSE(x, node.left)          │
│             └── If x[feature] > threshold → TRAVERSE(x, node.right)         │
│                                                                              │
│  4. EVALUASI                                                                │
│     ├── Accuracy, Precision, Recall, F1                                     │
│     ├── Confusion Matrix                                                     │
│     ├── Feature Importance                                                   │
│     └── Cross-Validation                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Operasi Matematika

### Semua Operasi dalam Decision Tree:

| Fase | Operasi | Detail |
|------|---------|--------|
| **Split Search** | | |
| | Sorting | N × F kali (per fitur, untuk cari threshold) |
| | Pengurangan | N per threshold |
| | Perbandingan | N per threshold (classifying left/right) |
| | Pengkuadratan | C per node (untuk Gini: Σpi²) |
| | Penjumlahan | C per node |
| | Pembagian | C per node |
| **Tree Building** | | |
| | Rekursion | Depth levels |
| | Split evaluation | O(N × F × T) per node (T = jumlah threshold) |
| **Prediction** | | |
| | Perbandingan | D per prediksi (D = depth) |
| | Array indexing | D per prediksi |

### Perhitungan Detail Gini:

```python
# Contoh: node dengan data [Ya, Tidak, Ya, Ya, Tidak]
# Ya = 3, Tidak = 2, Total = 5

# Step 1: Hitung proporsi setiap kelas
p_ya     = 3 / 5 = 0.6
p_tidak  = 2 / 5 = 0.4

# Step 2: Kuadratkan proporsi
p_ya²    = 0.6² = 0.36
p_tidak² = 0.4² = 0.16

# Step 3: Jumlahkan kuadrat
sum_sq   = 0.36 + 0.16 = 0.52

# Step 4: Gini = 1 - sum
gini     = 1 - 0.52 = 0.48

# Interpretasi:
# Gini = 0  → Perfect (semua satu kelas)
# Gini = 0.5 → Maximum impurity (biner, seimbang)
# Gini = 0.48 → Agak impure
```

### Perhitungan Detail Information Gain:

```python
# Parent: [Ya, Tidak, Ya, Ya, Tidak] → Gini = 0.48
# Split threshold: Umur ≤ 30
# Left:  [Tidak, Tidak] → Gini = 0
# Right: [Ya, Ya, Ya] → Gini = 0

# Weighted Gini
n_parent = 5
n_left = 2
n_right = 3

weighted_gini = (n_left/n_parent) * gini_left + (n_right/n_parent) * gini_right
              = (2/5) * 0 + (3/5) * 0
              = 0

# Information Gain
gain = gini_parent - weighted_gini
     = 0.48 - 0
     = 0.48  # Perfect split!
```

---

## Pruning dan Mencegah Overfitting

### Masalah Overfitting pada Decision Tree:

Decision Tree yang tidak dibatasi akan **menghafal** data training → 100% train accuracy tapi jelek di test.

### Teknik Pruning:

#### 1. Pre-Pruning (Early Stopping):
```python
from sklearn.tree import DecisionTreeClassifier

# Batasi kedalaman
dt = DecisionTreeClassifier(max_depth=5)

# Minimum samples untuk split
dt = DecisionTreeClassifier(min_samples_split=10)

# Minimum samples di leaf
dt = DecisionTreeClassifier(min_samples_leaf=5)

# Maximum leaf nodes
dt = DecisionTreeClassifier(max_leaf_nodes=20)

# Minimum impurity decrease
dt = DecisionTreeClassifier(min_impurity_decrease=0.01)

# Gabungan
dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_leaf_nodes=50,
    min_impurity_decrease=0.001
)
```

#### 2. Post-Pruning (Cost Complexity Pruning):
```python
from sklearn.tree import DecisionTreeClassifier

# Hitung pruning path
dt_full = DecisionTreeClassifier(random_state=42)
path = dt_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Train untuk setiap alpha
train_scores = []
test_scores = []
for alpha in ccp_alphas:
    dt_pruned = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    dt_pruned.fit(X_train, y_train)
    train_scores.append(dt_pruned.score(X_train, y_train))
    test_scores.append(dt_pruned.score(X_test, y_test))

# Cari alpha optimal
best_idx = np.argmax(test_scores)
best_alpha = ccp_alphas[best_idx]
print(f"Best alpha: {best_alpha:.6f}")
print(f"Best test accuracy: {test_scores[best_idx]:.4f}")

# Train final model
dt_final = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
dt_final.fit(X_train, y_train)
```

---

## Pro dan Kontra

### Pro:
1. **Interpretability** - Mudah dipahami dan divisualisasikan
2. **Tanpa scaling** - Tidak perlu normalisasi/standardisasi
3. **Handle mixed data** - Bisa numerik dan kategorikal
4. **Non-linear** - Bisa menangkap relasi non-linear
5. **Feature importance** - Memberi ranking fitur penting
6. **Sedikit preprocessing** - Handle missing (dengan modifikasi)

### Kontra:
1. **Overfitting** - Sangat rentan jika tidak dibatasi
2. **Instability** - Perubahan kecil pada data → pohon sangat berbeda
3. **Bias** - Cenderung memilih fitur dengan banyak level
4. **Non-optimal** - Greedy algorithm, tidak menjamin pohon global optimum
5. **Extrapolation** - Regression tree tidak bisa prediksi di luar range training

### Solusi: Ensemble Methods

| Metode | Cara | Keunggulan |
|--------|------|------------|
| **Random Forest** | Banyak DT dengan random subset | Kurangi variance, lebih stabil |
| **Gradient Boosting** | DT bertahap, koreksi error sebelumnya | Akurasi tinggi |
| **AdaBoost** | Weighted sampling, fokus pada error | Sederhana, efektif |
| **XGBoost/LightGBM** | Optimized gradient boosting | State-of-the-art |