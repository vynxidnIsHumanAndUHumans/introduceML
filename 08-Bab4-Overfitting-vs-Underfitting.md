# Bab 4: Overfitting vs Underfitting — Musuh Besar Machine Learning

---

## Cerita Pembuka: Siswa Penghafal vs Siswa Pemaham

Ada dua siswa yang akan menghadapi ujian matematika.

**Siswa A — Si Penghafal:**
- Menghafal setiap soal latihan dan jawabannya
- Saat ujian dengan soal yang SAMA PERSIS → skor 100
- Saat ujian dengan soal BARU → skor 30
- Dia tidak memahami konsep — dia hanya menghafal

**Siswa B — Si Pemaham:**
- Belajar konsep dasar dan prinsip
- Saat ujian dengan soal yang sama → skor 85 (tidak sempurna)
- Saat ujian dengan soal baru → skor 80 (konsisten)
- Dia memahami pola — dia bisa generalisasi

**Siswa C — Si Malas:**
- Tidak belajar sama sekali
- Saat ujian apa saja → skor 20
- Dia tidak punya pengetahuan sama sekali

Dalam Machine Learning:

| Siswa | Istilah ML | Training Score | Test Score | Penjelasan |
|-------|-----------|---------------|------------|------------|
| A | **Overfitting** | Tinggi (100) | Rendah (30) | Model terlalu kompleks, menghafal noise |
| B | **Good Fit** | Tinggi (85) | Tinggi (80) | Model seimbang, generalisasi baik |
| C | **Underfitting** | Rendah (20) | Rendah (20) | Model terlalu sederhana, tidak belajar |

Ini adalah **masalah paling fundamental** di Machine Learning. Setiap model berada di suatu titik antara underfitting dan overfitting. Tugas kita adalah menemukan titik manis di tengah.

---

## 4.1 Apa Itu Overfitting?

### Definisi Formal

**Overfitting** terjadi ketika model mempelajari pola **dan noise** dalam data training, sehingga performa di data training sangat baik tapi performa di data baru (test) buruk.

Model overfit "menghafal" data training — termasuk semua keanehan, noise, dan fluktuasi acak yang tidak muncul kembali di data baru.

### Visualisasi Overfitting

```
Underfitting          Good Fit              Overfitting
                    
  ╱╲                  ╱───╲                 ╱╲╱╲╱╲╱╲
 ╱  ╲                ╱     ╲               ╱╲╱╲╱╲╱╲╱╲
╱    ╲              ╱       ╲             ╱╲╱╲╱╲╱╲╱╲╱╲

Model terlalu        Model menangkap       Model mengikuti
sederhana            pola utama           SETIAP titik data
(garis lurus)        (kurva smooth)       (kurva bergerigi)

Training error:      Training error:       Training error:
TINGGI               SEDANG               RENDAH (mendekati 0)

Test error:          Test error:           Test error:
TINGGI               RENDAH               TINGGI (lebih tinggi dari training)
```

### Contoh Konkret: Polynomial Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 30)).reshape(-1, 1)
y_true = 0.5 * X.ravel() ** 2 - X.ravel() + 2
y = y_true + np.random.normal(0, 1.5, len(X))  # Tambahkan noise

X_test = np.linspace(-3, 3, 200).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (degree, title) in enumerate([(1, "Underfitting (degree=1)"),
                                      (3, "Good Fit (degree=3)"),
                                      (15, "Overfitting (degree=15)")]):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)

    y_train_pred = model.predict(X)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y, y_train_pred)

    axes[i].scatter(X, y, c='blue', s=30, label='Data training')
    axes[i].plot(X_test, y_test_pred, 'r-', linewidth=2, label=f'Model (degree={degree})')
    axes[i].plot(X_test, 0.5 * X_test.ravel() ** 2 - X_test.ravel() + 2,
                  'g--', linewidth=1, label='Fungsi sebenarnya')
    axes[i].set_title(f'{title}\nTraining MSE: {train_mse:.2f}')
    axes[i].legend()
    axes[i].set_ylim(-5, 15)

plt.tight_layout()
plt.savefig('overfitting_visual.png', dpi=150, bbox_inches='tight')
plt.show()
```

Penjelasan:
- **Degree 1 (garis lurus)**: Model terlalu sederhana. Tidak bisa menangkap pola kuadratik. Underfitting.
- **Degree 3 (kurva smooth)**: Model menangkap pola dengan baik. Good fit.
- **Degree 15 (kurva bergerigi)**: Model mengikuti setiap titik data termasuk noise. Overfitting.

### Penyebab Overfitting

| Penyebab | Penjelasan | Contoh |
|----------|-----------|--------|
| **Model terlalu kompleks** | Parameter terlalu banyak untuk data yang ada | Neural network 10 layer untuk 100 data |
| **Data training terlalu sedikit** | Tidak cukup variasi untuk model belajar pola yang benar | 20 data point, model polynomial degree 10 |
| **Training terlalu lama** | Model mulai menghafal noise | Neural network 1000 epoch |
| **Fitur terlalu banyak** | Curse of dimensionality | 1000 fitur, 100 data |
| **Noise di data** | Model belajar error pengukuran | Data yang salah/kotor |
| **Tidak ada regularization** | Model tidak dihukum karena terlalu kompleks | Decision tree tanpa max_depth |

### Gambaran Intuitif: Mengapa Overfitting Terjadi?

Bayangkan kamu diminta menggambar garis yang menghubungkan titik-titik ini:

```
Data training: ● ●   ●  ●  ●   ●  ●
```

Model sederhana menggambar garis lurus → meleset di beberapa titik (underfitting).

Model yang tepat menggambar kurva halus → melewati dekat sebagian besar titik (good fit).

Model overfit menggambar kurva yang melewati SETIAP titik tepat — termasuk titik yang sebenarnya adalah noise:

```
Model overfit: ●─╲╱─●─╲╱───●╲╱──●──╲╱─●──╲╱●
```

Masalahnya: titik data baru tidak akan berada di kurva bergerigi ini. Model overfit TIDAK BISA generalisasi.

---

## 4.2 Apa Itu Underfitting?

### Definisi Formal

**Underfitting** terjadi ketika model terlalu sederhana untuk menangkap pola dalam data. Model tidak belajar bahkan dari data training.

### Penyebab Underfitting

| Penyebab | Penjelasan | Contoh |
|----------|-----------|--------|
| **Model terlalu sederhana** | Parameter tidak cukup | Linear regression untuk data non-linear |
| **Feature yang kurang** | Informasi penting tidak disertakan | Prediksi harga rumah hanya dari luas tanah |
| **Regularisasi terlalu kuat** | Model terlalu ditekan | L1 penalty terlalu besar |
| **Training tidak cukup** | Epoch terlalu sedikit | Neural network hanya training 10 epoch |
| **Data kurang informatif** | Fitur tidak berkorelasi dengan target | Prediksi cuaca dari nomor HP |

### Contoh Underfitting

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Data dengan pola kuadratik
X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_train = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # y = x²

# Model underfitting: Linear Regression (hanya garis lurus)
model_underfit = LinearRegression()
model_underfit.fit(X_train, y_train)
print(f"Linear Regression R² (train): {model_underfit.score(X_train, y_train):.4f}")
# Output: ~0.94  (masih ada error signifikan karena data non-linear)

# Model good fit: Polynomial Regression degree 2
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
model_good = LinearRegression()
model_good.fit(X_poly, y_train)
print(f"Polynomial(2) R² (train): {model_good.score(X_poly, y_train):.4f}")
# Output: ~1.00  (hampir sempurna, pola degree 2 cocok dengan data x²)
```

---

## 4.3 Bias-Variance Tradeoff — Teori di Balik Overfitting dan Underfitting

### Konsep Intuitif

Setiap model ML punya dua jenis error:

1. **Bias** — Error karena model terlalu sederhana (underfitting)
2. **Variance** — Error karena model terlalu sensitif terhadap data training (overfitting)

Ada **tradeoff** antara keduanya: kalau kurangi bias, variance naik, dan sebaliknya.

### Bias — Kecenderungan Bukaan

> Bias mengukur seberapa jauh rata-rata prediksi model dari nilai sebenarnya.

Model dengan **bias tinggi** cenderung meleset secara sistematis — selalu "meleset ke arah yang sama".

Analogi: Busur panah yang peruntungan selalu menyimpang ke kanan. Tembakannya konsisten (low variance), tapi selalu meleset ke kanan (high bias).

```
HIGH BIAS (Underfitting):          LOW BIAS (Good fit):

    ○                                ●
      ○                            ●  ● ●
        ○                          ● ● ● ● ●
                                    ● ● ●
          ○                          ● ●

  (Semua meleset ke kanan)      (Tepat di target)
```

### Variance — Sensitivitas terhadap Data

> Variance mengukur seberapa banyak prediksi berubah kalau data training diganti.

Model dengan **variance tinggi** sangat sensitif terhadap data training — ganti sedikit saja data training, prediksi berubah drastis.

Analogi: Busur panah yang peruntungan tidak konsisten — kadang ke kiri, kadang ke kanan, kadang tepat. Rata-rata dekat target (low bias), tapi menyebar (high variance).

```
HIGH VARIANCE (Overfitting):       LOW VARIANCE (Good fit):

    ●                             ●
  ●       ●                       ● ●
      ●                       ● ● ● ● ●
  ●   ●   ●                       ● ●
●         ●   ●                      ●

  (Menyebar, tidak konsisten)    (Konsisten dan akurat)
```

### Tradeoff Visual

```
Error
  ^
  |   Total Error
  |    ╱
  |   ╱
  |  ╱  ← Titik optimum di sini
  | ╱
  |╲____________________________╱
  | ╲ Bias↓                  ╱ ← Variance↑
  |   ╲                   ╱
  |     ╲             ╱
  |       ╲       ╱
  |         ╲  ╱
  +─────────────────────────────→ Model Complexity
  Simple                Complex
```

| Kompleksitas Model | Bias | Variance | Penjelasan |
|-------------------|------|----------|------------|
| Sederhana (Linear) | Tinggi | Rendah | Underfitting |
| Sedang (Polynomial d=3) | Sedang | Sedang | Good fit |
| Kompleks (Polynomial d=20) | Rendah | Tinggi | Overfitting |

### Dekomposisi Bias-Variance

Secara matematis, error total bisa didekomposisi:

```
Total Error = Bias² + Variance + Irreducible Error
```

- **Bias²**: Error dari asumsi model yang terlalu sederhana
- **Variance**: Error dari sensitivitas model terhadap variasi data training
- **Irreducible Error**: Error yang tidak bisa dihilangkan (noise di data)

```python
from sklearn.model_selection import cross_val_score
import numpy as np

def compute_bias_variance(model, X, y, n_trials=100, test_size=0.3):
    """Estimasi bias dan variance secara empiris"""
    predictions = []
    
    for _ in range(n_trials):
        # Sample random training set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size
        )
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        pred = model_clone.predict(X_test)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Bias: rata-rata error antara prediksi dan nilai sebenarnya
    avg_predictions = predictions.mean(axis=0)
    bias = np.mean((avg_predictions - y_test) ** 2)
    
    # Variance: seberapa bervariasi prediksi antar trial
    variance = np.mean(predictions.var(axis=0))
    
    return bias, variance

# Contoh penggunaan
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor

X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)

for depth in [1, 3, 5, 10, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    bias, variance = compute_bias_variance(model, X, y)
    print(f"Depth={str(depth):5s} | Bias²={bias:.2f} | Variance={variance:.2f} | Total={bias+variance:.2f}")
```

Output kira-kira:
```
Depth=1     | Bias²=850.32 | Variance=2.15  | Total=852.47  ← Underfitting
Depth=3     | Bias²=120.50  | Variance=45.30 | Total=165.80  
Depth=5     | Bias²=50.20   | Variance=80.45 | Total=130.65  ← Good fit
Depth=10    | Bias²=15.80   | Variance=150.60| Total=166.40  
Depth=None  | Bias²=8.50    | Variance=280.30| Total=288.80  ← Overfitting
```

Perhatikan: Semakin kompleks model (depth lebih besar), bias turun tapi variance naik. Titik optimum ada di sekitar depth=5.

---

## 4.4 Mendeteksi Overfitting dan Underfitting

### Metode 1: Learning Curves

Learning curve menunjukkan performa model di training dan validation set seiring bertambahnya data training.

```python
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Model yang overfit
model_overfit = SVC(kernel='rbf', gamma=10, C=1000)
train_sizes, train_scores, val_scores = learning_curve(
    model_overfit, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

print("LEARNING CURVE - Model Overfit (SVC gamma=10):")
print(f"Training score akhir:  {train_scores.mean(axis=1)[-1]:.4f}")
print(f"Validation score akhir: {val_scores.mean(axis=1)[-1]:.4f}")
print(f"Gap: {train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]:.4f}")
# Training hampir sempurna, tapi validation jauh lebih rendah → OVERFITTING

# Model yang underfit
model_underfit = SVC(kernel='linear', C=0.001)
train_sizes, train_scores, val_scores = learning_curve(
    model_underfit, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

print("\nLEARNING CURVE - Model Underfit (SVC C=0.001):")
print(f"Training score akhir:  {train_scores.mean(axis=1)[-1]:.4f}")
print(f"Validation score akhir: {val_scores.mean(axis=1)[-1]:.4f}")
# Training DAN validation jelek → UNDERFITTING

# Model yang good fit
model_good = SVC(kernel='rbf', gamma=0.1, C=10)
train_sizes, train_scores, val_scores = learning_curve(
    model_good, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

print("\nLEARNING CURVE - Model Good Fit (SVC gamma=0.1, C=10):")
print(f"Training score akhir:  {train_scores.mean(axis=1)[-1]:.4f}")
print(f"Validation score akhir: {val_scores.mean(axis=1)[-1]:.4f}")
print(f"Gap: {train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]:.4f}")
# Training dan validation sama-sama bagus, gap kecil → GOOD FIT
```

### Interpretasi Learning Curves

```
OVERFITTING:                        UNDERFITTING:
                                    
Score ^  ╱─── Training              Score ^
      | ╱                          |      ╱─── Training
      |╱                           |    ╱ ╲─── Validation
      |        ╲─── Validation      |  ╱
      |      ╱                      |╱
      |    ╱                        |
      |──╱                         ─────────────────→ Data
      +─────────────────→ Data        Gap KECIL, score RENDAH
      Gap BESAR, training baik,       keduanya jelek
      validation jelek
                                    
GOOD FIT:                           
                                    
Score ^    ╱─── Training             
      |  ╱                         
      |╱─────── Validation           
      |                             (Dekat, keduanya baik)
      +─────────────────→ Data
      Gap KECIL, score TINGGI
```

### Metode 2: Training vs Validation Score

```python
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Analisis pengaruh max_depth
param_range = range(1, 21)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X, y, param_name='max_depth', param_range=param_range, cv=5
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

print("Depth | Train Score | Val Score | Gap")
print("-" * 45)
for d, tr, va in zip(param_range, train_mean, val_mean):
    gap = tr - va
    flag = ""
    if d <= 3: flag = "← Underfitting"
    elif d >= 12: flag = "← Overfitting"
    else: flag = "← Good area"
    print(f"  {d:2d}  |   {tr:.4f}    |  {va:.4f}   | {gap:.4f} {flag}")
```

### Metode 3: Perbedaan Training dan Test Score

| Kondisi | Training Score | Test Score | Gap | Penjelasan |
|---------|---------------|------------|-----|-----------|
| **Underfitting** | Rendah | Rendah | Kecil | Model tidak belajar |
| **Overfitting** | Sangat Tinggi | Rendah | Besar | Model menghafal |
| **Good Fit** | Tinggi | Tinggi | Kecil | Model generalisasi baik |

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
gap = train_score - test_score

print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy:     {test_score:.4f}")
print(f"Gap:               {gap:.4f}")

if gap > 0.15:
    print("⚠️ KEMUNGKINAN OVERFITTING — Gap antara train dan test terlalu besar")
elif train_score < 0.7 and test_score < 0.7:
    print("⚠️ KEMUNGKINAN UNDERFITTING — Keduanya rendah")
else:
    print("✅ Model terlihat good fit")
```

---

## 4.5 Cara Mengatasi Overfitting

### Strategi 1: Regularization — Menghukum Kompleksitas

Regularization menambahkan **penalti** pada parameter model, mencegah model menjadi terlalu kompleks.

```python
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=50, n_informative=5, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Tanpa regularization (overfitting pada 50 fitur, hanya 5 informatif)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression — Train: {lr.score(X_train, y_train):.4f}, Test: {lr.score(X_test, y_test):.4f}")

# L2 Regularization (Ridge)
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
print(f"Ridge (α=10)    — Train: {ridge.score(X_train, y_train):.4f}, Test: {ridge.score(X_test, y_test):.4f}")

# L1 Regularization (Lasso)
lasso = Lasso(alpha=1)
lasso.fit(X_train, y_train)
print(f"Lasso (α=1)     — Train: {lasso.score(X_train, y_train):.4f}, Test: {lasso.score(X_test, y_test):.4f}")

# Berapa fitur yang di-nolkan oleh Lasso?
zero_features = np.sum(lasso.coef_ == 0)
print(f"Lasso men-nolkan {zero_features} dari {len(lasso.coef_)} fitur")
```

**Rumus Regularization:**

```
Tanpa regularization:  Loss = Σ(y_pred - y_true)²
L2 (Ridge):           Loss = Σ(y_pred - y_true)² + α × Σ(w²)
L1 (Lasso):            Loss = Σ(y_pred - y_true)² + α × Σ(|w|)
```

| Regularization | Cara Kerja | Efek |
|---------------|-----------|------|
| **L2 (Ridge)** | Penalti kuadrat parameter | Menyusutkan parameter kecil, TIDAK menolkan |
| **L1 (Lasso)** | Penalti absolut parameter | Bisa menolkan parameter (feature selection otomatis) |
| **ElasticNet** | Gabungan L1 + L2 | Feature selection + shrinkage |

### Strategi 2: Early Stopping — Berhenti Sebelum Terlambat

```python
from sklearn.neural_network import MLPRegressor

# Training tanpa early stopping → overfitting
model_no_stop = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000,
                              random_state=42, early_stopping=False)
model_no_stop.fit(X_train, y_train)

# Training DENGAN early stopping → berhenti saat validation mulai naik
model_early_stop = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000,
                                 random_state=42, early_stopping=True,
                                 validation_fraction=0.2, n_iter_no_change=10)
model_early_stop.fit(X_train, y_train)

print(f"Tanpa early stopping: {model_no_stop.n_iter_} iterasi")
print(f"Dengan early stopping: {model_early_stop.n_iter_} iterasi (berhenti lebih awal!)")
```

### Strategi 3: Dropout (Neural Network)

```python
from sklearn.neural_network import MLPClassifier

# Tanpa dropout
model_no_dropout = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500)

# Dropout = secara random "matikan" neuron saat training
# Saat inference, semua neuron aktap → prediksi lebih robust
# Di scikit-learn, tidak ada dropout langsung, tapi kita bisa simulasi
# dengan noise atau smaller network
model_smaller = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500)

# Di PyTorch/TensorFlow:
# nn.Dropout(p=0.5)  # Matikan 50% neuron secara random
```

### Strategi 4: Data Augmentation — Perbanyak Data Tanpa Mengumpulkan Data Baru

```python
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X, y = digits.data, digits.target

# Data augmentation untuk gambar: rotasi, shift, noise
def augment_data(X, y, factor=3):
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(factor):
        noise = np.random.normal(0, 0.5, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)
    
    # Shift horizontally
    X_shifted = np.roll(X, 1, axis=1)
    X_aug.append(X_shifted)
    y_aug.append(y)
    
    return np.vstack(X_aug), np.hstack(y_aug)

X_aug, y_aug = augment_data(X, y, factor=2)
print(f"Original: {len(X)} samples")
print(f"Augmented: {len(X_aug)} samples (3x lipat)")
```

### Strategi 5: Model Simplification — Kurangi Kompleksitas

```python
# Decision Tree: kurangi kedalaman
dt_overfit = DecisionTreeClassifier(random_state=42)  # Unlimited depth
dt_good = DecisionTreeClassifier(max_depth=5, random_state=42)  # Limited depth

# Neural Network: kurangi neuron
nn_overfit = MLPClassifier(hidden_layer_sizes=(500, 500, 500))
nn_good = MLPClassifier(hidden_layer_sizes=(50, 25))

# K-NN: tambah K (lebih banyak tetangga = lebih smooth)
knn_overfit = KNeighborsClassifier(n_neighbors=1)  # Very specific
knn_good = KNeighborsClassifier(n_neighbors=7)     # More general
```

### Strategi 6: Ensemble Methods — Gabungkan Model Lemah Menjadi Kuat

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Decision Tree tunggal → rentan overfitting
dt = DecisionTreeClassifier(random_state=42)

# Random Forest → gabungan banyak Decision Tree → kurangi overfitting
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Gradient Boosting → gabungan sequential → perlu hati-hati tuning
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

for name, model in [('Decision Tree', dt), ('Random Forest', rf), ('Gradient Boosting', gb)]:
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name:20s} | Train: {train_score:.4f} | Test: {test_score:.4f}")
```

### Ringkasan Strategi Anti-Overfitting

| Strategi | Cara Kerja | Kapan Efektif |
|----------|-----------|---------------|
| **Regularization (L1/L2)** | Penalti parameter besar | Linear model, Neural Network |
| **Early Stopping** | Berhenti training saat val loss naik | Neural Network, Gradient Boosting |
| **Dropout** | Random matikan neuron saat training | Neural Network |
| **Data Augmentation** | Perbanyak data secara artifisial | Image, Audio |
| **Model Simplification** | Kurangi parameter | Decision Tree, NN, Polynomial |
| **Ensemble** | Gabungkan banyak model | Tree-based, semua |
| **Cross-Validation** | Evaluasi lebih robust | Semua model |
| **Feature Selection** | Kurangi fitur tidak penting | High-dimensional data |
| **Batch Normalization** | Stabilkan training | Deep Neural Network |

---

## 4.6 Cara Mengatasi Underfitting

Underfitting adalah kebalikan dari overfitting — model terlalu sederhana.

### Strategi 1: Tambah Kompleksitas Model

```python
# Dari model sederhana ke model lebih kompleks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Underfitting: Linear regression (garis lurus)
model_simple = LinearRegression()

# Lebih baik: Polynomial regression (kurva)
model_complex = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])
```

### Strategi 2: Tambah Fitur

```python
# Hanya luas tanah → skor rendah
features_basic = ['luas_tanah']

# Tambah fitur baru → skor meningkat
features_enhanced = ['luas_tanah', 'lokasi_tier', 'jumlah_kamar',
                      'umur_rumah', 'luas_per_kamar', 'dekat_mrt']
```

### Strategi 3: Kurangi Regularisasi

```python
# Regularisasi terlalu kuat → underfitting
model_over_regularized = Ridge(alpha=1000)  # Alpha terlalu besar

# Kurangi regularisasi
model_balanced = Ridge(alpha=1)  # Alpha lebih kecil
```

### Strategi 4: Training Lebih Lama

```python
# Neural network: training terlalu sebentar
model_short = MLPClassifier(max_iter=10)   # Underfitting

# Training lebih lama
model_long = MLPClassifier(max_iter=500)   # Lebih baik
```

### Strategi 5: Gunakan Model yang Lebih Kuat

```python
# Linear model tidak cukup untuk data non-linear
model_weak = LinearRegression()  # Underfit

# Decision tree bisa menangkap non-linearitas
model_strong = DecisionTreeRegressor(max_depth=10)  # Lebih baik
```

---

## 4.7 The Sweet Spot — Menemukan Keseimbangan

### Grid Search: Cari Parameter Terbaik

```python
from sklearn.model_selection import GridSearchCV

# Definisikan grid parameter
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

# Grid search dengan cross-validation
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,           # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,       # Pakai semua CPU
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# Analisis hasil
results = pd.DataFrame(grid_search.cv_results_)
print(f"\nTrain scores range: {results['mean_train_score'].min():.4f} - {results['mean_train_score'].max():.4f}")
print(f"Test scores range: {results['mean_test_score'].min():.4f} - {results['mean_test_score'].max():.4f}")
```

### Random Search: Lebih Efisien untuk Banyak Parameter

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions,
    n_iter=50,       # Coba 50 kombinasi random
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

### Checklist: Apakah Modelmu Overfitting atau Underfitting?

```
DIAGNOSA MODEL:

1. Training score TINGGI, test score RENDAH → ⚠️ OVERFITTING
   Solusi:
   □ Kurangi kompleksitas model (max_depth, fewer neurons)
   □ Tambah regularization (L1, L2, dropout)
   □ Perbanyak data (augmentation, collect more)
   □ Gunakan cross-validation
   □ Early stopping
   □ Ensemble methods

2. Training score RENDAH, test score RENDAH → ⚠️ UNDERFITTING
   Solusi:
   □ Tambah kompleksitas model
   □ Tambah fitur yang relevan
   □ Kurangi regularization
   □ Training lebih lama (more epochs)
   □ Ganti model yang lebih kuat

3. Training score TINGGI, test score TINGGI, gap KECIL → ✅ GOOD FIT
   Tetap monitoring:
   □ Cross-validation score konsisten?
   □ Score di data baru juga bagus?
   □ Tidak ada data leakage?
```

---

## 4.8 Contoh End-to-End: Diagnosis dan Perbaikan

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. BUAT DATA
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                            n_redundant=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MODEL AWAL (mungkin overfit)
print("=" * 60)
print("MODEL 1: Decision Tree tanpa batas (kemungkinan overfit)")
print("=" * 60)
dt_overfit = DecisionTreeClassifier(random_state=42)
dt_overfit.fit(X_train, y_train)
print(f"Train accuracy: {dt_overfit.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {dt_overfit.score(X_test, y_test):.4f}")
print(f"Gap: {dt_overfit.score(X_train, y_train) - dt_overfit.score(X_test, y_test):.4f}")
print(f"Max depth: {dt_overfit.get_depth()}")
# Train: 1.0000, Test: ~0.87, Gap: ~0.13 → OVERFITTING!

# 3. FIX: Kurangi kompleksitas
print("\n" + "=" * 60)
print("MODEL 2: Decision Tree dengan max_depth (lebih baik)")
print("=" * 60)
dt_fixed = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
dt_fixed.fit(X_train, y_train)
print(f"Train accuracy: {dt_fixed.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {dt_fixed.score(X_test, y_test):.4f}")
print(f"Gap: {dt_fixed.score(X_train, y_train) - dt_fixed.score(X_test, y_test):.4f}")
# Train: ~0.92, Test: ~0.89, Gap: ~0.03 → LEBIH BAIK!

# 4. FIX LEBIH BAIK: Random Forest (ensemble)
print("\n" + "=" * 60)
print("MODEL 3: Random Forest (best)")
print("=" * 60)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print(f"Train accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {rf.score(X_test, y_test):.4f}")
print(f"Gap: {rf.score(X_train, y_train) - rf.score(X_test, y_test):.4f}")

# 5. Cross-validation untuk evaluasi lebih robust
cv_scores = cross_val_score(rf, X, y, cv=10)
print(f"\n10-Fold CV scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# Rata-rata konsisten, std rendah → model stabil dan generalisasi baik
```

---

## 4.9 Ringkasan Bab 4

### Peta Konsep

```
                    COMPLEXITY
                    ─────────────────→
                    Simple          Complex
                    │                 │
              Underfitting      Overfitting
              │                 │
              • Bias tinggi     • Bias rendah
              • Variance rendah • Variance tinggi
              • Training jelek  • Training sempurna
              • Test jelek      • Test jelek
              │                 │
              SOLUSI:           SOLUSI:
              • Model lebih     • Model lebih
                kompleks          sederhana
              • Tambah fitur    • Regularization
              • Training        • Early stopping
                lebih lama      • Data augmentation
              • Kurangi         • Dropout
                regularisasi    • Ensemble
                                • Cross-validation
                    │                 │
                    └────────┬────────┘
                             │
                      SWEET SPOT
                    (Good Fit)
                             │
                      • Bias rendah
                      • Variance rendah
                      • Training bagus
                      • Test bagus
                      • Gap kecil
```

1. **Overfitting** = Model menghafal data training termasuk noise. Training tinggi, test rendah.
2. **Underfitting** = Model terlalu sederhana untuk menangkap pola. Training rendah, test rendah.
3. **Bias-Variance Tradeoff** = Semakin kompleks model, bias turun tapi variance naik.
4. **Regularization** menambahkan penalti pada kompleksitas model.
5. **Early Stopping** mencegah training berlebihan.
6. **Cross-Validation** memberikan evaluasi yang lebih dapat dipercaya.
7. **Learning Curves** membantu mendiagnosis overfitting vs underfitting.
8. **Sweet Spot** ada di tengah — model cukup kompleks untuk menangkap pola, tapi tidak terlalu kompleks untuk menghafal noise.

---

**Selanjutnya → Bab 5: Evaluasi Model — Mengukur Seberapa Baik Modelmu**