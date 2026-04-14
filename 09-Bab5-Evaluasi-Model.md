# Bab 5: Evaluasi Model — Mengukur Seberapa Baik Modelmu

---

## Analogi Pembuka: Ujian dan Penilaian

Model ML seperti seorang murid yang sudah belajar. Sekarang waktunya diuji. Tapi bagaimana kita mengukur seberapa "pintar" murid itu?

Kalau ujian terlalu mudah (soal yang sama dengan latihan), semua murid dapat nilai sempurna — tapi kita tidak tahu apakah mereka benar-benar paham.

Kalau ujian terlalu sulit (materi yang tidak pernah diajarkan), semua murid gagal — tapi kita tidak tahu apakah itu karena mereka bodoh atau soalnya tidak adil.

Evaluasi model ML menghadapi masalah yang sama. Kita perlu **metrik yang tepat** dan **cara pengujian yang adil** untuk mengukur kualitas model kita secara jujur.

---

## 5.1 Mengapa Akurasi Saja Tidak Cukup?

### Contoh Problematis: Deteksi Penyakit Langka

```python
# Dataset: 10,000 pasien
# 9,950 sehat (99.5%)
# 50 sakit (0.5%)

# Model yang SELALU prediksi "sehat":
accuracy = 9950 / 10000
print(f"Akurasi: {accuracy:.2%}")  # 99.50%!
```

Akurasi 99.5%! Tapi model ini **sama sekali tidak berguna** — dia tidak pernah mendeteksi satupun pasien yang sakit.

Ini disebut **Accuracy Paradox**: dataset yang sangat tidak seimbang membuat akurasi menyesatkan.

### Kita Butuh Metrik yang Lebih Baik

```
                     PREDIKSI
                 Positif    Negatif
               ┌──────────┬──────────┐
   FAKTA  Positif │    TP    │    FN    │
               ├──────────┼──────────┤
          Negatif │    FP    │    TN    │
               └──────────┴──────────┘

TP = True Positive  (Sakit, diprediksi sakit)     ← BENAR
TN = True Negative  (Sehat, diprediksi sehat)     ← BENAR
FP = False Positive (Sehat, diprediksi sakit)     ← ERROR TIPE 1
FN = False Negative (Sakit, diprediksi sehat)     ← ERROR TIPE 2
```

---

## 5.2 Confusion Matrix — Peta Kesalahan Model

```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data tidak seimbang: 95% kelas 0, 5% kelas 1
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"""
              Prediksi
              0      1
  Fakt 0  [{cm[0,0]:3d}  {cm[0,1]:3d}]    ← TN={cm[0,0]}, FP={cm[0,1]}
  a    1  [{cm[1,0]:3d}  {cm[1,1]:3d}]    ← FN={cm[1,0]}, TP={cm[1,1]}
""")
```

### Interpetasi Confusion Matrix

```
              Prediksi
              Negatif   Positif
  
  Aktual  ┌──────────┬──────────┐
Negatif   │   TN     │   FP     │    ← Sehat tapi diprediksi sakit
          │  (Benar) │ (Salah)  │       (False Alarm)
          ├──────────┼──────────┤
Positif   │   FN     │   TP     │    ← Sakit tapi diprediksi sehat
          │ (Salah)  │ (Benar)  │       (Missed Diagnosis)
          └──────────┴──────────┘
```

**Bergantung pada konteks, FP dan FN punya konsekuensi berbeda:**

| Konteks | FP (False Positive) | FN (False Negative) |
|---------|--------------------|--------------------|
| Deteksi kanker | Pasien sehat diberitahu mungkin kanker → cemas, tes lanjutan | Pasien kanker diberitahu sehat → **FATAL** |
| Filter spam | Email penting masuk spam → **MERESEHKAU** | Email spam masuk inbox → sedikit mengganggu |
| Deteksi fraud | Transaksi normal diblokir → mengganggu nasabah | Transaksi fraud lolos → **KERUGIAN FINANSIAL** |
| Mobil otonom | Rem mendadak tanpa halangan → ketidaknyamanan | Tidak rem saat ada halangan → **KECELAKAAN** |

---

## 5.3 Metrik Evaluasi untuk Klasifikasi

### Rumus-Rumus

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)
           "Dari semua yang diprediksi positif, berapa yang benar-benar positif?"

Recall (Sensitivity/True Positive Rate) = TP / (TP + FN)
           "Dari semua yang sebenarnya positif, berapa yang berhasil dideteksi?"

Specificity (True Negative Rate) = TN / (TN + FP)
           "Dari semua yang sebenarnya negatif, berapa yang benar-benar negatif?"

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
           "Rata-rata harmonik dari Precision dan Recall"

Fβ-Score = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
           "F2 lebih menekankan Recall, F0.5 lebih menekankan Precision"
```

### Contoh Perhitungan Manual

```
Confusion Matrix:
              Prediksi
              0      1
  Aktual 0  [ 270    30 ]    TN=270, FP=30
         1  [  10    40 ]    FN=10,  TP=40

Accuracy    = (270 + 40) / (270 + 30 + 10 + 40) = 310 / 350 = 0.886 = 88.6%
Precision   = 40 / (40 + 30) = 40 / 70 = 0.571 = 57.1%
Recall      = 40 / (40 + 10) = 40 / 50 = 0.800 = 80.0%
F1-Score    = 2 × (0.571 × 0.800) / (0.571 + 0.800) = 0.667 = 66.7%
Specificity = 270 / (270 + 30) = 270 / 300 = 0.900 = 90.0%
```

### Kapan Pakai Metrik Apa?

| Situasi | Metrik Utama | Kenapa? |
|---------|-------------|---------|
| Dataset seimbang | Accuracy | Metrik umum, representatif |
| Dataset tidak seimbang, kelas minoritas penting | F1-Score | Balance precision & recall |
| False Positive sangat mahal | Precision | Minimalkan false alarm |
| False Negative sangat berbahaya | Recall | Tangkap sebanyak mungkin positif |
| Kedua jenis error sama penting | F1-Score | Kompromi terbaik |

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Sehat', 'Sakit']))
```

```
=== Classification Report ===
              precision    recall  f1-score   support

       Sehat       0.96      0.90      0.93       300
       Sakit       0.57      0.80      0.67        50

    accuracy                           0.89       350
   macro avg       0.77      0.85      0.80       350
weighted avg       0.91      0.89      0.89       350
```

---

## 5.4 ROC Curve dan AUC — Metrik yang Tidak Tergantung Threshold

### Masalah dengan Threshold Tetap

Secara default, model klasifikasi memprediksi kelas 1 jika probabilitas ≥ 0.5. Tapi 0.5 tidak selalu threshold terbaik!

```python
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Probabilitas prediksi (bukan kelas)
y_proba = model.predict_proba(X_test)[:, 1]

# Threshold berbeda → hasil berbeda
for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    y_pred_custom = (y_proba >= threshold).astype(int)
    tp = ((y_pred_custom == 1) & (y_test == 1)).sum()
    fp = ((y_pred_custom == 1) & (y_test == 0)).sum()
    fn = ((y_pred_custom == 0) & (y_test == 1)).sum()
    tn = ((y_pred_custom == 0) & (y_test == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Threshold={threshold:.1f} | Precision={precision:.3f} | Recall={recall:.3f} | Pred Positive={tp+fp}")
```

```
Threshold=0.1 | Precision=0.164 | Recall=1.000 | Pred Positive=305  → Terlalu sensitif
Threshold=0.3 | Precision=0.333 | Recall=0.940 | Pred Positive=141  → Masih sensitif
Threshold=0.5 | Precision=0.571 | Recall=0.800 | Pred Positive=70   → Default
Threshold=0.7 | Precision=0.750 | Recall=0.600 | Pred Positive=40   → Lebih spesifik
Threshold=0.9 | Precision=1.000 | Recall=0.260 | Pred Positive=13   → Terlalu spesifik
```

### ROC Curve: Semua Threshold Sekaligus

**ROC (Receiver Operating Characteristic)** memplot **True Positive Rate** vs **False Positive Rate** di setiap threshold.

```
TPR = Recall = TP / (TP + FN)    ← Semakin tinggi, semakin baik
FPR = FP / (FP + TN)              ← Semakin rendah, semakin baik

      TPR
    1.0 ┤         ╭──────────── ROC Curve (model baik)
        │       ╭─╯
        │     ╭─╯
        │   ╭─╯
        │ ╭─╯    ╱ Random Classifier (garis diagonal)
    0.0 ┤─╯─────╱────────────────── FPR
       0.0                  1.0
```

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Hitung ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"AUC Score: {auc:.4f}")
```

### AUC — Area Under the ROC Curve

AUC merangkum ROC curve menjadi **satu angka**:

| AUC | Interpretasi |
|-----|-------------|
| 1.0 | Sempurna — model membedakan kelas dengan sempurna |
| 0.9 - 1.0 | Sangat baik |
| 0.8 - 0.9 | Baik |
| 0.7 - 0.8 | Cukup |
| 0.6 - 0.7 | Kurang baik |
| 0.5 | Random — model tidak lebih baik dari tebakan acak |

**Interpretasi AUC:** "Jika kamu ambil satu instance positif dan satu instance negatif secara acak, probabilitas bahwa model memberi skor lebih tinggi pada instance positif adalah sebesar AUC."

### Precision-Recall Curve

Untuk dataset **sangat tidak seimbang**, PR Curve lebih informatif:

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_arr, precision_arr, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Kapan pakai ROC vs PR Curve?**

| Situasi | Gunakan | Kenapa? |
|---------|---------|---------|
| Dataset seimbang | ROC-AUC | Kedua kelas sama penting |
| Dataset TIDAK seimbang | PR-AUC | Fokus pada kelas minoritas |
| FP sangat mahal | PR Curve | Precision penting |
| FN sangat berbahaya | ROC Curve | Recall penting |

---

## 5.5 Metrik Evaluasi untuk Regresi

Untuk masalah regresi (prediksi angka), metriknya berbeda:

```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, root_mean_squared_error)

y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.0])
y_pred = np.array([2.8, 5.3, 2.0, 6.5, 4.5])

errors = y_true - y_pred
# errors = [0.2, -0.3, 0.5, 0.5, -0.5]
```

### Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ|y_true - y_pred|
    = (0.2 + 0.3 + 0.5 + 0.5 + 0.5) / 5
    = 2.0 / 5
    = 0.4
```

- Satuan sama dengan target (kalau target dalam juta rupiah, MAE dalam juta rupiah)
- **Robust terhadap outlier** (tidak menghukum error besar terlalu keras)
- Interpretasi mudah: "Rata-rata prediksi meleset 0.4 unit"

### Mean Squared Error (MSE)

```
MSE = (1/n) × Σ(y_true - y_pred)²
    = (0.04 + 0.09 + 0.25 + 0.25 + 0.25) / 5
    = 0.88 / 5
    = 0.176
```

- **Menghukum error besar lebih keras** (karena dikuadratkan)
- Satuan kuadrat dari target (juta rupiah²)
- Sensitif terhadap outlier

### Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √0.176 ≈ 0.42
```

- Satuan sama dengan target
- **Lebih sering digunakan** daripada MSE karena interpretasi lebih mudah
- Sensitif terhadap outlier

### R² Score (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)
   = 1 - (Σ(y_true - y_pred)² / Σ(y_true - ȳ)²)
```

- R² = 1.0 → Prediksi sempurna
- R² = 0.0 → Model sama baiknya dengan prediksi rata-rata
- R² < 0 → Model lebih buruk dari prediksi rata-rata (sangat buruk!)

```python
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"R²:   {r2_score(y_true, y_pred):.4f}")
```

### Perbandingan Metrik Regresi

| Metrik | Rumus | Satuan | Sensitif Outlier | Interpretasi |
|--------|-------|--------|------------------|-------------|
| MAE | Σ\|error\| / n | Sama dengan target | Rendah | Rata-rata kesalahan |
| MSE | Σerror² / n | Kuadrat target | Tinggi | Kesalahan dikuadratkan |
| RMSE | √MSE | Sama dengan target | Tinggi | Kesalahan standar |
| R² | 1 - SS_res/SS_tot | Tanpa satuan | Sedang | Persentase variansi yang dijelaskan |

---

## 5.6 Cross-Validation — Evaluasi yang Lebih Dapat Dipercaya

### Kenapa Single Train-Test Split Tidak Cukup?

Satu split bisa memberikan hasil yang berbeda tergantung random seed:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

# Satu split → bisa beruntung atau tidak
for seed in [42, 123, 456, 789, 0]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = LogisticRegression(max_iter=200)
    model.fit(X_tr, y_tr)
    print(f"Seed={seed:3d} → Accuracy={model.score(X_te, y_te):.4f}")
```

```
Seed= 42 → Accuracy=1.0000
Seed=123 → Accuracy=0.9778
Seed=456 → Accuracy=0.9778
Seed=789 → Accuracy=1.0000
Seed=  0 → Accuracy=0.9556
```

Hasil bervariasi dari 95.6% sampai 100%! Cross-validation memberikan estimasi yang lebih stabil.

### K-Fold Cross-Validation

```
5-Fold Cross-Validation:

Iterasi 1: [VALID][TRAI][TRAI][TRAI][TRAI]  → Score: 0.96
Iterasi 2: [TRAI][VALID][TRAI][TRAI][TRAI]  → Score: 0.98
Iterasi 3: [TRAI][TRAI][VALID][TRAI][TRAI]  → Score: 0.97
Iterasi 4: [TRAI][TRAI][TRAI][VALID][TRAI]  → Score: 0.95
Iterasi 5: [TRAI][TRAI][TRAI][TRAI][VALID]  → Score: 0.96

Rata-rata: 0.964 ± 0.011
```

```python
model = LogisticRegression(max_iter=200)

# 5-Fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV Scores: {scores}")
print(f"Rata-rata: {scores.mean():.4f}")
print(f"Std Deviasi: {scores.std():.4f}")

# 10-Fold CV (lebih stabil)
scores_10 = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print(f"\n10-Fold CV Scores: {scores_10}")
print(f"Rata-rata: {scores_10.mean():.4f}")
print(f"Std Deviasi: {scores_10.std():.4f}")

# Stratified K-Fold (untuk dataset tidak seimbang)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"\nStratified 5-Fold: {stratified_scores.mean():.4f} ± {stratified_scores.std():.4f}")
```

### Kapan Pakai K-Fold Berapa?

| Jumlah Fold | Keuntungan | Kerugangan | Kapan? |
|------------|-----------|-----------|--------|
| 5 | Seimbang, cepat | Variasi masih ada | Default, dataset menengah |
| 10 | Estimasi lebih stabil | Lebih lambat | Dataset cukup besar |
| Leave-One-Out | Maksimalkan data training | Sangat lambat | Dataset sangat kecil |

---

## 5.7 Metrik untuk Multi-Class Classification

```python
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X, y = digits.data, digits.target  # 10 kelas (0-9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Report
print(classification_report(y_test, y_pred))
```

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        53
           1       0.96      0.98      0.97        50
           2       1.00      0.98      0.99        47
           3       0.98      0.96      0.97        54
           4       0.98      1.00      0.99        60
           5       0.98      0.96      0.97        53
           6       1.00      0.98      0.99        55
           7       1.00      0.98      0.99        59
           8       0.96      0.98      0.97        54
           9       0.96      0.96      0.96        55

    accuracy                           0.98       540
   macro avg       0.98      0.98      0.98       540
weighted avg       0.98      0.98      0.98       540
```

### Macro vs Weighted Average

| Tipe | Cara Hitung | Kapan Pakai? |
|------|------------|-------------|
| **Macro avg** | Rata-rata sederhana semua kelas | Setiap kelas sama pentingnya |
| **Weighted avg** | Rata-rata tertimbang berdasarkan jumlah data per kelas | Kelas dengan data lebih banyak lebih penting |
| **Micro avg** | Hitung globally (TP, FP, FN dikumpulkan semua kelas) | Sama dengan accuracy untuk multi-class |

```python
# Perbandingan macro vs weighted
from sklearn.metrics import f1_score

print(f"Macro F1:    {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
```

---

## 5.8 Membandingkan Model

### Tabel Perbandingan

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'K-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB(),
}

results = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    model.fit(X_train, y_train)
    test_score = f1_score(y_test, model.predict(X_test), average='weighted')
    results.append({
        'Model': name,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test F1': test_score
    })

results_df = pd.DataFrame(results).sort_values('CV Mean', ascending=False)
print(results_df.to_string(index=False))
```

### Statistical Significance Testing

```python
from sklearn.model_selection import cross_val_score
from scipy import stats

# Bandingkan dua model secara statistik
model_a = RandomForestClassifier(n_estimators=100, random_state=42)
model_b = DecisionTreeClassifier(max_depth=5, random_state=42)

scores_a = cross_val_score(model_a, X, y, cv=10, scoring='accuracy')
scores_b = cross_val_score(model_b, X, y, cv=10, scoring='accuracy')

# t-test: Apakah perbedaan signifikan?
t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
print(f"Model A: {scores_a.mean():.4f} ± {scores_a.std():.4f}")
print(f"Model B: {scores_b.mean():.4f} ± {scores_b.std():.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Perbedaan SIGNIFIKAN (p < 0.05)")
else:
    print("Perbedaan TIDAK signifikan (p >= 0.05)")
```

---

## 5.9 Regression Evaluation — Contoh Lengkap

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)': Ridge(alpha=1),
    'Lasso (α=1)': Lasso(alpha=1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', C=10),
}

print(f"{'Model':20s} | {'MAE':>8s} | {'MSE':>8s} | {'RMSE':>8s} | {'R²':>8s}")
print("-" * 65)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{name:20s} | {mae:8.4f} | {mse:8.4f} | {rmse:8.4f} | {r2:8.4f}")
```

---

## 5.10 Checklist Evaluasi Model

```
□ METRIK YANG TEPAT
  ├── Dataset seimbang? → Accuracy + F1
  ├── Dataset tidak seimbang? → F1, Precision, Recall, AUC-PR
  ├── Regression? → MAE, RMSE, R²
  └── Multi-class? → Macro/Weighted F1

□ EVALUASI YANG ADIL
  ├── Train-Test Split (80/20 atau 70/30)
  ├── Cross-Validation (5-fold atau 10-fold)
  ├── Stratified Split (jika imbalance)
  └── JANGAN evaluasi di training data!

□ CONFUSION MATRIX
  ├── Cek per kelas, bukan hanya rata-rata
  ├── Cek FP dan FN — mana yang lebih berbahaya?
  └── Perhatikan kelas minoritas

□ COMPARISON
  ├── Bandingkan beberapa model
  ├── Gunakan statistical significance test
  └── Pilih model terbaik berdasarkan konteks (bukan hanya angka)

□ INTERPRETABILITY
  ├── Apakah hasil masuk akal secara bisnis?
  ├── Apakah model bisa dijelaskan ke stakeholder?
  └── Apakah ada bias yang tidak diinginkan?

□ ROBUSTNESS
  ├── Cek di data baru yang belum pernah dilihat
  ├── Cek di subgroup (gender, umur, lokasi)
  └── Cek terhadap adversarial examples
```

---

## 5.11 Ringkasan Bab 5

1. **Akurasi saja tidak cukup** — selalu periksa precision, recall, F1, dan confusion matrix.
2. **Confusion matrix** menunjukkan di mana model bersalah (FP vs FN).
3. **ROC-AUC** bagus untuk evaluasi keseluruhan, **PR-AUC** lebih baik untuk dataset tidak seimbang.
4. **Cross-validation** memberikan estimasi yang lebih stabil daripada single split.
5. **Pilih metrik berdasarkan konteks bisnis** — FP dan FN punya konsekuensi berbeda.
6. **Bandingkan model secara statistik**, bukan hanya berdasarkan satu angka.
7. **Interpretasi hasil** — akurasi 99% bisa menyesatkan jika dataset tidak seimbang.

---

**Selanjutnya → Bab 6: Ensemble Methods & Topik Lanjutan**