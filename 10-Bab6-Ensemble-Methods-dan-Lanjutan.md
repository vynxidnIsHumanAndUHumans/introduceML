# Bab 6: Ensemble Methods & Topik Lanjutan — Dari Bagus Menjadi Luar Biasa

---

## Analogi Pembuka: Dewan Juri

Bayangkan sebuah kontes menyanyi. Ada tiga cara menentukan pemenang:

1. **Satu juri**: Juri itu bisa bias, bisa salah, bisa sedang banyak masalah. Satu opini. Satu titik kegagalan.

2. **Tiga juri, suara terbanyak menang**: Kalau satu juri salah, dua juri lain bisa mengkoreksi. Lebih reliable.

3. **Ratusan penonton, voting**: Ribuan意见 digabung. Mayoritas hampir pasti benar.

**Itulah Ensemble Methods.** Alih-alih mengandalkan satu model, kita menggabungkan banyak model. Kekuatan kolektif mengalahkan kelemahan individu.

> *"Wisdom of the Crowd"* — Francis Galton menemukan bahwa rata-rata tebakan 800 orang tentang berat seekar sapi lebih akurat daripada tebakan satu ahli pun.

---

## 6.1 Mengapa Ensemble Bekerja?

### Intuisi Matematika

Asumsikan kita punya 3 model yang masing-masing punya akurasi 70% dan error yang **independen** (tidak berkorelasi):

```
Probabilitas 1 model benar: 0.70
Probabilitas 1 model salah: 0.30

Majority vote dari 3 model:
- Semua benar (3/3):             0.70³ = 0.343
- Dua benar, satu salah (2/3):   3 × 0.70² × 0.30 = 0.441
- Dua salah, satu benar (1/3):   3 × 0.70 × 0.30² = 0.189
- Semua salah (0/3):              0.30³ = 0.027

Akurasi ensemble (mayoritas benar) = 0.343 + 0.441 = 0.784 = 78.4%
```

Dari 70% per model, jadi 78.4% secara kolektif! Makin banyak model yang digabung, makin tinggi akurasinya (asalkan error independen).

### Syarat Ensemble Bekerja Baik

| Syarat | Penjelasan | Analogi |
|--------|-----------|---------|
| **Diversity** | Model harus berbeda (membuat error berbeda) | Juri dengan latar belakang berbeda |
| **Accuracy** | Setiap model harus lebih baik dari random | Juri yang kompeten, bukan clueless |
| **Independence** | Error model tidak berkorelasi tinggi | Juri yang independen, bukan saling mencontek |

---

## 6.2 Bagging — Bootstrap Aggregating

### Konsep

Bagging = melatih banyak model **yang sama** dengan data **yang berbeda** (hasil bootstrap sampling), lalu menggabungkan hasilnya.

**Bootstrap sampling**: Ambil sampel secara random WITH replacement dari data asli. Setiap sampel memiliki ukuran yang sama dengan data asli, tapi beberapa data mungkin muncul berkali-kali dan beberapa tidak muncul sama sekali.

```
Data asli: [A, B, C, D, E, F, G, H, I, J]

Bootstrap Sample 1: [A, C, C, E, F, H, I, I, J, J]  (C, I, J berulang; B, D, G tidak muncul)
Bootstrap Sample 2: [A, B, D, E, F, G, G, H, I, J]   (G berulang; C tidak muncul)
Bootstrap Sample 3: [B, B, C, D, E, F, H, I, I, J]    (B, I berulang; A, G tidak muncul)

Model 1 dilatih di Sample 1
Model 2 dilatih di Sample 2
Model 3 dilatih di Sample 3

Prediksi akhir = Majority Vote (klasifikasi) atau Average (regresi)
```

### Random Forest — Raja Bagging

Random Forest = Bagging + Random Feature Selection

Setiap Decision Tree di Random Forest:
1. Dilatih di **bootstrap sample** (data berbeda)
2. Di setiap split, hanya mempertimbangkan **subset acak fitur** (bukan semua fitur)

Langkah kedua inilah yang membuat pohon-pohon lebih **beragam** (diverse).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X, y = make_classification(n_samples=5000, n_features=20, n_informative=10,
                            n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Single Decision Tree (rentan overfitting)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print(f"Decision Tree:")
print(f"  Train accuracy: {dt.score(X_train, y_train):.4f}")
print(f"  Test accuracy:  {dt.score(X_test, y_test):.4f}")

# Random Forest (lebih robust)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"\nRandom Forest:")
print(f"  Train accuracy: {rf.score(X_train, y_train):.4f}")
print(f"  Test accuracy:  {rf.score(X_test, y_test):.4f}")

# Cross-validation
dt_cv = cross_val_score(dt, X, y, cv=5).mean()
rf_cv = cross_val_score(rf, X, y, cv=5).mean()
print(f"\nCV Decision Tree: {dt_cv:.4f}")
print(f"CV Random Forest: {rf_cv:.4f}")
```

### Random Forest — Hyperparameters Penting

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],          # Jumlah pohon
    'max_depth': [5, 10, 20, None],           # Kedalaman max
    'min_samples_split': [2, 5, 10],          # Min sampel untuk split
    'min_samples_leaf': [1, 2, 4],            # Min sampel di leaf
    'max_features': ['sqrt', 'log2', None],   # Jumlah fitur yang dipertimbangkan
    'bootstrap': [True],                      # Bootstrap sampling
}

# Grid search untuk cari parameter terbaik
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
```

### Feature Importance

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance dari Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for i, idx in enumerate(indices[:10]):
    print(f"  {i+1:2d}. Feature {idx:2d} — Importance: {importances[idx]:.4f}")
```

---

## 6.3 Boosting — Belajar dari Kesalahan

### Konsep

Boosting = melatih model secara **berurutan**, di mana setiap model baru **fokus pada kesalahan** model sebelumnya.

```
Data:      ● ● × ● × ● ● × ● ●
                              ↑
                           error

Model 1:   ────/──/──────/──  (Prediksi kesalahan di ×)
Model 2:   ──────/──/────/────  (Fokus pada data yang diprediksi salah oleh Model 1)
Model 3:   ────────/──/──────  (Fokus pada data yang masih salah)
                              ↓
Ensemble:  ────────────────────  (Gabungan, jauh lebih baik)
```

### Adaptive Boosting (AdaBoost)

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost: model berikutnya fokus pada data yang salah diklasifikasi
# Data yang salah → bobotnya dinaikkan
# Data yang benar → bobotnya diturunkan

ada = AdaBoostClassifier(
    n_estimators=50,       # Jumlah model (weak learners)
    learning_rate=1.0,     # Kontribusi setiap model
    random_state=42
)
ada.fit(X_train, y_train)
print(f"AdaBoost Test accuracy: {ada.score(X_test, y_test):.4f}")

# Setiap weak learner adalah Decision Tree dengan depth=1 (stump)
# Stump alone sangat jelek (~50% accuracy untuk binary)
# Tapi gabungan 50 stump bisa sangat kuat!
```

**Cara kerja AdaBoost:**

```
Iterasi 1:
  - Latih weak learner (decision stump)
  - Hitung error rate
  - Hitung bobot model (alpha = 0.5 × ln((1-error)/error))
  - Update bobot data: data yang salah → bobot naik

Iterasi 2:
  - Latih weak learner di data dengan bobot baru
  - Data yang salah di iterasi 1 sekarang PENTING (bobot besar)
  - Hitung error rate dan bobot model
  - Update bobot data lagi

...

Iterasi N:
  - Prediksi akhir = weighted majority vote
  - Setiap model berkontribusi berdasarkan bobotnya (alpha)
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting: setiap model baru memprediksi RESIDUAL (error) model sebelumnya
gb = GradientBoostingClassifier(
    n_estimators=100,      # Jumlah boosting stages
    learning_rate=0.1,     # Shrinkage (langkah kecil lebih baik tapi lambat)
    max_depth=3,           # Kedalaman setiap tree
    subsample=0.8,         # Fraksi data untuk setiap tree (stochastic)
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Gradient Boosting Test accuracy: {gb.score(X_test, y_test):.4f}")
```

**Cara kerja Gradient Boosting:**

```
Step 1: Latih model awal F₀(x) = mean(y)  (prediksi sederhana)
Step 2: Hitung residual r₁ = y - F₀(x)  (error)
Step 3: Latih decision tree h₁(x) pada residual r₁
Step 4: Update model F₁(x) = F₀(x) + learning_rate × h₁(x)
Step 5: Hitung residual baru r₂ = y - F₁(x)
Step 6: Latih decision tree h₂(x) pada residual r₂
Step 7: Update model F₂(x) = F₁(x) + learning_rate × h₂(x)
...
Step N: Model akhir F_N(x) = F₀(x) + Σ(lr × hᵢ(x))
```

### XGBoost — Extreme Gradient Boosting

```python
# XGBoost: Gradient Boosting yang dioptimasi untuk performa dan kecepatan
# Instalasi: pip install xgboost

try:
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    xgb_model = xgb.train(params, dtrain, num_boost_round=100)
    y_pred_xgb = xgb_model.predict(dtest)
    accuracy_xgb = (y_pred_xgb == y_test).mean()
    print(f"XGBoost Test accuracy: {accuracy_xgb:.4f}")
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
```

### LightGBM — Light Gradient Boosting Machine

```python
try:
    import lightgbm as lgb

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    print(f"LightGBM Test accuracy: {lgb_model.score(X_test, y_test):.4f}")
except ImportError:
    print("LightGBM not installed. Install with: pip install lightgbm")
```

### Perbandingan Boosting Algorithms

| Algoritma | Kecepatan | Akurasi | Handling | Penggunaan |
|-----------|-----------|---------|----------|-----------|
| **AdaBoost** | Lambat | Baik | Data categorical | Baseline, sederhana |
| **Gradient Boosting** | Sedang | Sangat Baik | Data numerik | Sklearn default |
| **XGBoost** | Cepat | Sangat Baik | Regularisasi built-in | Kompetisi, produksi |
| **LightGBM** | Sangat Cepat | Sangat Baik | Histogram-based | Dataset besar |
| **CatBoost** | Cepat | Sangat Baik | Categorical built-in | Data kategorikal |

---

## 6.4 Stacking — Meta-Learning

### Konsep

Stacking = menggunakan prediksi beberapa model sebagai **input** untuk model baru (meta-learner).

```
            ┌────────────┐
            │  Model 1   │────┐
            │ (K-NN)     │    │
            └────────────┘    │
            ┌────────────┐    │    ┌────────────┐
            │  Model 2   │────┼────│  Meta      │──── Prediksi Akhir
            │ (SVM)      │    │    │  Learner   │
            └────────────┘    │    │ (LogReg)  │
            ┌────────────┐    │    └────────────┘
            │  Model 3   │────┘
            │ (RF)       │
            └────────────┘
```

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Base models (level 0)
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
]

# Meta-learner (level 1)
meta_learner = LogisticRegression(random_state=42)

# Stacking
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5  # Cross-validation untuk menghindari data leakage
)

stacking.fit(X_train, y_train)
print(f"Stacking Test accuracy: {stacking.score(X_test, y_test):.4f}")
```

---

## 6.5 Bagging vs Boosting vs Stacking

| Aspek | Bagging | Boosting | Stacking |
|-------|---------|----------|----------|
| **Training** | Paralel (independen) | Sequential (bergantung) | Bertingkat |
| **Tujuan** | Kurangi variance | Kurangi bias | Kurangi keduanya |
| **Data per model** | Bootstrap sample | Seluruh data + bobot | Cross-validation split |
| **Model lemah** | Diversifikasi | Diperbaiki bertahap | Digabung oleh meta-learner |
| **Overfitting?** | Jarang | Rentan (perlu tuning) | Mungkin |
| **Contoh** | Random Forest | XGBoost, AdaBoost | StackingClassifier |
| **Kapan?** | Variance tinggi | Bias tinggi | Model berbeda |

### Decision Tree: Kapan Pakai Apa?

```
Model overfitting (variance tinggi)?
│
├── YA → BAGGING (Random Forest)
│         Kurangi variance dengan menggabungkan banyak model
│
├── Model underfitting (bias tinggi)?
│   │
│   └── YA → BOOSTING (XGBoost, AdaBoost)
│             Kurangi bias dengan fokus pada error
│
└── Model berbeda punya kekuatan berbeda?
    │
    └── YA → STACKING
              Gabungkan kekuatan masing-masing model
```

---

## 6.6 Pipeline Lengkap — Dari Data Sampai Model

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml

# 1. Load data
titanic = fetch_openml('titanic', version=1, as_frame=True)
df = titanic.frame
print(f"Shape: {df.shape}")
print(df.head())

# 2. Define features
numeric_features = ['age', 'fare']
categorical_features = ['sex', 'embarked', 'pclass']

# 3. Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Prepare data
feature_cols = numeric_features + categorical_features
X = df[feature_cols]
y = df['survived'].astype(int)

# 6. Cross-validation (ALL preprocessing inside CV!)
cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring='accuracy')
print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 7. Try different models
models_to_try = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

print("\nModel Comparison:")
for name, model in models_to_try.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"  {name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

# 8. Final model training
full_pipeline.fit(X, y)

# 9. Feature importance (for tree-based models)
rf_model = full_pipeline.named_steps['classifier']
feature_names = (numeric_features +
                 list(full_pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['onehot']
                      .get_feature_names_out(categorical_features)))
importances = rf_model.feature_importances_

print("\nTop 5 Features:")
for idx in np.argsort(importances)[::-1][:5]:
    print(f"  {feature_names[idx]:20s}: {importances[idx]:.4f}")
```

---

## 6.7 Dimensi Reduction Lanjutan

### PCA — Principal Component Analysis

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
print(f"Dimensi asli: {X.shape}")  # (1797, 64) — 64 fitur

# Reduksi ke 2 dimensi untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Dimensi setelah PCA: {X_pca.shape}")  # (1797, 2)
print(f"Variansi yang dipertahankan: {sum(pca.explained_variance_ratio_):.2%}")

# Berapa komponen yang perlu dipertahankan?
pca_full = PCA()
pca_full.fit(X)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"Komponen untuk 95% variansi: {n_components_95}")

# Gunakan PCA dalam pipeline
from sklearn.pipeline import Pipeline
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Pertahankan 95% variansi
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

scores = cross_val_score(pca_pipeline, X, y, cv=5, scoring='accuracy')
print(f"\nPCA Pipeline CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Bandingkan tanpa PCA
baseline_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
scores_baseline = cross_val_score(baseline_pipeline, X, y, cv=5, scoring='accuracy')
print(f"Baseline CV Accuracy: {scores_baseline.mean():.4f} ± {scores_baseline.std():.4f}")
```

---

## 6.8 Handling Imbalanced Data — Revisited

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import f1_score

# Dataset tidak seimbang
X, y = make_classification(n_samples=5000, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Distribusi kelas training: {np.bincount(y_train)}")

# Metode 1: Tanpa penanganan
rf_basic = RandomForestClassifier(random_state=42)
rf_basic.fit(X_train, y_train)
y_pred_basic = rf_basic.predict(X_test)
print(f"\nTanpa penanganan:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_basic):.4f}")
print(f"  F1 (minorty): {f1_score(y_test, y_pred_basic):.4f}")

# Metode 2: Class Weight
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)
y_pred_weighted = rf_weighted.predict(X_test)
print(f"\nClass Weight balanced:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_weighted):.4f}")
print(f"  F1 (minority): {f1_score(y_test, y_pred_weighted):.4f}")

# Metode 3: SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\nSMOTE distribusi: {np.bincount(y_train_sm)}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_sm, y_train_sm)
y_pred_smote = rf_smote.predict(X_test)
print(f"SMOTE:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_smote):.4f}")
print(f"  F1 (minority): {f1_score(y_test, y_pred_smote):.4f}")

# Metode 4: SMOTE + Undersampling
over = SMOTE(sampling_strategy=0.5, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)

pipeline_imb = ImbPipeline(steps=[
    ('over', over),
    ('under', under),
    ('model', RandomForestClassifier(random_state=42))
])
pipeline_imb.fit(X_train, y_train)
y_pred_combined = pipeline_imb.predict(X_test)
print(f"\nSMOTE + Undersampling:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_combined):.4f}")
print(f"  F1 (minority): {f1_score(y_test, y_pred_combined):.4f}")
```

---

## 6.9 Model Deployment — Dari Notebook ke Produksi

### Simpan dan Muat Model

```python
import joblib

# Simpan model
joblib.dump(full_pipeline, 'model_pipeline.pkl')

# Muat model
loaded_model = joblib.load('model_pipeline.pkl')

# Prediksi dengan model yang dimuat
new_data = pd.DataFrame({
    'age': [25, 45],
    'fare': [50.0, 120.0],
    'sex': ['male', 'female'],
    'embarked': ['S', 'C'],
    'pclass': [3, 1]
})

predictions = loaded_model.predict(new_data)
print(f"Predictions: {predictions}")

probabilities = loaded_model.predict_proba(new_data)
print(f"Probabilities:\n{probabilities}")
```

### API dengan FastAPI

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="ML Prediction API")
model = joblib.load('model_pipeline.pkl')

class PredictionRequest(BaseModel):
    age: float
    fare: float
    sex: str
    embarked: str
    pclass: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    df = pd.DataFrame([request.dict()])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return PredictionResponse(prediction=int(pred), probability=float(prob))

# Jalankan: uvicorn app:app --reload
```

### Monitoring Checklist

```
□ Data Drift: Apakah distribusi data berubah seiring waktu?
□ Concept Drift: Apakah hubungan fitur-target berubah?
□ Model Performance: Apakah metrik menurun di produksi?
□ Latency: Apakah prediksi cukup cepat?
□ Fairness: Apakah model adil untuk semua kelompok?
□ Logging: Apakah semua prediksi dicatat untuk audit?
```

---

## 6.10 Ringkasan Bab 6

1. **Ensemble Methods** menggabungkan banyak model untuk performa lebih baik:
   - **Bagging** (Random Forest): Kurangi variance, model paralel
   - **Boosting** (XGBoost, AdaBoost): Kurangi bias, model sequential
   - **Stacking**: Gabungkan model berbeda dengan meta-learner

2. **Random Forest** = Bagging + Random Feature Selection. Default yang kuat.

3. **XGBoost/LightGBM** = Gradient Boosting yang dioptimasi. Juara kompetisi ML.

4. **PCA** mereduksi dimensi sambil mempertahankan variansi.

5. **Imbalanced Data** perlu penanganan khusus: class weight, SMOTE, atau threshold adjustment.

6. **Pipeline** mengamankan proses end-to-end dan mencegah data leakage.

7. **Deployment** = menyimpan model, membuat API, dan monitoring performa produksi.

---

## Penutup: Dari Pemula Sampai Mahir

Kamu telah menyelesaikan perjalanan dari nol sampai memahami Machine Learning secara menyeluruh:

1. **Bab 1**: Apa itu ML, kenapa ada, bagaimana berbeda dari pemrograman biasa
2. **Bab 2**: Jenis-jenis ML — Supervised, Unsupervised, Reinforcement
3. **Bab 3**: Data — bahan bakar ML, cleaning, feature engineering, splitting
4. **Bab 4**: Overfitting vs Underfitting — musuh besar ML
5. **Bab 5**: Evaluasi — mengukur model dengan benar
6. **Bab 6**: Ensemble & Lanjutan — dari bagus ke luar biasa

Algoritma yang dibahas di file terpisah (01-04):
- **K-NN**: Instance-based, sederhana tapi lambat
- **Decision Tree**: Interpretable, rentan overfitting
- **Naive Bayes**: Probabilistic, cepat, bagus untuk teks
- **Neural Network**: Powerful, butuh data dan komputasi besar

Langkah selanjutnya:
- Praktikkan di proyek nyata (Kaggle, dataset publik)
- Ikuti kompetisi ML
- Pelajari Deep Learning (CNN, RNN, Transformer)
- Pelajari MLOps (deployment, monitoring, CI/CD)

Selamat belajar!