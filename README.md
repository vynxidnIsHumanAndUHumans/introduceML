# Machine Learning — Panduan Lengkap dari Nol Sampai Mahir

> *"Machine Learning adalah bidang studi yang memberi komputer kemampuan untuk belajar tanpa diprogram secara eksplisit."* — Arthur Samuel, 1959

---

## Tentang Panduan Ini

Panduan ini ditulis untuk pembaca yang **benar-benar baru** di Machine Learning. Setiap bab dibangun secara bertahap — dari dasar menuju kompleks — dengan analogi, contoh kode, dan penjelasan langkah demi langkah.

Bukan ringkasan. Bukan cheat sheet. Ini adalah **buku**.

---

## Daftar Isi

### Bagian I: Dasar-Dasar Machine Learning

| Bab | File | Topik | Deskripsi |
|-----|------|-------|-----------|
| 1 | [05-Bab1-Pendahuluan-Machine-Learning.md](05-Bab1-Pendahuluan-Machine-Learning.md) | Pendahuluan ML | Apa itu ML, kenapa ada, bagaimana bekerja, komponen, vocabulary |
| 2 | [06-Bab2-Jenis-Jenis-Machine-Learning.md](06-Bab2-Jenis-Jenis-Machine-Learning.md) | Jenis-Jenis ML | Supervised, Unsupervised, Reinforcement, Semi-supervised |

### Bagian II: Data & Persiapan

| Bab | File | Topik | Deskripsi |
|-----|------|-------|-----------|
| 3 | [07-Bab3-Data-Bahan-Bakar-ML.md](07-Bab3-Data-Bahan-Bakar-ML.md) | Data | Data cleaning, feature engineering, scaling, splitting, pipeline, data leakage |

### Bagian III: Konsep Kritis

| Bab | File | Topik | Deskripsi |
|-----|------|-------|-----------|
| 4 | [08-Bab4-Overfitting-vs-Underfitting.md](08-Bab4-Overfitting-vs-Underfitting.md) | Overfitting vs Underfitting | Bias-variance tradeoff, regularization, early stopping, diagnosis |
| 5 | [09-Bab5-Evaluasi-Model.md](09-Bab5-Evaluasi-Model.md) | Evaluasi Model | Confusion matrix, precision/recall, ROC-AUC, cross-validation |

### Bagian IV: Algoritma (Detail & Implementasi)

| Bab | File | Topik | Deskripsi |
|-----|------|-------|-----------|
| - | [01-K-Nearest-Neighbors.md](01-K-Nearest-Neighbors.md) | K-NN | Konsep, kode from scratch, contoh manual 4 & 8 data, scikit-learn, alur kerja |
| - | [02-Decision-Tree.md](02-Decision-Tree.md) | Decision Tree | Gini/Entropy, kode from scratch, contoh 5 & 10 data, pruning, scikit-learn |
| - | [03-Naive-Bayes.md](03-Naive-Bayes.md) | Naive Bayes | Teorema Bayes, Gaussian/Multinomial, contoh manual 4 & 8 data, Laplace smoothing |
| - | [04-Neural-Network.md](04-Neural-Network.md) | Neural Network | Forward/backprop, kode from scratch, XOR, activation functions, optimizers |

### Bagian V: Lanjutan & Ensemble

| Bab | File | Topik | Deskripsi |
|-----|------|-------|-----------|
| 6 | [10-Bab6-Ensemble-Methods-dan-Lanjutan.md](10-Bab6-Ensemble-Methods-dan-Lanjutan.md) | Ensemble & Lanjutan | Bagging, Boosting, Stacking, XGBoost, PCA, imbalanced data, deployment |

---

## Urutan Pembacaan yang Direkomendasikan

```
PEMULA:
  Bab 1 (Pendahuluan)
    → Bab 2 (Jenis-Jenis ML)
      → Bab 3 (Data)
        → 01-K-NN (Algoritma termudah)
          → 02-Decision-Tree
            → Bab 4 (Overfitting/Underfitting)
              → 03-Naive-Bayes
                → Bab 5 (Evaluasi)
                  → 04-Neural-Network
                    → Bab 6 (Ensemble)

INTERMEDIATE:
  Semua file, dengan fokus pada:
  - Kode from scratch di setiap algoritma
  - Bab 4 (Overfitting) dan Bab 5 (Evaluasi)
  - Bab 6 (Ensemble Methods)
```

---

## Ikhtisar Algoritma

### K-Nearest Neighbors (K-NN)

| Aspek | Detail |
|-------|--------|
| **Tipe** | Instance-based, lazy learning |
| **Cara kerja** | Hitung jarak ke semua data training, ambil K terdekat, voting |
| **Kelebihan** | Sederhana, tidak perlu training |
| **Kekurangan** | Lambat untuk data besar, sensitif scaling |
| **Kapan pakai** | Dataset kecil, baseline, rekomendasi sederhana |
| **File** | [01-K-Nearest-Neighbors.md](01-K-Nearest-Neighbors.md) |

### Decision Tree

| Aspek | Detail |
|-------|--------|
| **Tipe** | Rule-based |
| **Cara kerja** | Split data berdasarkan fitur terbaik (Gini/Entropy), buat pohon keputusan |
| **Kelebihan** | Mudah dipahami, divisualisasikan, tidak perlu scaling |
| **Kekurangan** | Rentan overfitting, instabil |
| **Kapan pakai** | Butuh interpretabilitas, data tabular |
| **File** | [02-Decision-Tree.md](02-Decision-Tree.md) |

### Naive Bayes

| Aspek | Detail |
|-------|--------|
| **Tipe** | Probabilistic |
| **Cara kerja** | Hitung probabilitas setiap kelas, asumsi fitur independen |
| **Kelebihan** | Cepat, efektif untuk teks, butuh sedikit data |
| **Kekurangan** | Asumsi independensi sering tidak realistis |
| **Kapan pakai** | Text classification, spam filter, baseline |
| **File** | [03-Naive-Bayes.md](03-Naive-Bayes.md) |

### Neural Network

| Aspek | Detail |
|-------|--------|
| **Tipe** | Connectionist |
| **Cara kerja** | Forward pass + backpropagation, weighted sum + activation function |
| **Kelebihan** | Sangat powerful, bisa menangkap pola kompleks |
| **Kekurangan** | Butuh banyak data, komputasi berat, black box |
| **Kapan pakai** | Image, text, audio, pola kompleks |
| **File** | [04-Neural-Network.md](04-Neural-Network.md) |

---

## Quick Reference: Pilih Algoritma

```
Punya label (jawaban)?
│
├── YA → SUPERVISED
│   │
│   ├── Output kategori (klasifikasi)?
│   │   ├── Data sedikit, sederhana → K-NN, Naive Bayes
│   │   ├── Butuh interpretabilitas → Decision Tree
│   │   ├── Data tabular, performa tinggi → Random Forest, XGBoost
│   │   ├── Data gambar/audio → CNN (Neural Network)
│   │   └── Data teks → Naive Bayes, Transformer
│   │
│   └── Output angka (regresi)?
│       ├── Linear sederhana → Linear Regression
│       ├── Non-linear → Decision Tree Regressor, Random Forest
│       └── Pola kompleks → Neural Network
│
└── TIDAK → UNSUPERVISED
    ├── Mau cari kelompok? → K-Means, DBSCAN
    ├── Mau sederhanakan data? → PCA, t-SNE
    └── Mau cari anomali? → Isolation Forest, One-Class SVM
```

---

## Prasyarat

1. **Python** (3.8+) — Bahasa utama
2. **NumPy** — Operasi numerik
3. **Pandas** — Manipulasi data
4. **Scikit-learn** — Algoritma ML klasik
5. **Matplotlib** — Visualisasi

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm imbalanced-learn  # Opsional
```

---

## Referensi Lanjutan

1. **Book**: "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" — Aurelien Geron
2. **Book**: "Pattern Recognition and Machine Learning" — Christopher Bishop
3. **Book**: "The Elements of Statistical Learning" — Hastie, Tibshirani, Friedman
4. **Course**: Andrew Ng's Machine Learning (Coursera)
5. **Course**: fast.ai Practical Deep Learning for Coders
6. **Kaggle**: Platform kompetisi dan dataset ML

---

*Ditulis sebagai panduan belajar lengkap. Setiap bab dapat dibaca secara mandiri, tapi urutan direkomendasikan untuk pemula.*
