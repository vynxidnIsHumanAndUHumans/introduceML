# Bab 2: Jenis-Jenis Machine Learning — Peta Jalan Paradigma

---

## Kenapa Kamu Perlu Tahu Jenis-Jenis ML?

Bayangkan kamu seorang dokter yang punya berbagai alat: stetoskop, termometer, X-ray, dan MRI. Setiap alat punya kegunaan berbeda. Kamu tidak akan pakai MRI untuk mengukur suhu tubuh, kan?

Begitu juga Machine Learning. Ada berbagai **paradigma** — cara berbeda untuk membuat mesin belajar. Masing-masing cocok untuk masalah yang berbeda. Memahami peta ini sejak awal akan membantumu:

1. **Memilih algoritma yang tepat** untuk masalahmu
2. **Tidak memaksakan alat yang salah** pada masalah yang tidak cocok
3. **Memahami literatur dan paper** dengan terminologi yang benar

Di bab ini, kita akan membangun pemahamanmu secara bertahap, dari yang paling umum sampai yang paling khusus.

---

## 2.1 Tiga Paradigma Besar

```
              MACHINE LEARNING
                     │
     ┌───────────────┼───────────────┐
     │               │               │
 SUPERVISED     UNSUPERVISED    REINFORCEMENT
  LEARNING       LEARNING        LEARNING
     │               │               │
  Punya label    Tanpa label    Belajar dari
  (jawaban)      (cari pola)    reward/hukuman
```

| Paradigma | Data | Tujuan | Analogi |
|-----------|------|--------|---------|
| **Supervised** | Input + Label | Prediksi label baru | Belajar dengan kunci jawaban |
| **Unsupervised** | Input saja | Temukan pola tersembunyi | Eksplorasi tanpa peta |
| **Reinforcement** | Input + Reward | Maksimalkan reward kumulatif | Belajar dari coba-coba |

Mari kita bahas satu per satu, mendalam.

---

## 2.2 Supervised Learning — Belajar dengan Guru

### Cerita Pembuka

Kamu seorang anak SD yang belajar mengenal hewan. Gurumu menunjukkan gambar dan berkata:

> "Ini kucing." (menunjukkan gambar kucing)
> "Ini anjing." (menunjukkan gambar anjing)
> "Ini kucing." (menunjukkan gambar kucing lain)
> "Ini anjing." (menunjukkan gambar anjing lain)

Setelah ratusan contoh, kamu mulai mengenali pola: kucing punya telinga runcing, mata besar, moncong kecil. Anjing punya moncong panjang, telinga floppy.

Kemudian gurumu menunjukkan gambar baru yang belum pernah kamu lihat:

> "Ini apa?"

Kamu menjawab: "Kucing!" — karena fitur-fiturnya cocok dengan pola yang sudah kamu pelajari.

**Itulah Supervised Learning.** Kamu belajar dari data yang sudah punya label (jawaban), lalu memprediksi label untuk data baru.

### Mengapa "Supervised"?

Karena ada **supervisor** (guru/pembimbing) yang memberikan jawaban yang benar. Model tahu mana prediksi yang benar dan mana yang salah, karena label (ground truth) tersedia.

### Tugas Supervised Learning

```
SUPERVISED LEARNING
         │
    ┌────┴────┐
    │         │
KLASIFIKASI  REGRESI
 (kategori)  (angka)
```

#### Klasifikasi — Menebak Kategori

Output berupa **kategori/diskrit**:

| Masalah | Input | Output | Jumlah Kelas |
|---------|-------|--------|-------------|
| Deteksi spam | Teks email | Spam/Bukan | 2 (Binary) |
| Pengenalan digit | Gambar 28×28 | 0-9 | 10 (Multi-class) |
| Diagnosa penyakit | Gejala, lab results | Nama penyakit | Multi-class |
| Sentimen review | Teks review | Positif/Netral/Negatif | 3 (Multi-class) |
| Deteksi fraud | Transaksi | Fraud/Valid | 2 (Binary) |

**Binary Classification** — Dua kelas:

```
Email masuk ──→ Model ──→ SPAM atau BUKAN SPAM?
Transaksi  ──→ Model ──→ FRAUD atau VALID?
X-ray      ──→ Model ──→ KANKER atau SEHAT?
```

**Multi-class Classification** — Lebih dari dua kelas:

```
Gambar buah ──→ Model ──→ APEL? PISANG? JERUK? MANGGA?
Teks berita  ──→ Model ──→ OLAHRAGA? POLITIK? EKONOMI? HIBURAN?
Suara       ──→ Model ──→ "HALO"? "YA"? "TIDAK"? "BANTUAN"?
```

#### Regresi — Menebak Angka

Output berupa **angka kontinu**:

| Masalah | Input | Output |
|---------|-------|--------|
| Prediksi harga rumah | Luas, lokasi, kamar | Rp 850.000.000 |
| Prediksi suhu | Kelembapan, tekanan, angin | 27.5°C |
| Prediksi penjualan | Bulan, promosi, harga | 15.300 unit |
| Prediksi umur | Foto wajah | 34 tahun |

**Perbedaan visual:**

```
KLASIFIKASI                    REGRESI
                               
  |  ●  ●        ●             |        ●
  |     ●   ●       ●          |     ●
y |                  ●         |  ●
  |        ●     ●             |●
  |  ●                          |   ●
  +----------→ x               +----------→ x
  (Output: kategori)            (Output: garis kontinu)
```

#### Contoh Kode: Klasifikasi vs Regresi

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import numpy as np

# ====== KLASIFIKASI ======
# Prediksi: Apakah mahasiswa lulus? (Ya/Tidak)
X_class = np.array([
    [50, 60],  # belajar 50 jam, nilai UKK 60 → Tidak lulus
    [70, 75],  # belajar 70 jam, nilai UKK 75 → Lulus
    [80, 85],  # belajar 80 jam, nilai UKK 85 → Lulus
    [40, 45],  # belajar 40 jam, nilai UKK 45 → Tidak lulus
    [90, 90],  # belajar 90 jam, nilai UKK 90 → Lulus
])
y_class = ['Tidak Lulus', 'Lulus', 'Lulus', 'Tidak Lulus', 'Lulus']

model_clf = KNeighborsClassifier(n_neighbors=3)
model_clf.fit(X_class, y_class)

mahasiswa_baru = np.array([[65, 70]])  # belajar 65 jam, UKK 70
prediksi = model_clf.predict(mahasiswa_baru)
print(f"Prediksi kelulusan: {prediksi[0]}")
# Output: "Lulus" atau "Tidak Lulus" ← KATEGORI, bukan angka

# ====== REGRESI ======
# Prediksi: Berapa nilai akhir mahasiswa? (angka 0-100)
X_reg = np.array([
    [50],  # belajar 50 jam
    [70],  # belajar 70 jam
    [80],  # belajar 80 jam
    [40],  # belajar 40 jam
    [90],  # belajar 90 jam
])
y_reg = np.array([60, 75, 85, 45, 92])  # nilai akhir (angka kontinu)

model_reg = KNeighborsRegressor(n_neighbors=3)
model_reg.fit(X_reg, y_reg)

mahasiswa_baru = np.array([[65]])  # belajar 65 jam
prediksi_nilai = model_reg.predict(mahasiswa_baru)
print(f"Prediksi nilai: {prediksi_nilai[0]:.1f}")
# Output: 73.3 ← ANGKA, bukan kategori
```

### Algoritma Supervised Learning (Map)

```
SUPERVISED LEARNING ALGORITHMS
         │
    ┌────┼────┐
    │    │    │
KLASIFIKASI   │  REGRESI
    │    │    │
    │    │    ├── Linear Regression
    │    │    ├── Ridge / Lasso
    │    │    ├── Polynomial Regression
    │    │    ├── SVR
    │    │    ├── Decision Tree Regressor
    │    │    ├── KNN Regressor
    │    │    └── Neural Network Regressor
    │    │
    ├── Logistic Regression ─── (walaupun namanya "regression", ini klasifikasi!)
    ├── K-Nearest Neighbors
    ├── Decision Tree / Random Forest
    ├── Naive Bayes
    ├── Support Vector Machine (SVM)
    ├── Neural Network
    └── Gradient Boosting (XGBoost, LightGBM)
```

### Bagaimana Memilih Algoritma?

Tidak ada algoritma yang selalu terbaik (**No Free Lunch Theorem**). Tapi ada panduan praktis:

| Kondisi | Rekomendasi | Kenapa? |
|---------|------------|---------|
| Baseline / awal eksplorasi | Logistic Regression / Decision Tree | Sederhana, cepat, interpretable |
| Data sedikit (< 1000) | K-NN, Naive Bayes, SVM | Tidak butuh banyak data |
| Data banyak (> 100K) | Neural Network, XGBoost | Bisa menangkap pola kompleks |
| Butuh interpretabilitas | Decision Tree, Logistic Regression | Bisa dijelaskan ke stakeholder |
| Data teks | Naive Bayes, SVM, Transformer | Dirancang untuk data text |
| Data gambar | CNN (Neural Network) | Cara terbaik untuk image |
| Performa terbaik | XGBoost, LightGBM, Ensemble | Juara kompetisi ML |
| Data tabular sederhana | Random Forest / XGBoost | Default yang kuat |

---

## 2.3 Unsupervised Learning — Belajar Tanpa Guru

### Cerita Pembuka

Sekarang bayangkan kamu diberi sekantong batu berwarna-warni. Tidak ada yang memberitahumu nama batu-batu itu. Tidak ada label. Tapi kamu bisa **melihat pola**: batu merah mirip satu sama lain, batu biru juga mirip, batu hijau juga.

Kamu secara natural mengelompokkannya: *"Aku rasa batu-batu ini bisa dikategorikan jadi 3 kelompok berdasarkan warna."*

**Itulah Unsupervised Learning.** Tanpa label, tanpa jawaban yang benar — cuma kamu dan data yang penuh pola.

### Mengapa "Unsupervised"?

Karena **tidak ada supervisor** — tidak ada yang memberikan jawaban. Model harus **menemukan struktur tersembunyi** dalam data sendiri.

### Tugas Unsupervised Learning

```
UNSUPERVISED LEARNING
         │
    ┌────┼────────────┐
    │    │            │
CLUSTERING  DIMENSIONALITY  ANOMALY
            REDUCTION      DETECTION
```

#### 1. Clustering — Mengelompokkan Data

> *"Kelompokkan data yang mirip menjadi kelompok-kelompok."*

| Contoh | Input | Output |
|--------|-------|--------|
| Segmentasi pelanggan | Data belanja | Kelompok A (loyal), B (jarang), C (baru) |
| Pengelompokan berita | Artikel | Topik 1, Topik 2, Topik 3 |
| Analisis gen | Data gen | Kelompok gen yang berfungsi sama |
| Kompresi warna | Gambar RGB | 16 warna representatif |

**Algoritma Clustering:**

| Algoritma | Cara Kerja | Cocok Untuk |
|-----------|-----------|-------------|
| K-Means | Bagi data jadi K kelompok berdasarkan jarak ke centroid | Cluster bulat, jumlah cluster diketahui |
| DBSCAN | Kelompokkan berdasarkan kepadatan | Cluster bentuk aneh, deteksi outlier |
| Hierarchical | Bangun hirarki cluster | Jumlah cluster tidak diketahui |
| Gaussian Mixture | Asumsi data campuran distribusi Gaussian | Cluster yang overlap |

**Contoh Kode: K-Means Clustering**

```python
from sklearn.cluster import KMeans
import numpy as np

# Data: tinggi dan berat 12 orang
data = np.array([
    [160, 50],  # Orang dengan postur kecil
    [155, 48],
    [162, 52],
    [158, 49],
    [175, 75],  # Orang dengan postur besar
    [180, 80],
    [172, 72],
    [178, 78],
    [145, 35],  # Anak-anak
    [140, 32],
    [148, 38],
    [142, 34],
])

# K-Means: bagi jadi 3 kelompok
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(data)

print("Cluster labels:", kmeans.labels_)
print("Centroids:\n", kmeans.cluster_centers_)

# Prediksi kelompok untuk data baru
orang_baru = np.array([[165, 55]])
kelompok = kmeans.predict(orang_baru)
print(f"Orang baru masuk cluster: {kelompok[0]}")
```

**K-Means Step-by-Step (Visual):**

```
LANGKAH 1: Inisialisasi — Pilih 3 centroid secara acak

  Langkah awal:
  ● ●              C1 = (150, 40)
      ● ●  ●       C2 = (170, 70)
          ● ●  ●   C3 = (160, 60)
  ● ● ●

LANGKAH 2: Assign — Kelompokkan setiap titik ke centroid terdekat

  ● ● = Cluster 1 (dekat C1)
      ● ●  ● = Cluster 2 (dekat C2)
                   = Cluster 3 (dekat C3)
  ● ● ●

LANGKAH 3: Update — Pindahkan centroid ke rata-rata titik di kelompoknya

  C1_baru = mean(titik di Cluster 1)
  C2_baru = mean(titik di Cluster 2)
  C3_baru = mean(titik di Cluster 3)

LANGKAH 4: Ulangi langkah 2 dan 3 sampai konvergen
  (centroid tidak bergerak lagi atau perubahan sangat kecil)
```

#### 2. Dimensionality Reduction — Menyederhanakan Data

> *"Sederhanakan data dari 100 dimensi jadi 2 dimensi, tanpa kehilangan terlalu banyak informasi."*

Bayangkan kamu punya data dengan 500 fitur. Kebanyakan fitur mungkin tidak penting atau saling berkorelasi. Dimensionality reduction membantu:

- **Visualisasi**: Plot data 500D ke 2D
- **Percepat komputasi**: Kurangi dari 500 → 50 fitur
- **Hapus noise**: Buang informasi tidak penting
- **Hindari curse of dimensionality**

| Algoritma | Cara Kerja | Cocok Untuk |
|-----------|-----------|-------------|
| PCA (Principal Component Analysis) | Temukan arah variansi terbesar | Reduksi linear, visualisasi |
| t-SNE | Preservasi struktur lokal | Visualisasi 2D/3D |
| UMAP | Preservasi struktur lokal & global | Visualisasi, clustering |
| Autoencoder | Neural network compression | Non-linear reduction |

**Contoh: PCA**

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # 4 fitur: sepal length, sepal width, petal length, petal width
print(f"Dimensi asli: {X.shape}")  # (150, 4) — 4 dimensi

# Reduksi ke 2 dimensi untuk visualisasi
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Dimensi setelah PCA: {X_reduced.shape}")  # (150, 2) — 2 dimensi

# Berapa banyak informasi yang dipertahankan?
print(f"Variansi yang dipertahankan: {pca.explained_variance_ratio_}")
print(f"Total: {sum(pca.explained_variance_ratio_):.2%}")
# Output: ~97.8% informasi dipertahankan dengan hanya 2 dimensi!

# Visualisasi
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
for target, color in zip([0, 1, 2], ['red', 'green', 'blue']):
    plt.scatter(X_reduced[iris.target == target, 0],
                X_reduced[iris.target == target, 1],
                c=color, label=iris.target_names[target])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Iris Dataset')
plt.legend()
plt.savefig('pca_iris.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### 3. Anomaly Detection — Mendeteksi yang Aneh

> *"Temukan data yang berbeda jauh dari yang lain."*

| Contoh | Input | Output |
|--------|-------|--------|
| Deteksi fraud | Transaksi kartu kredit | Normal / Mencurigakan |
| Deteksi intrusi | Log jaringan | Normal / Serangan |
| Deteksi penyakit langka | Data medis | Normal / Abnormal |
| Pemeliharaan mesin | Sensor IoT | Normal / Akan rusak |

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Data normal: suhu mesin 20-30°C
data_normal = np.random.normal(25, 2, 100).reshape(-1, 1)

# Ada beberapa anomali: suhu 50-60°C (mesin terlalu panas!)
data_anomali = np.array([[50], [55], [60], [48], [52]]).reshape(-1, 1)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data_normal)

# Cek data baru
data_uji = np.array([[24], [26], [55], [23], [58]]).reshape(-1, 1)
hasil = model.predict(data_uji)  # 1 = normal, -1 = anomali

for suhu, pred in zip(data_uji.flatten(), hasil):
    status = "NORMAL" if pred == 1 else "ANOMALI!"
    print(f"Suhu {suhu}°C → {status}")
```

### Perbedaan Supervised vs Unsupervised

| Aspek | Supervised | Unsupervised |
|-------|-----------|--------------|
| Data | Punya label (y) | Tidak punya label |
| Tujuan | Prediksi y untuk data baru | Temukan pola tersembunyi |
| Evaluasi | Akurasi, precision, recall | Silhouette score, visualisasi |
| Contoh | Spam detection, harga rumah | Segmentasi pelanggan, deteksi anomali |
| Kesulitan | Butuh data berlabel (mahal!) | Hasil bisa subjektif |
| Ground truth | Ada | Tidak ada (siapa yang menentukan benar/salah?) |

---

## 2.4 Semi-Supervised Learning — Setengah Guru

### Cerita

Seorang guru medis punya 10,000 X-ray, tapi hanya 100 yang sudah diberi label oleh dokter spesialis (lengkap dengan diagnosa). Memberi label itu **mahal** — butuh waktu dan keahlian.

Bisa tidak model belajar dari 100 data berlabel DAN 9,900 data tanpa label?

**Bisa!** Itulah Semi-Supervised Learning.

### Cara Kerja

```
1. Training awal dengan 100 data berlabel → Model awal (jelek)
2. Prediksi label untuk 9,900 data tak berlabel → Pseudo-label
3. Pilih prediksi paling percaya diri → Tambahkan ke data berlabel
4. Re-training dengan data berlabel yang diperluas → Model lebih baik
5. Ulangi langkah 2-4 → Model makin baik
```

### Contoh Nyata

- **Google Photos**: Kamu label beberapa foto sebagai "kucing Google", lalu sistem otomatis mengenali semua foto kucing lainnya
- **Medical imaging**: Sedikit data berlabel dari dokter, banyak data tidak berlabel dari rumah sakit
- **Text classification**: Sebagian artikel berlabel, sebagian tidak

```python
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier

# 100 data berlabel, 9900 data tidak berlabel (label = -1)
X_labeled = X_train[:100]
y_labeled = y_train[:100]
X_unlabeled = X_train[100:]
y_unlabeled = np.full(len(X_unlabeled), -1)  # -1 = tidak berlabel

X_semi = np.vstack([X_labeled, X_unlabeled])
y_semi = np.hstack([y_labeled, y_unlabeled])

base_classifier = RandomForestClassifier(n_estimators=50)
self_training = SelfTrainingClassifier(base_classifier, threshold=0.9)
self_training.fit(X_semi, y_semi)

# Model sekarang belajar dari 100 data berlabel + 9900 data pseudo-label
```

---

## 2.5 Reinforcement Learning — Belajar dari Hadiah dan Hukuman

### Cerita Pembuka

Bayangkan kamu belajar main catur. Tidak ada yang mengajarimu aturan langkah terbaik. Yang kamu tahu:

- Kalau kamu menang → kamu dapat **hadiah** (reward +1)
- Kalau kamu kalah → kamu dapat **hukuman** (reward -1)
- Setiap langkah bisa membawa lebih dekat ke menang atau kalah

Awalnya, kamu coba-coba secara acak. Kadang menang, sering kalah. Tapi secara perlahan, kamu menyadari:

> *"Oh, kalau aku pindahkan ksatria ke sini, aku bisa menangkap ratu!"*
> *"Ternyata, mengorbankan pion di awal sering berujung kalah."*

Kamu **belajar dari pengalaman** — berdasarkan reward yang kamu terima setiap langkah. Tidak ada yang memberitahumu langkah mana yang benar, tapi kamu belajar dari konsekuensi tindakanmu.

**Itulah Reinforcement Learning (RL).**

### Komponen RL

```
┌─────────────────────────────────────────┐
│                                         │
│   ┌──────────┐   Action   ┌──────────┐  │
│   │          │────────────→│          │  │
│   │  AGENT   │             │  ENV     │  │
│   │(Pemain)  │←────────────│(Dunia)   │  │
│   │          │  State +     │          │  │
│   └──────────┘  Reward      └──────────┘  │
│                                         │
│   Agent: Entitas yang belajar           │
│   Environment: Dunia tempat agent berada│
│   State: Situasi saat ini (s)           │
│   Action: Tindakan yang dipilih (a)     │
│   Reward: Hadiah/hukuman (r)           │
│   Policy: Strategi agent π(s) → a      │
│                                         │
└─────────────────────────────────────────┘

Siklus: State → Action → Reward → New State → Action → ...
```

**Contoh konkret:**

| Elemen | Chess | Pac-Man | Mobil Otonom |
|--------|-------|---------|--------------|
| Agent | Pemain catur | Karakter Pac-Man | Sistem AI mobil |
| Environment | Papan catur | Labyrinth | Jalan raya |
| State | Posisi semua bidak | Posisi Pac-Man, hantu, titik | Posisi mobil, jalan, mobil lain |
| Action | Pindahkan bidak | Gerak atas/bawah/kiri/kanan | Belok, gas, rem |
| Reward | +1 menang, -1 kalah | +10 makan titik, -500 mati | +1 aman, -100 tabrakan |
| Policy | Strategi bermain | Rute yang diambil | Kebijakan mengemudi |

### Jenis-Jenis RL

| Tipe | Penjelasan | Contoh |
|------|-----------|--------|
| Model-based | Agent tahu aturan environment | Chess (aturan jelas) |
| Model-free | Agent tidak tahu aturan, belajar dari pengalaman | Robot di dunia nyata |
| On-policy | Belajar dari policy yang sedang dijalankan | SARSA |
| Off-policy | Belajar dari policy lain (replay buffer) | Q-Learning, DQN |

### Contoh Kode: Q-Learning Sederhana

```python
import numpy as np

# Environment sederhana: Grid 4x4
# Agent harus dari START (0,0) ke GOAL (3,3)
# 0=atas, 1=kanan, 2=bawah, 3=kiri

SIZE = 4
N_ACTIONS = 4
REWARD_GOAL = 100
REWARD_STEP = -1  # Setiap langkah kena penalty (ingin efisien)

# Q-Table: menyimpan "nilai" setiap (state, action)
Q = np.zeros((SIZE * SIZE, N_ACTIONS))

# Hyperparameters
alpha = 0.1    # Learning rate
gamma = 0.95   # Discount factor (berapa jauh ke depan agent mempertimbangkan)
epsilon = 1.0  # Exploration rate (mulai dari 1.0, turun perlahan)
epsilon_min = 0.01
epsilon_decay = 0.995

def state_to_idx(row, col):
    return row * SIZE + col

def step(row, col, action):
    """Eksekusi aksi, kembali ke state baru dan reward"""
    if action == 0:    row = max(0, row - 1)       # atas
    elif action == 1:  col = min(SIZE - 1, col + 1) # kanan
    elif action == 2:  row = min(SIZE - 1, row + 1) # bawah
    elif action == 3:  col = max(0, col - 1)        # kiri

    if row == SIZE - 1 and col == SIZE - 1:  # goal!
        return row, col, REWARD_GOAL, True
    return row, col, REWARD_STEP, False

# Training
for episode in range(1000):
    row, col = 0, 0  # Start
    done = False

    while not done:
        state = state_to_idx(row, col)

        # Epsilon-greedy: explore vs exploit
        if np.random.random() < epsilon:
            action = np.random.randint(N_ACTIONS)  # Explore: acak
        else:
            action = np.argmax(Q[state])           # Exploit: terbaik

        next_row, next_col, reward, done = step(row, col, action)
        next_state = state_to_idx(next_row, next_col)

        # Q-Learning update:
        # Q(s,a) = Q(s,a) + α × [r + γ × max_a'Q(s',a') - Q(s,a)]
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next * (1 - done) - Q[state, action])

        row, col = next_row, next_col

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Test: agent yang sudah belajar
print("Policy yang dipelajari:")
arrows = ['↑', '→', '↓', '←']
for r in range(SIZE):
    row_str = ''
    for c in range(SIZE):
        if r == SIZE-1 and c == SIZE-1:
            row_str += '  ★  '
        else:
            state = state_to_idx(r, c)
            row_str += f'  {arrows[np.argmax(Q[state])]}  '
    print(row_str)

# Output yang diharapkan:
#   →  →  ↓  ↓
#   →  ↓  ↓  ↓
#   →  ↓  ↓  ↓
#   →  ↓  →  ★
```

### Perbandingan Seluruh Paradigma

| Aspek | Supervised | Unsupervised | Reinforcement |
|-------|-----------|--------------|--------------|
| **Data** | Input + Label | Input saja | Input + Reward |
| **Tujuan** | Prediksi label | Temukan pola | Maksimalkan reward |
| **Evaluasi** | Akurasi | Silhouette, visualisasi | Cumulative reward |
| **Interaksi** | Pasif (data statis) | Pasif (data statis) | Aktif (agent berinteraksi) |
| **Contoh** | Spam filter | Segmentasi pelanggan | Game AI, robot |
| **Kesulitan** | Butuh data berlabel | Evaluasi subjektif | Reward sparse, training sulit |
| **Kebutuhan data** | Sedang-sangat banyak | Banyak | Banyak (dari simulasi) |

---

## 2.6 Variasi Lainnya (Sekilas)

Selain tiga paradigma utama, ada beberapa variasi penting:

### Self-Supervised Learning

> *"Buat label sendiri dari data."*

Tidak punya label? Buat sendiri! Contoh:
- **Masked Language Model**: Sembunyikan kata, prediksi kata yang disembunyikan (BERT)
- **Next Sentence Prediction**: Prediksi apakah kalimat B mengikuti kalimat A
- **Contrastive Learning**: Pilih mana dua gambar yang mirip

```python
# Analogi sederhana self-supervised
# Tugas: prediksi kata berikutnya

kalimat = "Saya pergi ke pasar untuk membeli sayur"
# Input:  "Saya pergi ke pasar untuk membeli"
# Target: "pergi ke pasar untuk membeli sayur"
#            ↑ Model belajar dari KONTeks, bukan label manusia
```

### Transfer Learning

> *"Kamu sudah belajar matematika? Gunakan itu untuk belajar fisika."*

Model yang dilatih di tugas A, digunakan (dan disesuaikan sedikit) untuk tugas B.

```
Pre-training (tugas besar):  Belajar dari miliaran gambar → Model yang mengerti "bentuk"
Fine-tuning (tugas spesifik): Sesuaikan sedikit untuk mengenali tumor → Model diagnosis
```

### Online Learning

> *"Belajar dari data yang datang satu per satu, secara terus-menerus."*

Tidak seperti batch learning (latih sekali, pakai terus), online learning terus memperbarui model saat data baru datang.

```
Data 1 → Update model
Data 2 → Update model
Data 3 → Update model
... (tanpa batas)
```

Cocok untuk: sistem rekomendasi, deteksi fraud, analisis streaming.

### Federated Learning

> *"Belajar dari data yang tidak boleh dibagi."`

```
Phone A → Training lokal → Gradient → Server (aggregate gradients)
Phone B → Training lokal → Gradient → Server (aggregate gradients)
Phone C → Training lokal → Gradient → Server (aggregate gradients)
                                         ↓
                               Model yang diperbarui → Dikirim balik ke semua phone
```

Data tetap di device masing-masing. Server hanya menerima gradient, bukan data mentah. Ini menjaga privasi.

---

## 2.7 Bagaimana Memilih Paradigma?

### Decision Tree untuk Memilih Paradigma

```
Punya label (jawaban)?
│
├── YA → SUPERVISED LEARNING
│   │
│   ├── Output berupa kategori?
│   │   └── YA → KLASIFIKASI
│   │       └── Contoh: spam/not spam, pengenalan gambar, diagnosa
│   │
│   └── Output berupa angka kontinu?
│       └── YA → REGRESI
│           └── Contoh: harga rumah, suhu, penjualan
│
└── TIDAK → Punya reward/feedback?
    │
    ├── YA → REINFORCEMENT LEARNING
    │   └── Contoh: game AI, robot, mobil otonom
    │
    └── TIDAK → UNSUPERVISED LEARNING
        │
        ├── Mau cari kelompok?
        │   └── CLUSTERING
        │       └── Contoh: segmentasi pelanggan, grup berita
        │
        ├── Mau sederhanakan data?
        │   └── DIMENSIONALITY REDUCTION
        │       └── Contoh: visualisasi, kompresi, preprocessing
        │
        └── Mau cari hal aneh?
            └── ANOMALY DETECTION
                └── Contoh: deteksi fraud, intrusi, kerusakan mesin
```

### Contoh Nyata per Paradigma

| Masalah Nyata | Paradigma | Mengapa? |
|---------------|-----------|----------|
| Gmail memfilter spam | Supervised | Punya label (spam/bukan) dari jutaan email yang ditandai user |
| Netflix merekomendasikan film | Semi-supervised / Unsupervised | Sedikit rating eksplisit, banyak behavior implisit |
| AlphaGo bermain Go | Reinforcement | Belajar dari hasil menang/kalah, bukan dari dataset berlabel |
| Segmentasi pelanggan Shopify | Unsupervised | Tidak ada label "tipe pelanggan" — harus ditemukan sendiri |
| Tesla Autopilot | Reinforcement + Supervised | Kombinasi: supervised untuk deteksi objek, RL untuk keputusan mengemudi |
| ChatGPT memahami bahasa | Self-supervised | Model prediksi kata berikutnya (label dibuat dari teks sendiri) |

---

## 2.8 Kesalahan Umum dalam Memilih Paradigma

### Kesalahan 1: "Pakai Deep Learning untuk Semua Masalah"

Tidak. Deep Learning itu seperti membunuh lalat dengan bazooka.

| Masalah | Solusi Overkill | Solusi Tepat |
|---------|----------------|-------------|
| Prediksi harga rumah di kota kecil (100 data) | Neural Network 10 layer | Linear Regression |
| Klasifikasi teks pendek (spam) | Transformer XL | Naive Bayes |
| Segmentasi 3 kelompok pelanggan | Autoencoder + GMM | K-Means |

### Kesalahan 2: "Butuh Label Tapi Tidak Punya, Jadi Pakai Unsupervised"

Kalau kamu butuh label tapi tidak punya, jangan langsung lompat ke unsupervised. Pertimbangkan:

1. **Label sebagian data secara manual** (semi-supervised)
2. **Gunakan heuristic/aturan sederhana** sebagai proxy label
3. **Pre-trained model** untuk pseudo-labeling
4. **Active learning** — model pilih data mana yang paling perlu dilabeli

### Kesalahan 3: "Reinforcement Learning Tersedia untuk Masalah Terbatas"

RL sangat powerful tapi juga sangat sulit. Jangan pakai RL kalau masalahmu bisa diselesaikan dengan supervised learning.

| Kondisi | Pakai RL? | Solusi Alternatif |
|---------|-----------|-------------------|
| Punya data berlabel lengkap | ❌ | Supervised learning |
| Butuh keputusan real-time tapi data simulasi ada | ⚠️ | Coba supervised dulu |
| Environment dinamis, interaksi perlu | ✅ | RL cocok |

---

## 2.9 Ringkasan Bab 2

1. **Supervised Learning** = Belajar dengan guru (data berlabel)
   - Klasifikasi → output kategori
   - Regresi → output angka

2. **Unsupervised Learning** = Belajar tanpa guru (data tanpa label)
   - Clustering → cari kelompok
   - Dimensionality Reduction → sederhanakan data
   - Anomaly Detection → cari yang aneh

3. **Reinforcement Learning** = Belajar dari reward (hadiah/hukuman)
   - Agent berinteraksi dengan environment
   - Belajar policy yang memaksimalkan reward kumulatif

4. **Variasi**: Semi-supervised, Self-supervised, Transfer Learning, Online Learning, Federated Learning

5. **Pilih paradigma berdasarkan data yang kamu punya**: punya label? punya reward? tidak punya keduanya?

6. **Mulailah dari yang sederhana** — jangan langsung lompat ke Deep Learning.

---

**Selanjutnya → Bab 3: Data — Bahan Bakar Machine Learning**

Data adalah fondasi. Model terbaik di dunia tidak bisa berbuat apa-apa tanpa data yang baik. Di bab berikutnya, kita akan mendalami bagaimana menyiapkan data yang tepat untuk ML.