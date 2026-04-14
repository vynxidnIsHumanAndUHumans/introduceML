# Bab 1: Pendahuluan Machine Learning — Dari Nol Sampai Paham

---

## Kenapa Kamu Perlu Membaca Bab Ini?

Katakanlah kamu seorang dokter. Setiap hari, puluhan pasien datang ke klinikmu. Kamu memeriksa gejala mereka, mengambil keputusan, dan memberi resep. Tapi satu hari, ada 500 pasien yang datang sekaligus. Kamu tidak mungkin memeriksa satu-satu. Lalu kau berpikir: *"Bagaimana kalau ada asisten yang bisa memprediksi penyakit berdasarkan gejala — tanpa aku harus memeriksa setiap pasien secara manual?"*

Itulah inti **Machine Learning**.

Bab ini akan membawamu dari nol — benar-benar nol — sampai kamu memahami apa itu Machine Learning, kenapa itu ada, bagaimana cara kerjanya, dan apa yang membedakannya dari pemrograman biasa. Kita tidak akan melompat-lompat. Kita akan membangun pemahamanmu layaknya membangun rumah: fondasi dulu, baru dinding, baru atap.

---

## 1.1 Dunia Sebelum Machine Learning

### Pemrograman Tradisional: Resep Masakan yang Kaku

Bayangkan kamu mau membuat program yang bisa membedakan email spam dan email biasa. Dengan pemrograman tradisional, kamu akan menulis **aturan eksplisit**:

```python
def is_spam(email):
    if "beli sekarang" in email:
        return True
    if "diskon 90%" in email:
        return True
    if "klik di sini" in email:
        return True
    if email.sender in blacklist:
        return True
    return False
```

Masalahnya? Daftar aturan itu **tidak pernah cukup**. Spammer terus menemukan cara baru:
- "Beli s3karang" (ganti huruf)
- "D1skon 90 persen" (typo sengaja)
- "Klik d1 s1n1" (obfuscation)

Kamu akan terus menulis aturan baru, tapi spammer selalu satu langkah di depan. Pemrograman tradisionan itu seperti menulis **buku resep yang kaku** — kalau bahannya berubah, resepnya harus ditulis ulang.

### Masalah Dunia Nyata yang Tidak Bisa Diatur dengan IF-ELSE

| Masalah | Kenapa IF-ELSE Gagal? |
|---------|----------------------|
| Pengenalan wajah | Miliaran variasi cahaya, sudut, ekspresi |
| Terjemahan bahasa | Konteks, idiom, ambigu |
| Prediksi saham | Ribuan variabel, noise, kompleksitas |
| Mobil otonom | Jutaan skenario jalan yang berbeda |
| Rekomendasi film | Selera manusia tidak bisa diresepkan |

Untuk masalah seperti ini, kita butuh pendekatan yang **berbeda secara fundamental**.

---

## 1.2 Lalu, Apa Itu Machine Learning?

### Definisi yang Seharusnya Kamu Ingat

**Machine Learning** adalah bidang ilmu yang membuat komputer bisa **belajar dari data**, tanpa perlu diprogram secara eksplisit untuk setiap aturan.

Arthur Samuel, salah satu pionir ML, mendefinisikannya pada tahun 1959:

> *"Machine Learning adalah bidang studi yang memberi komputer kemampuan untuk belajar tanpa diprogram secara eksplisit."*

Tom Mitchell, profesor ML terkenal, memberi definisi yang lebih teknis:

> *"Sebuah program komputer dikatakan belajar dari pengalaman E terhadap tugas T dan ukuran performa P, jika performanya pada T, sebagaimana diukur oleh P, meningkat dengan pengalaman E."*

### Terjemahkan ke Bahasa Manusia

Kamu mau membuat program yang bisa membedakan gambar kucing dan anjing (Tugas T). Kamu memberinya ribuan gambar kucing dan anjing yang sudah dilabeli (Pengalaman E). Program belajar dari gambar-gambar tersebut. Setelah belajar, akurasinya dalam membedakan kucing vs anjing (Performa P) meningkat.

Itu Machine Learning.

### Analogi Terbaik: Belajar Mengendarai Sepeda

Pikirkan bagaimana kamu belajar naik sepeda:

1. **Awalnya kamu tidak tahu caranya.** Kamu naik, jatuh, naik lagi, jatuh lagi.
2. **Kamu mencoba berbagai strategi.** Miring ke kiri, sepeda jatuh ke kiri. Tekan pedal lebih keras, sepeda jalan lebih cepat.
3. **Kamu belajar dari pengalaman.** Setiap kali kamu jatuh, otakmu merekam: *"Ah, kalau miring terlalu jauh ke kiri, aku jatuh."*
4. **Lama-kelamaan kamu bisa.** Tanpa sadar, otakmu sudah membuat "model" internal tentang cara menyeimbangkan sepeda.

Perhatikan: **Tidak ada yang mengajarmu rumus fisika keseimbangan.** Tidak ada yang bilang *"letakkan pusat gravitasmu di titik X"*. Kamu belajar dari **pengalaman** — dari mencoba, gagal, dan menyesuaikan.

Itulah yang dilakukan Machine Learning. Komputer tidak diprogram dengan aturan "kalau pixel di sini bernilai X, maka ini kucing". Komputer **belajar sendiri** dari contoh-contoh gambar yang kita berikan.

---

## 1.3 Pemrograman Tradisional vs Machine Learning

Ini adalah perbedaan paling fundamental yang harus kamu pahami:

### Pemrograman Tradisional

```
Input Data + Aturan (Kode Program) → Output
```

Kamu menulis aturan. Komputer mengikuti. Hasilnya deterministic — selalu sama untuk input yang sama.

**Contoh:** Program kalkulator. Kamu tulis `3 + 5 = 8`. Selamanya `3 + 5 = 8`. Aturannya jelas, pasti, tidak berubah.

### Machine Learning

```
Input Data + Output (Label) → Aturan (Model) → Prediksi
```

Kamu memberikan data dan jawaban. Komputer menemukan aturannya sendiri. Aturan ini (model) lalu bisa dipakai untuk memprediksi jawaban baru.

**Contoh:** Kamu beri 10,000 gambar kucing (labeled "kucing") dan 10,000 gambar anjing (labeled "anjing"). Komputer secara otomatis menemukan pola — bentuk telinga, tekstur bulu, bentuk mata — dan membuat model yang bisa mengklasifikasi gambar baru.

### Diagram Visual

```
 PEMROGRAMAN TRADISIONAL              MACHINE LEARNING
 ┌──────────────────┐               ┌──────────────────┐
 │  Input Data      │               │  Input Data      │
 │  +               │               │  +               │
 │  ATURAN (kode)   │               │  OUTPUT (jawaban)│
 │        ↓         │               │        ↓         │
 │  ┌──────────┐    │               │  ┌──────────┐    │
 │  │ Program  │    │               │  │ Algoritma│    │
 │  └──────────┘    │               │  │  ML      │    │
 │        ↓         │               │  └──────────┘    │
 │  Output          │               │        ↓         │
 │  (pasti & tetap) │               │  ATURAN (model)  │
 └──────────────────┘               │        ↓         │
                                    │  Prediksi baru   │
                                    └──────────────────┘
```

### Contoh Konkret: Prediksi Harga Rumah

**Cara Tradisional:**
```python
def harga_rumah(luas, lokasi, kamar):
    if lokasi == "Jakarta":
        return luas * 15000000 + kamar * 50000000
    elif lokasi == "Bandung":
        return luas * 8000000 + kamar * 30000000
    # ... 1000 baris if-else lainnya
```
Masalah: Rumus ini **kamu yang buat**. Kalau pasar berubah, kamu harus update manual. Kalau ada variabel baru (misal dekat MRT), kamu harus nulis rumus baru.

**Cara Machine Learning:**
```python
from sklearn.linear_model import LinearRegression

# Data: luas, lokasi(kode), kamar, jarak_MRT → harga
model = LinearRegression()
model.fit(X_training, y_harga)  # Komputer menemukan rumus sendiri!

# Prediksi rumah baru
harga = model.predict([[100, 1, 3, 500]])
```
Komputer menganalisis ribuan data penjualan rumah dan **menemukan polanya sendiri** — termasuk pola yang mungkin tidak kamu sadari (misalnya, rumah dekat MRT 30% lebih mahal).

---

## 1.4 Sejarah Singkat: Dari Mimpi Menjadi Kenyataan

Machine Learning bukan hal baru. Sejarahnya panjang dan penuh pasang-surut.

### Timeline

```
1943 ─── McCulloch & Pitts: Model neuron pertama (perceptron teori)
  │
1950 ─── Alan Turing: "Computing Machinery and Intelligence"
  │       (Paper: Bisakah mesin berpikir?)
  │
1957 ─── Frank Rosenblatt: Perceptron (neural network pertama di hardware)
  │
1969 ─── Minsky & Papert: "Perceptrons" (buku yang membuktikan
  │       perceptron TIDAK BISA menyelesaikan XOR → Zaman Gelap AI dimulai)
  │
1970s ─── Zaman Gelap AI (AI Winter): Dana dipotong, optimisme hilang
  │
1980s ─── Revival: Backpropagation ditemukan, neural network kembali
  │
1990s ─── SVM, Random Forest, Boosting muncul
  │       └── ML berpindah dari neural ke metode statistik
  │
2000s ─── Data mining boom, ML di industri mulai masuk
  │
2006 ─── Hinton: Deep Learning (pre-training deep nets)
  │
2009 ─── ImageNet dataset diluncurkan
  │
2012 ─── AlexNet: Deep Learning menghancurkan kompetisi ImageNet
  │       └── Error rate turun dari 26% → 15% (loncatan besar!)
  │
2014 ─── GAN (Generative Adversarial Network) ditemukan
  │
2016 ─── AlphaGo mengalahkan juara dunia Go (Lee Sedol)
  │
2017 ─── Transformer architecture ditemukan ("Attention Is All You Need")
  │
2020s ─── GPT, DALL-E, Stable Diffusion, dan era AI generatif
  │       └── ML ada di mana-mana: ponsel, rumah, mobil, rumah sakit
```

### Pelajaran dari Sejarah

1. **ML butuh data.** Sebelum era internet, tidak ada cukup data untuk melatih model besar.
2. **ML butuh komputasi.** GPU memungkinkan training neural network dalam hitungan jam, bukan tahun.
3. **ML butuh algoritma yang tepat.** Backpropagation, ReLU, Dropout — setiap terobosan membuat Deep Learning semakin praktis.
4. **Optimisme naik-turun.** AI sudah "mati" dua kali (AI Winter 1970s dan 1990s). Jangan terlalu euphoris, tapi juga jangan terlalu skeptis.

---

## 1.5 Komponen-Komponen Machine Learning

Mesin ML punya bagian-bagian yang saling bekerja sama. Seperti mobil punya mesin, bensin, roda, dan kemudi — masing-masing punya peran.

### 1. Data — Bahan Bakar

Tanpa data, ML hanyalah teori. Data adalah segalanya.

Bayangkan kamu mau mengajarkan anak kecil mengenali buah. Kamu tunjukin apel merah, apel hijau, jeruk, pisang. Makin banyak buah yang kamu tunjukin, makin baik anak itu mengenal buah.

Tapi kualitas juga penting. Kalau kamu hanya tunjukin apel merah, anak itu mungkin mengira semua buah merah adalah apel. Ini masalah — dan kita akan bahas ini mendalam di bab tentang **Overfitting**.

**Jenis data:**

| Tipe | Contoh | Bentuk |
|------|--------|--------|
| Numerik (kontinu) | Suhu: 27.5°C, Harga: 150000 | Angka desimal |
| Numerik (diskrit) | Jumlah kamar: 3, Umur: 25 | Angka bulat |
| Kategorikal | Warna: merah/hijau/biru | Kategori tetap |
| Ordinal | Ukuran: S/M/L/XL | Kategori berurutan |
| Biner | Ya/Tidak, 0/1 | Hanya 2 pilihan |
| Teks | Review produk, email | String karakter |
| Gambar | Foto 224x224 pixel | Array 3D (H×W×C) |
| Audio | Suara .wav | Array 1D (time series) |

### 2. Fitur — Lensa Yang Melihat Dunia

Fitur adalah cara kita **mewakili data**. Bayangkan kamu mau memprediksi apakah seseorang akan membeli produk. Apa informasi yang relevan?

- Umur? Mungkin.
- Pendapatan? Mungkin.
- Warna sepatu? Mungkin tidak.

Memilih fitur yang tepat disebut **Feature Engineering** — dan ini sering menjadi perbedaan antara model yang biasa dan model yang luar biasa.

```python
# Data mentah
data = {
    'nama': 'Budi',
    'umur': 28,
    'gaji': 8000000,
    'kota': 'Jakarta',
    'sepatu_warna': 'hitam'  # ← fitur tidak relevan!
}

# Fitur yang dipilih untuk ML
features = [28, 8000000, 1]  # [umur, gaji, kota_kode]
# Kita buang nama (unik, tidak membantu) dan warna sepatu (tidak relevan)
```

### 3. Model — Otak Yang Belajar

Model adalah "mesin" yang memproses input dan menghasilkan output. Model punya **parameter** — angka-angka yang disesuaikan selama proses belajar.

Pikirkan model seperti **rumus matematika dengan variabel**:

```
y = w1×x1 + w2×x2 + b
```

Awalnya, `w1`, `w2`, dan `b` diisi angka acak. Selama training, angka-angka ini disesuaikan sampai rumus menghasilkan prediksi yang akurat.

**Model = rumus + parameter yang sudah dioptimalkan.**

### 4. Loss Function — Hakim Yang Menilai

Loss function mengukur **seberapa salah** prediksi model. Semakin kecil loss, semakin baik model.

Analoginya seperti ujian:
- Jawaban benar → dapat 100 (loss = 0)
- Jawaban sedikit salah → dapat 80 (loss = 20)
- Jawaban totally wrong → dapat 0 (loss = 100)

Model berusaha **meminimalkan loss** — yaitu, berusaha menjawab sedikit mungkin salah.

```python
# Contoh: Mean Squared Error (untuk regression)
def mse(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

# y_true = [3.0, 5.0], y_pred = [2.8, 5.3]
# mse = ((3.0-2.8)² + (5.0-5.3)²) / 2
#     = (0.04 + 0.09) / 2
#     = 0.065
```

### 5. Optimizer — Mesin Penyesuai

Optimizer mengatur **bagaimana parameter diubah** berdasarkan loss. Optimizer melihat loss, menghitung arah yang membuat loss lebih kecil (gradien), lalu melangkah ke arah itu.

Analoginya: Kamu buta, berdiri di gunung, dan mau turun ke lembah. Kamu meraba-raba tanah di sekelilingmu, mencari arah yang menurun, lalu melangkah ke sana. Itulah **Gradient Descent** — optimizer paling dasar.

```
Posisi saat ini: ketinggian 1000m (loss tinggi)
Raba ke utara: 1010m (naik) ✗
Raba ke selatan: 990m (turun) ✓ → Langkah ke selatan!
Raba ke timur: 1005m (naik) ✗
Raba ke barat: 995m (turun) ✓ → Langkah ke barat juga!

Ulangi sampai mencapai lembah (loss minimum)
```

### 6. Training Loop — Siklus Belajar

Proses training adalah siklus berulang:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. FORWARD PASS: Model membuat prediksi            │
│        X → Model → y_pred                           │
│                                                     │
│  2. HITUNG LOSS: Seberapa salah prediksi?            │
│        Loss = loss_function(y_true, y_pred)         │
│                                                     │
│  3. BACKWARD PASS: Hitung gradient (arah perbaikan)  │
│        ∂Loss/∂W = seberapa sensitif loss terhadap W  │
│                                                     │
│  4. UPDATE: Sesuaikan parameter                      │
│        W = W - learning_rate × ∂Loss/∂W             │
│                                                     │
│  └──────────→ Ulangi dari 1 ──────────────────┘     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 1.6 Vocabulary: Istilah-Istilah yang Harus Kamu Kuasai

Bab-bab selanjutnya akan menggunakan istilah-istilah ini. Hafalkan pohonnya, bukan daunnya — pahami konsepnya, bukan sekadar definisinya.

### Dataset Terminology

| Istilah | Penjelasan | Analogi |
|---------|-----------|---------|
| **Dataset** | Kumpulan seluruh data | Seluruh buku di perpustakaan |
| **Training Set** | Data untuk melatih model | Buku yang kamu pelajari |
| **Validation Set** | Data untuk menyetel hyperparameter | Latihan ujian simulasi |
| **Test Set** | Data untuk menguji model final | Ujian sesungguhnya |
| **Feature (X)** | Input / variabel independen | Soal ujian |
| **Label (y)** | Output / variabel dependen | Kunci jawaban |
| **Sample** | Satu baris data (satu observasi) | Satu soal ujian |
| **Dimension** | Jumlah fitur | Jumlah soal dalam satu lembar |

### Model Terminology

| Istilah | Penjelasan | Analogi |
|---------|-----------|---------|
| **Parameter** | Angka yang dipelajari model (weight, bias) | Jawaban yang kamu temukan sendiri |
| **Hyperparameter** | Setting yang kamu tentukan sebelum training | Kebijakan belajar (jam berapa, berapa lama) |
| **Training** | Proses model belajar dari data | Belajar dari buku |
| **Inference** | Proses model membuat prediksi | Mengerjakan ujian |
| **Fit** | Melatih model pada data | "Fit" = menyesuaikan |
| **Predict** | Model menebak jawaban | Menjawab soal ujian |
| **Loss** | Ukuran kesalahan | Nilai salah di ujian |
| **Gradient** | Arah dan besarnya perubahan loss | Arah "turun gunung" |
| **Epoch** | Satu kali model melihat seluruh data | Satu kali membaca buku dari depan sampai belakang |
| **Batch** | Sebagian data yang diproses sekaligus | Satu bab buku |
| **Iteration** | Satu update parameter | Satu kali belajar dari satu bab |
| **Learning Rate** | Seberapa besar langkah update | Kecepatan berjalan turun gunung |

### Performance Terminology

| Istilah | Penjelasan | Analogi |
|---------|-----------|---------|
| **Accuracy** | Persentase prediksi benar | Nilai ujian |
| **Precision** | Dari yang diprediksi positif, berapa yang benar? | Dari jawaban yang kamu bilang benar, berapa yang memang benar? |
| **Recall** | Dari yang sebenarnya positif, berapa yang tertangkap? | Dari soal yang harusnya dijawab benar, berapa yang berhasil kamu jawab? |
| **F1-Score** | Rata-rata harmonik precision & recall | Keseimbangan antara teliti dan lengkap |
| **Overfitting** | Model "menghafal" data training tapi jelek di data baru | Menghafal jawaban ujian tahun lalu, tapi tidak bisa jawab soal baru |
| **Underfitting** | Model terlalu sederhana, jelek di semua data | Tidak belajar sama sekali |
| **Generalisasi** | Kemampuan model di data yang belum pernah dilihat | Bisa menjawab soal yang belum pernah kamu lihat |

---

## 1.7 Contoh End-to-End: ML dari Awal Sampai Akhir

Mari kita ikuti satu contoh lengkap — dari punya masalah sampai punya solusi. **Tidak ada yang bisa dilompati.**

### Masalah

Kamu bekerja di bank. Bank ingin memprediksi: apakah nasabah yang mengajukan kredit akan **lancar bayar** atau **gagal bayar**?

### Step 1: Kumpulkan Data

```python
import pandas as pd

data = pd.DataFrame({
    'umur':       [25, 45, 32, 28, 55, 35, 42, 23, 50, 30],
    'gaji_juta':  [5,  15, 8,  4,  20, 10, 12, 3,  18, 7],
    'hutang_juta':[2,  3,  5,  4,  2,  1,  6,  2,  1,  8],
    'status':     ['Lancar', 'Lancar', 'Lancar', 'Gagal', 'Lancar',
                   'Lancar', 'Gagal',  'Gagal',  'Lancar', 'Gagal']
})

print(data)
```

Output:
```
   umur  gaji_juta  hutang_juta  status
0    25          5           2  Lancar
1    45         15           3  Lancar
2    32          8           5  Lancar
3    28          4           4  Gagal
4    55         20           2  Lancar
5    35         10           1  Lancar
6    42         12           6  Gagal
7    23          3           2  Gagal
8    50         18           1  Lancar
9    30          7           8  Gagal
```

### Step 2: Pahami Data

Pola yang terlihat:
- Gaji tinggi + hutang rendah → cenderung Lancar
- Gaji rendah + hutang tinggi → cenderung Gagal
- Tapi ada pengecualian (lihat baris #7: gaji rendah hutang rendah tapi gagal)

ML tidak butuh kita menulis aturan ini. ML akan **menemukan pola ini sendiri**.

### Step 3: Pisahkan Features dan Labels

```python
X = data[['umur', 'gaji_juta', 'hutang_juto']].values  # Features (input)
y = data['status'].values  # Labels (output yang ingin diprediksi)
```

### Step 4: Split Data — Training dan Testing

Kenapa harus di-split? Karena kita harus tahu apakah model benar-benar **belajar** atau hanya **menghafal**.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 70% untuk training, 30% untuk testing

print(f"Training: {len(X_train)} data")
print(f"Testing:  {len(X_test)} data")
```

Analoginya: Kalau kamu mau tahu apakah siswa benar-benar paham matematika, kamu tidak memberi soal ujian yang sama dengan soal latihan. Kamu kasih **soal baru** yang mirip tapi berbeda.

### Step 5: Pilih Algoritma

Untuk masalah ini (klasifikasi biner), banyak pilihan:

| Algoritma | Kapan Cocok? |
|-----------|-------------|
| K-NN | Data sedikit, sederhana |
| Decision Tree | Butuh interpretasi |
| Naive Bayes | Data teks, cepat |
| Neural Network | Data banyak, pola kompleks |
| Logistic Regression | Baseline, linear |

Kita pilih **Decision Tree** kali ini karena hasilnya mudah dipahami.

### Step 6: Training

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
```

Di balik `fit()`, komputer sedang:
1. Mencoba semua kemungkinan split (gaji ≤ 5? hutang ≤ 3? umur ≤ 30?)
2. Menghitung impurity (seberapa "campur" data di setiap cabang)
3. Memilih split yang membuat data paling "murni"
4. Mengulang secara rekursif untuk setiap cabang

### Step 7: Evaluasi

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
```

### Step 8: Prediksi Data Baru

```python
nasabah_baru = [[27, 6, 3]]  # Umur 27, gaji 6jt, hutang 3jt
hasil = model.predict(nasabah_baru)
print(f"Prediksi: {hasil[0]}")  # Lancar atau Gagal?
```

### Step 9: Deploy dan Monitor

Model yang sudah dilatih bisa:
- Disimpan ke file (`joblib.dump`)
- Di-deploy sebagai API (Flask, FastAPI)
- Di-monitor untuk deteksi drift (kalau pola data berubah seiring waktu)

---

## 1.8 Mind Map: Peta Besar Machine Learning

```
                    MACHINE LEARNING
                         │
          ┌──────────────┼───────────────┐
          │              │               │
    Supervised      Unsupervised    Reinforcement
    Learning        Learning        Learning
          │              │               │
    ┌─────┼─────┐   ┌───┼───┐      ┌────┼────┐
    │     │     │   │   │   │      │        │
  Class. Reg. Time │   │   │      Game   Robotics
   │     │   Series Clust Dim     Playing
   │     │    │   er   Reduc    │
   │     │  Forecast  │    │   Autonomous
   │     │    │     K-M  PCA    Vehicles
   │     │  ARIMA  DBSCAN
   │     │
K-NN  Linear  SVM
DT    Logistic RF
NB    Poly     XGBoost
NN    Ridge    LightGBM
```

---

## 1.9 Mitos-Mitos tentang Machine Learning

Sebelum lanjut, mari hancurkan beberapa mitos:

### Mitos 1: "ML itu AI yang bisa berpikir sendiri"

**Kenyataan:** ML tidak "berpikir". ML adalah optimisasi matematika — mencari parameter yang meminimalkan loss. Tidak ada kesadaran, tidak ada pemahaman, tidak ada niat.

ML itu seperti kalkulator yang sangat canggih: masukkan data, keluar prediksi. "Kecerdasan" nya adalah pola dalam data, bukan pemikiran.

### Mitos 2: "ML selalu lebih baik dari pemrograman tradisional"

**Kenyataan:** Kalau aturannya jelas dan tetap, pemrograman tradisional tetap yang terbaik. ML unggul ketika:
- Aturannya terlalu banyak untuk ditulis manual
- Aturannya berubah seiring waktu
- Aturannya tidak diketahui (kita bahkan tidak tahu polanya)

Untuk menghitung pajak? Pakai aturan (pemrograman tradisional). Untuk mengenali wajah? Pakai ML.

### Mitos 3: "ML butuh data jutaan baris"

**Kenyataan:** Tergantung masalahnya. Untuk masalah sederhana (klasifikasi biner, sedikit fitur), ratusan data bisa cukup. Untuk Deep Learning (pengenalan gambar), ya, butuh ribuan sampai jutaan.

Aturan praktis: jumlah data training ≥ 10× jumlah parameter model.

### Mitos 4: "ML yang kompleks selalu lebih baik"

**Kenyataan:** Mulailah dari model sederhana (Linear Regression, Decision Tree). Kalau performa belum cukup, naikkan kompleksitas secara bertahap. Model sederhana yang benar > model kompleks yang tidak dipahami.

Ini terkait langsung dengan **Overfitting** — topik yang akan kita bahas mendalam di Bab 4.

### Mitos 5: "ML bisa memprediksi masa depan dengan sempurna"

**Kenyataan:** ML memprediksi berdasarkan **pola di data masa lalu**. Kalau masa depan fundamentally berbeda dari masa lalu (black swan event), ML akan salah. Pandemi COVID-19 adalah contoh sempurna — model yang dilatih di data pra-pandemi tidak bisa memprediksi perubahan drastis.

---

## 1.10 Prasyarat: Apa yang Kamu Butuhkan Sebelum Lanjut?

### Matematika

Kamu **tidak perlu** jadi ahli matematika untuk menggunakan ML. Tapi kamu perlu nyaman dengan:

| Topik | Level yang Dibutuhkan | Kenapa? |
|-------|----------------------|---------|
| Aljabar linear | Dasar | Operasi matriks di mana-mana |
| Kalkulus | Konsep turunan | Gradient descent butuh turunan |
| Probabilitas | Menengah | Banyak algoritma ML berbasis probabilitas |
| Statistik | Dasar | Mean, variance, distribusi, korelasi |

### Programming

Python adalah bahasa de facto untuk ML:

```python
# Yang harus kamu kuasai:
# 1. Variabel, tipe data, list, dict
# 2. Loop dan kondisi
# 3. Fungsi dan class
# 4. NumPy dasar (array, operasi vektor)
# 5. Pandas dasar (DataFrame, read CSV)
```

### Library ML yang Akan Digunakan

| Library | Fungsi | Besar? |
|---------|--------|--------|
| NumPy | Operasi numerik | Ya, fondasi |
| Pandas | Manipulasi data | Ya, must-have |
| Scikit-learn | Algoritma ML klasik | Ya, utama |
| Matplotlib | Visualisasi | Ya, must-have |
| TensorFlow/PyTorch | Deep Learning | Ya, untuk bab NN |

---

## 1.11 Ringkasan Bab 1

1. **Pemrograman tradisional** = kamu tulis aturan, komputer ikuti. **ML** = kamu kasih data + jawaban, komputer temukan aturannya.
2. ML belajar dari **pengalaman** (data), bukan dari instruksi eksplisit.
3. Komponen ML: **Data → Fitur → Model → Loss → Optimizer → Training Loop → Evaluasi**
4. ML cocok untuk masalah di mana aturan terlalu banyak, terlalu kompleks, atau tidak diketahui.
5. ML bukan sihir — ini optimisasi matematika yang mencari pola dalam data.
6. Mulailah dari yang sederhana, naikkan kompleksitas hanya jika perlu.

---

**Selanjutnya → Bab 2: Jenis-Jenis Machine Learning**

Di bab berikutnya, kita akan masuk lebih dalam ke tiga paradigma besar ML: Supervised Learning, Unsupervised Learning, dan Reinforcement Learning. Kita akan memahami kapan memakai yang mana, apa perbedaannya, dan apa contoh nyatanya.