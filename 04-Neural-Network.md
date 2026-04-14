# Neural Network (Jaringan Saraf Tiruan) - Penjelasan Lengkap

## Daftar Isi
1. [Konsep Dasar](#konsep-dasar)
2. [Arsitektur dan Komponen](#arsitektur-dan-komponen)
3. [Cara Kerja Detail (Step-by-Step)](#cara-kerja-detail)
4. [Kode Python dari Nol (Tanpa Library)](#kode-python-dari-nol)
5. [Contoh Manual: 4 Data Point (XOR)](#contoh-manual-4-data)
6. [Contoh Manual: 6 Data Point (Regressi)](#contoh-manual-6-data)
7. [Contoh Skala Besar dengan PyTorch/TensorFlow](#contoh-skala-besar)
8. [Alur Kerja Program Lengkap](#alur-kerja-program)
9. [Semua Operasi Matematika yang Dilakukan](#operasi-matematika)
10. [Backpropagation Detail](#backpropagation-detail)
11. [Activation Functions](#activation-functions)
12. [Optimizers](#optimizers)
13. [Pro dan Kontra](#pro-dan-kontra)

---

## Konsep Dasar

Neural Network (NN) terinspirasi dari cara kerja **otak manusia**. Otak terdiri dari miliaran **neuron** yang saling terhubung, menerima sinyal, memprosesnya, dan mengirim sinyal keluar.

Analogi sederhana:
> Bayangkan sebuah pabrik dengan banyak pekerja. Setiap pekerja menerima bahan baku dari pekerja sebelumnya, memprosesnya, lalu mengirim hasilnya ke pekerja berikutnya. Di akhir, produk jadi diperiksa — jika salah, pesan kesalahan dikirim mundur agar setiap pekerja menyesuaikan cara kerjanya.

### Perceptron (Neuron Tunggal)

```
Input:  x1 ──→ w1 ─┐
        x2 ──→ w2 ─┤
        x3 ──→ w3 ─┤──→ Σ(wx+b) ──→ f(·) ──→ Output
                       │
        bias ──→ b ───┘
```

**Operasi dalam 1 neuron:**
```
z = w1×x1 + w2×x2 + w3×x3 + b    (weighted sum)
a = f(z)                           (activation function)
```

### Dari Perceptron ke Network:

```
Layer Input (3 neurons)     Hidden Layer 1 (4 neurons)    Hidden Layer 2 (4 neurons)    Output (1 neuron)
  x1 ──────────────────────→ h1 ────────────────────────→ h5 ──────────────────────→ y
  x2 ──────────────────────→ h2 ────────────────────────→ h6 ──────────────────────→ y
  x3 ──────────────────────→ h3 ────────────────────────→ h7
                             h4 ────────────────────────→ h8
```

Setiap koneksi punya **weight** (bobot), setiap neuron punya **bias**.

---

## Arsitektur dan Komponen

| Komponen | Penjelasan | Contoh |
|----------|-----------|--------|
| **Input Layer** | Menerima data mentah, tidak memproses | Pixel gambar, fitur data |
| **Hidden Layer** | Memproses data, belajar representasi | Feature extraction |
| **Output Layer** | Menghasilkan prediksi akhir | Probabilitas kelas |
| **Weight (w)** | Kekuatan koneksi antar neuron | w = 0.5 |
| **Bias (b)** | Threshold aktifasi per neuron | b = -0.3 |
| **Activation (f)** | Fungsi nonlinear yang menentukan output | ReLU, Sigmoid, Softmax |

### Tipe Neural Network:

| Tipe | Arsitektur | Penggunaan |
|------|-----------|------------|
| **Feedforward (FNN)** | Input → Hidden → Output | Tabular data, klasifikasi sederhana |
| **Convolutional (CNN)** | Conv → Pool → FC | Gambar, video |
| **Recurrent (RNN)** | Recurrent connections | Teks, time series |
| **LSTM/GRU** | Gated RNN | Long sequences |
| **Transformer** | Self-attention | NLP modern (GPT, BERT) |

Fokus dokumen ini: **Feedforward Neural Network (FNN)**

---

## Cara Kerja Detail

### Training Loop (Satu Epoch):

```
1. FORWARD PASS:
   Untuk setiap data (x, y_true):
     a. Input: z[0] = x
     b. Untuk setiap layer l:
        z[l] = W[l] × a[l-1] + b[l]     (linear transformation)
        a[l] = f(z[l])                    (activation)
     c. Output: y_pred = a[L]
     d. Hitung loss: L = loss_function(y_true, y_pred)

2. BACKWARD PASS (BACKPROPAGATION):
   a. Hitung gradient loss terhadap output: dL/dy_pred
   b. Untuk setiap layer (dari output ke input):
      dL/dW[l] = dL/da[l] × da[l]/dz[l] × dz[l]/dW[l]
      dL/db[l] = dL/da[l] × da[l]/dz[l] × dz[l]/db[l]
   c. Simpan semua gradient

3. UPDATE WEIGHTS:
   W[l] = W[l] - learning_rate × dL/dW[l]
   b[l] = b[l] - learning_rate × dL/db[l]

4. Ulangi dari langkah 1 untuk epoch berikutnya
```

---

## Kode Python dari Nol

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, activations=None):
        """
        layer_sizes: list of ints, e.g. [2, 4, 4, 1]
                     (input, hidden1, hidden2, output)
        activations: list of strings, e.g. ['relu', 'relu', 'sigmoid']
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes) - 1

        if activations is None:
            self.activations = ['relu'] * (self.n_layers - 1) + ['sigmoid']
        else:
            self.activations = activations

        # Xavier/He Initialization
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            if self.activations[i] == 'relu':
                std = np.sqrt(2.0 / fan_in)  # He initialization
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier

            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        # For tracking
        self.loss_history = []

    def _activation(self, z, func):
        if func == 'relu':
            return np.maximum(0, z)
        elif func == 'sigmoid':
            z_clipped = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z_clipped))
        elif func == 'tanh':
            return np.tanh(z)
        elif func == 'softmax':
            z_shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif func == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {func}")

    def _activation_derivative(self, z, a, func):
        if func == 'relu':
            return (z > 0).astype(float)
        elif func == 'sigmoid':
            return a * (1 - a)
        elif func == 'tanh':
            return 1 - a ** 2
        elif func == 'linear':
            return np.ones_like(z)
        else:
            raise ValueError(f"Derivative not implemented for: {func}")

    def forward(self, X):
        """Forward pass: X → output"""
        self.z_list = []
        self.a_list = [X]

        a = X
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.z_list.append(z)
            a = self._activation(z, self.activations[i])
            self.a_list.append(a)

        return a

    def _compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss"""
        m = y_true.shape[0]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        if self.activations[-1] == 'sigmoid':
            loss = -(1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.activations[-1] == 'softmax':
            loss = -(1 / m) * np.sum(y_true * np.log(y_pred + eps))
        else:
            loss = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

        return loss

    def backward(self, y_true):
        """Backpropagation: compute gradients"""
        m = y_true.shape[0]
        self.dW = []
        self.db = []

        # Output layer gradient
        if self.activations[-1] == 'sigmoid':
            da = -(y_true / self.a_list[-1] - (1 - y_true) / (1 - self.a_list[-1]))
            dz = da * self._activation_derivative(self.z_list[-1], self.a_list[-1], 'sigmoid')
        elif self.activations[-1] == 'softmax':
            dz = self.a_list[-1] - y_true
        else:
            dz = (self.a_list[-1] - y_true) * self._activation_derivative(
                self.z_list[-1], self.a_list[-1], self.activations[-1])

        dW_l = (1 / m) * (self.a_list[-2].T @ dz)
        db_l = (1 / m) * np.sum(dz, axis=0, keepdims=True)
        self.dW.insert(0, dW_l)
        self.db.insert(0, db_l)

        # Hidden layers gradient (backpropagate)
        da_prev = dz @ self.weights[-1].T

        for i in range(self.n_layers - 2, -1, -1):
            dz = da_prev * self._activation_derivative(self.z_list[i], self.a_list[i + 1],
                                                         self.activations[i])
            dW_l = (1 / m) * (self.a_list[i].T @ dz)
            db_l = (1 / m) * np.sum(dz, axis=0, keepdims=True)
            self.dW.insert(0, dW_l)
            self.db.insert(0, db_l)

            if i > 0:
                da_prev = dz @ self.weights[i].T

    def update_weights(self):
        """Gradient descent update"""
        for i in range(self.n_layers):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.db[i]

    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """Train the network"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Forward
                y_pred = self.forward(X_batch)
                loss = self._compute_loss(y_pred, y_batch)
                epoch_loss += loss

                # Backward
                self.backward(y_batch)

                # Update
                self.update_weights()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    def predict(self, X):
        """Predict class labels"""
        X = np.array(X, dtype=float)
        probs = self.forward(X)

        if self.activations[-1] == 'sigmoid':
            return (probs >= 0.5).astype(int).flatten()
        elif self.activations[-1] == 'softmax':
            return np.argmax(probs, axis=1)
        else:
            return probs.flatten()

    def predict_proba(self, X):
        """Predict probabilities"""
        X = np.array(X, dtype=float)
        return self.forward(X)

    def score(self, X, y):
        """Accuracy score"""
        predictions = self.predict(X)
        y = np.array(y).flatten()
        return np.mean(predictions == y)
```

---

## Contoh Manual: 4 Data Point

### Dataset: XOR Problem (Masalah klasik yang tidak bisa diselesaikan oleh single perceptron)

| # | x1 | x2 | y  |
|---|----|----|----|
| 1 | 0  | 0  | 0  |
| 2 | 0  | 1  | 1  |
| 3 | 1  | 0  | 1  |
| 4 | 1  | 1  | 0  |

**Kenapa XOR penting?** Single perceptron TIDAK BISA menyelesaikan XOR — ini yang membuktikan perlunya hidden layer!

### Arsitektur: 2-3-1

```
Input (2) → Hidden (3) → Output (1)
Activation: ReLU → Sigmoid
```

### Forward Pass (Satu Data: x1=0, x2=1, y=1)

**Inisialisasi (contoh setelah beberapa iterasi training):**

```
W[0] = [[ 0.5, -0.3,  0.8],   # x1 → h1, h2, h3
         [-0.4,  0.7, -0.2]]   # x2 → h1, h2, h3

b[0] = [[ 0.1,  0.0, -0.1]]

W[1] = [[ 0.6],   # h1 → output
         [-0.5],   # h2 → output
         [ 0.4]]   # h3 → output

b[1] = [[-0.2]]
```

**Step 1: Input Layer → Hidden Layer**
```
z[0] = X @ W[0] + b[0]
     = [0, 1] @ [[ 0.5, -0.3,  0.8],
                  [-0.4,  0.7, -0.2]] + [0.1, 0.0, -0.1]

     = [0×0.5+1×(-0.4), 0×(-0.3)+1×0.7, 0×0.8+1×(-0.2)] + [0.1, 0.0, -0.1]
     = [-0.4, 0.7, -0.2] + [0.1, 0.0, -0.1]
     = [-0.3, 0.7, -0.3]

a[0] = ReLU(z[0])
     = [max(0,-0.3), max(0,0.7), max(0,-0.3)]
     = [0, 0.7, 0]
```

**Step 2: Hidden Layer → Output Layer**
```
z[1] = a[0] @ W[1] + b[1]
     = [0, 0.7, 0] @ [[ 0.6],
                        [-0.5],
                        [ 0.4]] + [-0.2]

     = [0×0.6 + 0.7×(-0.5) + 0×0.4] + [-0.2]
     = [-0.35] + [-0.2]
     = [-0.55]

a[1] = Sigmoid(z[1])
     = 1 / (1 + e^(-(-0.55)))
     = 1 / (1 + e^(0.55))
     = 1 / (1 + 1.733)
     = 1 / 2.733
     = 0.366
```

**Step 3: Hitung Loss**
```
y_true = 1, y_pred = 0.366
Loss = -(1 × log(0.366) + 0 × log(1-0.366))
     = -(log(0.366))
     = -(-1.004)
     = 1.004
```

### Backward Pass (Backpropagation)

**Step 4: Output Layer Gradient**
```
da = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
   = -(1/0.366 - 0/(1-0.366))
   = -2.732

dz_output = da × sigmoid_derivative
          = da × y_pred × (1 - y_pred)
          = -2.732 × 0.366 × 0.634
          = -0.634

dW[1] = a[0].T @ dz_output / m
      = [0, 0.7, 0].T @ [-0.634] / 1
      = [[0×(-0.634)], [0.7×(-0.634)], [0×(-0.634)]]
      = [[0], [-0.444], [0]]

db[1] = dz_output / m = [-0.634]
```

**Step 5: Hidden Layer Gradient**
```
da_hidden = dz_output @ W[1].T
          = [-0.634] @ [[0.6, -0.5, 0.4]]
          = [-0.634×0.6, -0.634×(-0.5), -0.634×0.4]
          = [-0.380, 0.317, -0.254]

dz_hidden = da_hidden × ReLU_derivative(z[0])
          = [-0.380, 0.317, -0.254] × [0>0?, 0.7>0?, 0>0?]
          = [-0.380×0, 0.317×1, -0.254×0]
          = [0, 0.317, 0]

dW[0] = X.T @ dz_hidden / m
      = [0, 1].T @ [0, 0.317, 0] / 1
      (Perhitungan matriks lengkap)
      = [[0, 0, 0],
         [0, 0.317, 0]]

db[0] = [0, 0.317, 0]
```

**Step 6: Update Weights** (learning_rate = 0.1)
```
W[1] = W[1] - 0.1 × dW[1]
     = [[0.6], [-0.5], [0.4]] - 0.1 × [[0], [-0.444], [0]]
     = [[0.6], [-0.5+0.044], [0.4]]
     = [[0.6], [-0.456], [0.4]]

b[1] = b[1] - 0.1 × db[1]
     = [-0.2] - 0.1 × [-0.634]
     = [-0.2 + 0.063]
     = [-0.137]

W[0] = W[0] - 0.1 × dW[0]
     = [[0.5, -0.3, 0.8], [-0.4, 0.7, -0.2]] - 0.1 × [[0, 0, 0], [0, 0.317, 0]]
     = [[0.5, -0.3, 0.8], [-0.4, 0.668, -0.2]]

b[0] = b[0] - 0.1 × db[0]
     = [0.1, 0.0, -0.1] - 0.1 × [0, 0.317, 0]
     = [0.1, -0.032, -0.1]
```

### Kode:

```python
nn = NeuralNetwork([2, 3, 1], learning_rate=0.1, activations=['relu', 'sigmoid'])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

nn.fit(X, y, epochs=5000, batch_size=4, verbose=True)

print("\n=== Prediksi XOR ===")
for xi, yi in zip(X, y):
    pred = nn.predict([xi])[0]
    prob = nn.predict_proba([xi])[0][0]
    print(f"  Input: {xi} → Prediksi: {pred} (prob: {prob:.4f}) → Sebenarnya: {yi}")
```

---

## Contoh Manual: 6 Data Point

### Dataset: Prediksi Gaji Berdasarkan Pengalaman (Regression)

| # | Pengalaman (thn) | Level Edukasi | Gaji (jt) |
|---|------------------|---------------|-----------|
| 1 | 1                | 1             | 4.0       |
| 2 | 2                | 1             | 5.5       |
| 3 | 3                | 2             | 7.0       |
| 4 | 5                | 2             | 9.5       |
| 5 | 8                | 3             | 14.0      |
| 6 | 10               | 3             | 18.0      |

### Arsitektur: 2-4-1 (Regression)

```
Input (2) → Hidden (4, ReLU) → Output (1, Linear)
Loss: Mean Squared Error
```

### Forward Pass Detail (x=[3, 2], y_true=7.0)

```
z[0] = [3, 2] @ W[0] + b[0]   (2×4 weight matrix)
a[0] = ReLU(z[0])
z[1] = a[0] @ W[1] + b[1]     (4×1 weight matrix)
a[1] = z[1]                    (linear activation)
loss = MSE(a[1], 7.0)
```

### Kode:

```python
nn_reg = NeuralNetwork(
    layer_sizes=[2, 4, 1],
    learning_rate=0.01,
    activations=['relu', 'linear']
)

X_reg = np.array([[1, 1], [2, 1], [3, 2], [5, 2], [8, 3], [10, 3]], dtype=float)
y_reg = np.array([[4.0], [5.5], [7.0], [9.5], [14.0], [18.0]], dtype=float)

# Normalize inputs
X_mean, X_std = X_reg.mean(axis=0), X_reg.std(axis=0)
X_norm = (X_reg - X_mean) / (X_std + 1e-8)

nn_reg.fit(X_norm, y_reg, epochs=3000, batch_size=6)

print("\n=== Prediksi Gaji ===")
for xi, yi in zip(X_norm, y_reg):
    pred = nn_reg.predict_proba([xi])[0][0]
    print(f"  Prediksi: {pred:.1f} jt | Sebenarnya: {yi[0]:.1f} jt")
```

---

## Contoh Skala Besar

```python
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import time

# ============================================
# CONTOH 1: Klasifikasi Besar (50,000 data, 4 kelas)
# ============================================
print("=" * 60)
print("CONTOH 1: Klasifikasi - 50,000 data")
print("=" * 60)

X, y = make_classification(
    n_samples=50000, n_features=20, n_informative=12,
    n_classes=4, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# KRITICAL: Neural Networks require feature scaling!
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0001,            # L2 regularization
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)

start = time.time()
mlp.fit(X_train_s, y_train)
train_time = time.time() - start
print(f"\nWaktu training: {train_time:.2f} detik")

start = time.time()
y_pred = mlp.predict(X_test_s)
pred_time = time.time() - start
print(f"Waktu prediksi ({len(X_test_s)} data): {pred_time:.4f} detik")

print(f"Train accuracy: {mlp.score(X_train_s, y_train):.4f}")
print(f"Test accuracy:  {mlp.score(X_test_s, y_test):.4f}")
print(f"Iterations: {mlp.n_iter_}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ============================================
# CONTOH 2: Digits Dataset (Image Classification)
# ============================================
print("\n" + "=" * 60)
print("CONTOH 2: Handwritten Digits (8x8 images)")
print("=" * 60)

digits = load_digits()
X_d = digits.data / 16.0  # Normalize pixel values to [0, 1]
y_d = digits.target

Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(X_d, y_d, test_size=0.3)

mlp_digits = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
mlp_digits.fit(Xd_tr, yd_tr)

print(f"Test accuracy: {mlp_digits.score(Xd_te, yd_te):.4f}")

# Visualisasi weight matrix
print(f"Weight shapes:")
for i, coef in enumerate(mlp_digits.coefs_):
    print(f"  Layer {i} → {i+1}: {coef.shape}")

# ============================================
# CONTOH 3: Regression
# ============================================
print("\n" + "=" * 60)
print("CONTOH 3: Regression - Prediksi Nilai Kontinu")
print("=" * 60)

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X_h, y_h = housing.data, housing.target
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(X_h, y_h, test_size=0.2)

scaler_h = StandardScaler()
Xh_tr_s = scaler_h.fit_transform(Xh_tr)
Xh_te_s = scaler_h.transform(Xh_te)

mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)
mlp_reg.fit(Xh_tr_s, yh_tr)

from sklearn.metrics import mean_squared_error, r2_score
yh_pred = mlp_reg.predict(Xh_te_s)
print(f"R² Score: {r2_score(yh_te, yh_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(yh_te, yh_pred)):.4f}")

# ============================================
# EKSPERIMEN: Pengaruh Arsitektur
# ============================================
print("\n" + "=" * 60)
print("EKSPERIMEN: Pengaruh Arsitektur Hidden Layer")
print("=" * 60)

X_exp, y_exp = make_classification(n_samples=5000, n_features=10,
                                     n_classes=3, random_state=42)
Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(X_exp, y_exp, test_size=0.3)
sc = StandardScaler()
Xe_tr_s = sc.fit_transform(Xe_tr)
Xe_te_s = sc.transform(Xe_te)

architectures = [
    (10,),              # 1 hidden layer, 10 neurons
    (50,),              # 1 hidden layer, 50 neurons
    (100,),             # 1 hidden layer, 100 neurons
    (50, 25),           # 2 hidden layers
    (100, 50, 25),      # 3 hidden layers
    (200, 100, 50, 25), # 4 hidden layers
]

for arch in architectures:
    mlp_exp = MLPClassifier(hidden_layer_sizes=arch, max_iter=300, random_state=42)
    mlp_exp.fit(Xe_tr_s, ye_tr)
    train_acc = mlp_exp.score(Xe_tr_s, ye_tr)
    test_acc = mlp_exp.score(Xe_te_s, ye_te)
    n_params = sum(a * b for a, b in zip((10,) + arch, arch + (3,)))
    print(f"  {str(arch):22s} | Train={train_acc:.4f} | Test={test_acc:.4f} | Params≈{n_params}")

# ============================================
# EKSPERIMEN: Pengaruh Learning Rate
# ============================================
print("\n=== Pengaruh Learning Rate ===")
for lr in [0.0001, 0.001, 0.01, 0.1]:
    mlp_lr = MLPClassifier(hidden_layer_sizes=(50, 25), learning_rate_init=lr,
                            max_iter=300, random_state=42)
    mlp_lr.fit(Xe_tr_s, ye_tr)
    print(f"  LR={lr:8s} | Train={mlp_lr.score(Xe_tr_s, ye_tr):.4f} | Test={mlp_lr.score(Xe_te_s, ye_te):.4f}")

# ============================================
# EKSPERIMEN: Pengaruh Activation Function
# ============================================
print("\n=== Pengaruh Activation Function ===")
for act in ['relu', 'tanh', 'logistic']:
    mlp_act = MLPClassifier(hidden_layer_sizes=(50, 25), activation=act,
                             max_iter=300, random_state=42)
    mlp_act.fit(Xe_tr_s, ye_tr)
    print(f"  Activation={act:10s} | Train={mlp_act.score(Xe_tr_s, ye_tr):.4f} | Test={mlp_act.score(Xe_te_s, ye_te):.4f}")
```

---

## Alur Kerja Program

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      ALUR KERJA NEURAL NETWORK                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. INISIALISASI                                                            │
│     ├── Definisikan arsitektur: [input_size, hidden_sizes..., output_size]   │
│     ├── Inisialisasi weight (Xavier/He) untuk menghindari vanishing gradient│
│     ├── Inisialisasi bias = 0                                               │
│     ├── Set hyperparameters: learning_rate, batch_size, epochs               │
│     └── Set activation functions untuk setiap layer                           │
│                                                                              │
│  2. TRAINING LOOP (untuk setiap epoch):                                     │
│     │                                                                        │
│     ├── SHUFFLE data training                                                │
│     │                                                                        │
│     ├── For each MINI-BATCH:                                                │
│     │   │                                                                    │
│     │   ├── FORWARD PASS:                                                   │
│     │   │   │                                                                │
│     │   │   ├── For layer 0:                                                │
│     │   │   │   z[0] = X_batch @ W[0] + b[0]                               │
│     │   │   │   a[0] = activation[0](z[0])                                  │
│     │   │   │                                                                │
│     │   │   ├── For layer 1:                                                │
│     │   │   │   z[1] = a[0] @ W[1] + b[1]                                  │
│     │   │   │   a[1] = activation[1](z[1])                                  │
│     │   │   │                                                                │
│     │   │   ├── ...                                                          │
│     │   │   │                                                                │
│     │   │   └── For layer L (output):                                       │
│     │   │       z[L] = a[L-1] @ W[L] + b[L]                                │
│     │   │       a[L] = activation[L](z[L])   ← y_pred                      │
│     │   │                                                                    │
│     │   ├── COMPUTE LOSS:                                                   │
│     │   │   Classification: Cross-Entropy                                   │
│     │   │   Regression: MSE                                                  │
│     │   │                                                                    │
│     │   ├── BACKWARD PASS (Backpropagation):                                 │
│     │   │   │                                                                │
│     │   │   ├── Output layer gradient:                                       │
│     │   │   │   dz[L] = a[L] - y_batch  (sigmoid/softmax)                  │
│     │   │   │   dW[L] = a[L-1].T @ dz[L] / batch_size                      │
│     │   │   │   db[L] = sum(dz[L]) / batch_size                             │
│     │   │   │                                                                │
│     │   │   ├── Hidden layer gradients (dari L-1 ke 0):                     │
│     │   │   │   da = dz[l+1] @ W[l+1].T                                     │
│     │   │   │   dz[l] = da * activation_derivative(z[l], a[l])              │
│     │   │   │   dW[l] = a[l-1].T @ dz[l] / batch_size                       │
│     │   │   │   db[l] = sum(dz[l]) / batch_size                              │
│     │   │   │                                                                │
│     │   │   └─ Chain rule: error disebarkan mundur                           │
│     │   │                                                                    │
│     │   └── UPDATE WEIGHTS (Gradient Descent):                               │
│     │       W[l] = W[l] - learning_rate * dW[l]                              │
│     │       b[l] = b[l] - learning_rate * db[l]                              │
│     │                                                                        │
│     └── Log loss untuk monitoring                                            │
│                                                                              │
│  3. PREDICTION:                                                              │
│     └── Forward pass saja (tanpa backward)                                   │
│         x_new → z[0] → a[0] → z[1] → a[1] → ... → a[L] = y_pred            │
│                                                                              │
│  4. EVALUASI                                                                 │
│     ├── Classification: Accuracy, Precision, Recall, F1                      │
│     ├── Regression: MSE, R², MAE                                            │
│     └── Monitoring: Loss curve, overfitting detection                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Operasi Matematika

### Semua Operasi dalam Neural Network:

| Fase | Operasi | Jumlah | Detail |
|------|---------|--------|--------|
| **Forward** | | | |
| | Perkalian matriks | L kali | a[l-1] @ W[l] |
| | Penjumlahan vektor | L kali | + b[l] |
| | Activation | L kali | ReLU/Sigmoid/etc |
| | **Total per sample** | O(Σ ni × ni+1) | ni = neuron di layer i |
| **Backward** | | | |
| | Perkalian matriks (transpose) | L kali | dz @ W.T |
| | Perkalian element-wise | L kali | da * f'(z) |
| | Outer product | L kali | a.T @ dz |
| | **Total per sample** | O(Σ ni × ni+1) | Sama seperti forward |
| **Update** | | | |
| | Pengurangan | 2L kali | W -= lr*dW, b -= lr*db |
| | **Total per epoch** | O(E × N × Σ ni × ni+1) | E=jumlah parameter |

### Contoh Perhitungan Parameter:

```
Arsitektur: [2, 3, 1]

Layer 0→1: W[0] = 2×3 = 6 weights  + 3 biases   = 9 parameters
Layer 1→2: W[1] = 3×1 = 3 weights  + 1 bias      = 4 parameters
Total: 9 + 4 = 13 parameters

Arsitektur: [784, 128, 64, 10]  (contoh MNIST)
Layer 0→1: 784×128 + 128 = 100,480
Layer 1→2: 128×64 + 64    = 8,256
Layer 2→3: 64×10 + 10     = 650
Total: 109,386 parameters
```

---

## Backpropagation Detail

### Intuisi Backpropagation:

> "Siapa yang paling berkontribusi pada error? Beritahu mereka untuk mengubah diri."

### Chain Rule:

```
Misalkan Loss = L, output = a[L]

dL/dW[L-1] = dL/da[L] × da[L]/dz[L] × dz[L]/dW[L-1]
            = dL/da[L] × f'(z[L]) × a[L-1].T

Ini adalah chain rule:
- dL/da[L] = gradient dari loss terhadap output
- da[L]/dz[L] = f'(z[L]) = turunan activation
- dz[L]/dW[L-1] = a[L-1].T = input ke layer ini
```

### Step-by-Step Backprop untuk [2,3,1]:

```
GIVEN:
  z[0] = [z1, z2, z3]     (pre-activation hidden)
  a[0] = [a1, a2, a3]     (post-activation hidden, ReLU)
  z[1] = scalar            (pre-activation output)
  a[1] = scalar            (post-activation output, sigmoid)

STEP 1: Output gradient
  dz[1] = a[1] - y_true        (for sigmoid + binary cross-entropy)
  This simplifies gradient calculation!

STEP 2: W[1] and b[1] gradients
  dW[1] = a[0].T × dz[1]       (3×1 matrix)
         = [a1, a2, a3].T × [dz[1]]
         = [[a1×dz[1]], [a2×dz[1]], [a3×dz[1]]]

  db[1] = dz[1]                 (scalar)

STEP 3: Propagate gradient to hidden layer
  da[0] = dz[1] × W[1].T        (1×3 matrix)
        = [dz[1]×w1, dz[1]×w2, dz[1]×w3]

  dz[0] = da[0] * ReLU'(z[0])   (element-wise)
        = [dz1, dz2, dz3] where:
          dzi = dai × (1 if zi > 0 else 0)

STEP 4: W[0] and b[0] gradients
  dW[0] = X.T × dz[0]           (2×3 matrix)

  db[0] = dz[0]                  (1×3 matrix)

STEP 5: Update (learning_rate = η)
  W[0] = W[0] - η × dW[0]
  b[0] = b[0] - η × db[0]
  W[1] = W[1] - η × dW[1]
  b[1] = b[1] - η × db[1]
```

### Gradient Vanishing/Exploding Problem:

```
Dengan banyak layer:
  dL/dW[0] = dL/da[L] × Π(da[l]/dz[l]) × ...

Jika activation = sigmoid (turunan max 0.25):
  Setiap layer mengalikan gradient dengan max 0.25
  10 layer: 0.25^10 = 9.5×10⁻⁷ (gradient HAMPIR NOL = vanishing!)

Solusi:
  - ReLU (turunan = 0 atau 1, tidak memperkecil)
  - Batch Normalization
  - Residual connections (ResNet)
  - Proper weight initialization (He/Xavier)
```

---

## Activation Functions

### Perbandingan:

| Nama | Rumus | Range | Turunan | Penggunaan |
|------|-------|-------|---------|------------|
| **ReLU** | max(0, x) | [0, ∞) | 0 or 1 | Hidden layer (default) |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | f(1-f) | Binary output |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | 1-f² | Hidden (jarang) |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1), Σ=1 | complex | Multi-class output |
| **Leaky ReLU** | max(αx, x)⁺ | (-∞, ∞) | 1 or α | Hidden (fix dying ReLU) |
| **ELU** | x if x>0, α(eˣ-1) | (-α, ∞) | 1 or f+α | Hidden |
| **Linear** | x | (-∞, ∞) | 1 | Regression output |

### Visualisasi Activation Functions:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# ReLU
axes[0, 0].plot(x, np.maximum(0, x), 'b-', linewidth=2)
axes[0, 0].set_title('ReLU: max(0, x)')
axes[0, 0].grid(True)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

# Sigmoid
axes[0, 1].plot(x, 1 / (1 + np.exp(-x)), 'r-', linewidth=2)
axes[0, 1].set_title('Sigmoid: 1/(1+e^(-x))')
axes[0, 1].grid(True)

# Tanh
axes[0, 2].plot(x, np.tanh(x), 'g-', linewidth=2)
axes[0, 2].set_title('Tanh')
axes[0, 2].grid(True)

# Leaky ReLU
alpha = 0.1
axes[1, 0].plot(x, np.where(x > 0, x, alpha * x), 'm-', linewidth=2)
axes[1, 0].set_title(f'Leaky ReLU (α={alpha})')
axes[1, 0].grid(True)

# ELU
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
axes[1, 1].plot(x, elu(x), 'c-', linewidth=2)
axes[1, 1].set_title('ELU')
axes[1, 1].grid(True)

# Softmax (2D example)
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
x_softmax = np.linspace(-2, 2, 5)
sm = softmax(x_softmax)
axes[1, 2].bar(range(len(sm)), sm, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
axes[1, 2].set_title('Softmax (example)')
axes[1, 2].set_xlabel('Class')

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Optimizers

### Gradient Descent Variants:

| Optimizer | Cara Kerja | Kelebihan | Kekurangan |
|-----------|-----------|-----------|-----------|
| **SGD** | W -= η × ∇L | Sederhana | Lambat, oscillating |
| **SGD+Momentum** | v = βv + ∇L; W -= η×v | Lebih halus | Perlu set momentum |
| **AdaGrad** | W -= η/(√G+ε) × ∇L | Adaptive LR | LR menurun terus |
| **RMSProp** | Fix AdaGrad LR decay | Balanced | Perlu set decay |
| **Adam** | Gabungan Momentum + RMSProp | Terbaik umum | Perlu set β1, β2 |

### Adam Optimizer (Paling Populer):

```
m = β1 × m + (1-β1) × gradient       (1st moment - momentum)
v = β2 × v + (1-β2) × gradient²      (2nd moment - adaptive LR)
m_hat = m / (1 - β1^t)                (bias correction)
v_hat = v / (1 - β2^t)                (bias correction)
W = W - lr × m_hat / (√v_hat + ε)     (update)

Default: β1=0.9, β2=0.999, ε=1e-8, lr=0.001
```

---

## Pro dan Kontra

### Pro:
1. **Powerful** - Bisa mempelajari pola sangat kompleks
2. **Universal Approximator** - 1 hidden layer cukup untuk approx fungsi apapun
3. **Feature learning** - Otomatis belajar representasi fitur
4. **Scalable** - Performa meningkat dengan data
5. **Versatile** - Classification, regression, generation, dll
6. **Transfer learning** - Bisa reuse model yang sudah ditraining

### Kontra:
1. **Black box** - Sulit diinterpretasi (mengapa model memutuskan X?)
2. **Data hungry** - Butuh banyak data untuk performa baik
3. **Komputasi berat** - Butuh GPU untuk model besar
4. **Hyperparameter banyak** - Arsitektur, LR, batch size, epoch, dll
5. **Overfitting** - Sangat rentan, perlu regularization
6. **Tidak konvergen pasti** - Bisa terjebak di local minima

### Regularization Techniques:

| Teknik | Cara | Parameter |
|--------|------|-----------|
| **L2 (Ridge)** | Tambahkan λΣw² ke loss | alpha (λ) |
| **L1 (Lasso)** | Tambahkan λΣ|w| ke loss | alpha (λ) |
| **Dropout** | Random matikan neuron saat training | rate (0.2-0.5) |
| **Early Stopping** | Stop saat validation loss naik | patience |
| **Batch Normalization** | Normalisasi output setiap layer | momentum |
| **Data Augmentation** | Perbanyak data secara artifisial | - |

### Tips Praktis:

| Problem | Solusi |
|---------|--------|
| Loss tidak turun | Cek: learning rate, data, gradient flow |
| Overfitting | Dropout, L2, early stopping, more data |
| Vanishing gradient | ReLU, BatchNorm, residual connections |
| Exploding gradient | Gradient clipping, smaller LR, BatchNorm |
| Training lambat | Batch size, optimizer, GPU |
| Tak pilih arsitektur | Start simple, add complexity gradually |