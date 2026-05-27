# README — Safety Incident NLP Classification Model

`` READ THE TESTING_GUIDE.md TO RUNNING THE API FOR TESTING ``

## Deskripsi Project

Model ini digunakan untuk melakukan klasifikasi laporan kecelakaan kerja (*safety incident report*) berbasis NLP (*Natural Language Processing*).

Input model berupa narasi/laporan insiden kerja dalam bentuk teks bebas, kemudian model akan memprediksi kategori sumber laporan:

- `osha`
- `industrial_safety`

Model dibangun menggunakan:

- TF-IDF Vectorizer
- SGDClassifier (Linear SVM)
- Scikit-learn Pipeline

---

# Tujuan Model

Model digunakan untuk:

- klasifikasi otomatis laporan K3
- membantu kategorisasi laporan incident
- integrasi ke backend/API/interface
- inferensi cepat dan ringan

---

# Struktur Dataset

Dataset memiliki 2 kolom utama:

| Kolom | Deskripsi |
|---|---|
| `text_description` | Narasi/laporan kecelakaan kerja |
| `source` | Label kategori (`osha` / `industrial_safety`) |

## Contoh Data

| text_description | source |
|---|---|
| Worker slipped while carrying equipment | osha |
| Sodium sulphide exposure during maintenance | industrial_safety |

---

# Arsitektur Model

Pipeline model:

```text
Input Text
    ↓
TF-IDF Vectorizer
    ↓
SGDClassifier
    ↓
Prediction
```

---

# Detail Model

## TF-IDF Vectorizer

Mengubah text menjadi representasi numerik.

### Konfigurasi

```python
TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5)
)
```

Menggunakan *character n-grams* untuk:

- menangani typo
- menangani istilah teknis industri
- meningkatkan robustness terhadap variasi text

---

## SGDClassifier

Digunakan untuk klasifikasi text ringan dan cepat.

### Konfigurasi

```python
SGDClassifier(
    loss='hinge',
    class_weight='balanced',
    random_state=42
)
```

`class_weight='balanced'` digunakan karena dataset tidak seimbang:

- `osha` mendominasi dataset
- `industrial_safety` jumlahnya lebih sedikit

---

# Pembagian Dataset

Dataset dibagi menjadi:

- 80% training
- 20% testing

Menggunakan:

```python
train_test_split(
    test_size=0.2,
    stratify=df['source']
)
```

`stratify` digunakan agar distribusi kelas tetap seimbang pada train dan test set.

---

# Training Model

## Jalankan Training

```python
pipeline.fit(X_train, y_train)
```

---

# Evaluasi Model

Evaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1-Score

## Contoh

```text
Accuracy: 0.95
```

---

# Menyimpan Model

Model disimpan menggunakan Joblib:

```python
joblib.dump(pipeline, 'safety_model.joblib')
```

Output:

```text
safety_model.joblib
```

File ini digunakan untuk deployment dan integrasi backend.

---

# Cara Menggunakan Model di Backend

## Install Dependency

```bash
pip install scikit-learn joblib
```

---

## Load Model

```python
import joblib

model = joblib.load('safety_model.joblib')
```

---

## Melakukan Prediksi

```python
text = [
    "Worker slipped while carrying equipment"
]

prediction = model.predict(text)

print(prediction[0])
```

## Contoh Output

```text
osha
```

---

# Contoh Integrasi API Sederhana

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('safety_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
    text = data['text']

    prediction = model.predict([text])

    return jsonify({
        "prediction": prediction[0]
    })

app.run(debug=True)
```

---

# Contoh Request API

```json
{
  "text": "Worker injured during welding process"
}
```

---

# Contoh Response API

```json
{
  "prediction": "osha"
}
```

# Keterbatasan Model

- hanya dapat memprediksi:
  - `osha`
  - `industrial_safety`
- tidak mendeteksi relevansi text
- tidak cocok untuk text di luar domain keselamatan kerja
- model dapat bias jika dataset tidak seimbang

---

# File Penting

| File | Fungsi |
|---|---|
| `train.ipynb` | Notebook training model |
| `safety_model.joblib` | Saved model |
| `app.py` | Test File for Integrasi backend/API |
| `nlp.csv` | Dataset training |


# Other Docs

https://colab.research.google.com/drive/1EZB7gLinir6aSRa8I0NZgmS0foRo7aUE?usp=sharing

