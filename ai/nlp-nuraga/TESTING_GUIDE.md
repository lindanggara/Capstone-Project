# FastAPI Testing Guide

## 1. Pastikan File Berikut Ada

```text
project/
│
├── app.py
├── safety_model.joblib
└── requirements.txt
```

---

## 2. Install Dependency

```bash
pip install fastapi uvicorn scikit-learn joblib
```

Atau:

```bash
pip install -r requirements.txt
```

---

## 3. Jalankan API

```bash
python app.py
```

Jika berhasil:

```text
Uvicorn running on http://0.0.0.0:8000
```

---

## 4. Buka Swagger Docs

Buka browser:

```text
http://127.0.0.1:8000/docs
```

---

## 5. Test Endpoint

Pilih endpoint:

```text
POST /predict
```

Klik:

```text
Try it out
```

Isi request body:

```json
{
  "text": "Worker injured during welding process"
}
```

Klik:

```text
Execute
```

---

## 6. Contoh Response

```json
{
  "success": true,
  "input_text": "Worker injured during welding process",
  "prediction": "osha"
}
```

---

## Catatan

Kemungkinan output model:

- `osha`
- `industrial_safety`

Model hanya cocok untuk text/laporan keselamatan kerja.