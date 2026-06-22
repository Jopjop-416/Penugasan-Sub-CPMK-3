# Sistem Case-Based Reasoning (CBR) - Tindak Pidana Disersi

Implementasi sistem Case-Based Reasoning berbasis Python untuk mendukung analisis putusan pengadilan militer, khususnya tindak pidana Disersi (Pasal 87 KUHPM).

## Ringkasan

- Domain: Pidana Militer, tindak pidana Disersi
- Jumlah data: 34 putusan
- Alur kerja: preprocessing, case representation, retrieval, solution reuse, evaluation
- Bahasa dan tools: Python, pandas, scikit-learn, pdfminer.six, joblib

## Struktur Folder

```text
cbr-desersi/
|-- data/
|   |-- pdf/
|   |-- raw/
|   |-- processed/
|   |-- eval/
|   `-- results/
|-- logs/
|-- models/
|-- notebooks/
|-- 01_preprocessing.py
|-- python 02_case_representation.py
|-- python 03_retrieval.py
|-- python 04_predict.py
|-- python 05_evaluation.py
|-- cbr_text.py
|-- requirements.txt
`-- README.md
```

## Instalasi

1. Buat environment Python yang sesuai.
2. Install dependensi:

```bash
pip install -r requirements.txt
```

## Menjalankan Pipeline

Jalankan script secara berurutan:

### One-Click Runner

Di Windows, jalankan salah satu ini dari root repository:

```bash
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

atau:

```bash
run_all.bat
```

Kalau ingin melewati preprocessing karena `data/raw/` sudah siap:

```bash
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -SkipPreprocessing
```

### 1. Preprocessing

```bash
python 01_preprocessing.py
```

Output:

- `data/raw/case_*.txt`
- `logs/cleaning.log`

### 2. Case Representation

```bash
python "python 02_case_representation.py"
```

Output:

- `data/processed/cases.csv`
- `data/processed/cases.json`
- `logs/representation.log`

### 3. Case Retrieval

```bash
python "python 03_retrieval.py"
```

Output:

- `models/tfidf_vectorizer.pkl`
- `models/tfidf_matrix.pkl`
- `models/case_ids.pkl`
- `models/svm_model.pkl` jika model berhasil dilatih
- `data/eval/queries.json`
- `data/eval/retrieval_initial.json`
- `logs/retrieval.log`

### 4. Case Solution Reuse

```bash
python "python 04_predict.py"
```

Output:

- `data/results/predictions.csv`
- `logs/predict.log`

### 5. Model Evaluation

```bash
python "python 05_evaluation.py"
```

Output:

- `data/eval/retrieval_metrics.csv`
- `data/eval/prediction_metrics.csv`
- `logs/evaluation.log`

## Notebook

Lihat `notebooks/CBR_pipeline_overview.ipynb` untuk ringkasan pipeline dan alur kerja proyek.

## Catatan Teknis

- Pipeline ini menggunakan TF-IDF dan cosine similarity untuk retrieval.
- Query prediksi dan evaluasi memakai preprocessing yang sama dengan case base agar hasilnya konsisten.
- Ekstraksi metadata di tahap representation memakai beberapa fallback regex untuk menangani variasi format putusan.
