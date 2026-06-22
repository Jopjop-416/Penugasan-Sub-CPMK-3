# Rubric Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the CBR repository so it better satisfies the grading rubric for preprocessing, case representation, retrieval, reuse, and repository documentation.

**Architecture:** Keep the existing 5-stage script pipeline intact, but centralize shared text preprocessing, harden metadata extraction fallbacks, and add missing repo documentation artifacts. This preserves the current workflow while improving rubric-visible quality indicators and reproducibility.

**Tech Stack:** Python, pandas, scikit-learn, pdfminer.six, joblib, Jupyter Notebook markdown cells, plain Markdown documentation.

---

### Task 1: Add shared text utilities

**Files:**
- Create: `C:/file/cbr-desersi/cbr_text.py`
- Modify: `C:/file/cbr-desersi/python 03_retrieval.py`
- Modify: `C:/file/cbr-desersi/python 04_predict.py`
- Modify: `C:/file/cbr-desersi/python 05_evaluation.py`

- [ ] **Step 1: Add shared text normalization and TF-IDF preprocessing helpers**

```python
import re

STOPWORDS_ID = {...}

def normalize_text(text: str) -> str:
    text = text.translate(...)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def preprocess_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^#.*?$", " ", text, flags=re.MULTILINE)
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS_ID and len(w) > 2]
    return " ".join(words)
```

- [ ] **Step 2: Switch retrieval, prediction, and evaluation to shared preprocessing**

```python
from cbr_text import preprocess_text

query_vec = vectorizer.transform([preprocess_text(query)])
```

- [ ] **Step 3: Verify imports and retrieval still run**

Run:
```bash
python python 03_retrieval.py
python python 04_predict.py
python python 05_evaluation.py
```
Expected: scripts complete and reuse the same query preprocessing path.

### Task 2: Harden metadata extraction

**Files:**
- Modify: `C:/file/cbr-desersi/python 02_case_representation.py`

- [ ] **Step 1: Normalize extracted text before regex parsing**

```python
from cbr_text import normalize_text

with open(txt_path, "r", encoding="utf-8") as f:
    text = normalize_text(f.read())
```

- [ ] **Step 2: Expand regex fallbacks for key metadata fields**

```python
def extract_tanggal_putusan(text: str) -> str:
    patterns = [
        rf"(?:hari\s+\w+\s+)?tanggal\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"pada\s+tanggal\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"diputuskan.*?tanggal\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
    ]
```

```python
def extract_amar_putusan(text: str) -> str:
    m = re.search(r"MENGADILI[:\s]*([\s\S]{50,1200})(?=Demikian\s+diputuskan|Hakim|Panitera|$)", text, re.IGNORECASE)
```

- [ ] **Step 3: Verify processed dataset coverage**

Run:
```bash
python python 02_case_representation.py
```
Expected: `data/processed/cases.csv` and `data/processed/cases.json` regenerate with improved non-empty coverage for `tanggal_putusan`, `kesatuan`, `pangkat_nrp`, and `amar_putusan`.

### Task 3: Improve preprocessing readability and validation notes

**Files:**
- Modify: `C:/file/cbr-desersi/01_preprocessing.py`

- [ ] **Step 1: Keep the current output shape but clarify validation logic**

```python
min_words = 100
required_keywords = ["terdakwa", "desersi", "putusan", "mengadili", "militer", "pidana"]
```

- [ ] **Step 2: Add clearer logging about why a file is skipped**

```python
logger.warning(f"[SKIP] {filename} - terlalu pendek ({word_count} kata)")
logger.warning(f"[SKIP] {filename} - tidak ada keyword ditemukan sama sekali")
```

- [ ] **Step 3: Re-run preprocessing only if needed**

Run:
```bash
python 01_preprocessing.py
```
Expected: `data/raw/*.txt` remains available and the log is easier to audit.

### Task 4: Add documentation and notebook artifacts

**Files:**
- Create: `C:/file/cbr-desersi/README.md`
- Create: `C:/file/cbr-desersi/requirements.txt`
- Create: `C:/file/cbr-desersi/notebooks/CBR_pipeline_overview.ipynb`

- [ ] **Step 1: Write a repo README with setup and execution steps**

```markdown
# Sistem Case-Based Reasoning (CBR) - Tindak Pidana Disersi

## Installation
pip install -r requirements.txt

## Pipeline
python 01_preprocessing.py
python python 02_case_representation.py
python python 03_retrieval.py
python python 04_predict.py
python python 05_evaluation.py
```

- [ ] **Step 2: Add a minimal requirements file**

```text
pdfminer.six
pandas
scikit-learn
joblib
```

- [ ] **Step 3: Add a notebook that explains the pipeline**

```json
{
  "cells": [
    {"cell_type": "markdown", "source": ["# CBR Pipeline Overview\\n"]},
    {"cell_type": "markdown", "source": ["This notebook summarizes the 5-stage pipeline.\\n"]}
  ]
}
```

- [ ] **Step 4: Verify files exist**

Run:
```bash
Get-ChildItem README.md, requirements.txt, notebooks
```
Expected: README, requirements, and at least one notebook artifact are present.

### Task 5: Re-run and inspect outputs

**Files:**
- Regenerate: `C:/file/cbr-desersi/data/processed/cases.csv`
- Regenerate: `C:/file/cbr-desersi/data/processed/cases.json`
- Regenerate: `C:/file/cbr-desersi/data/eval/queries.json`
- Regenerate: `C:/file/cbr-desersi/data/eval/retrieval_initial.json`
- Regenerate: `C:/file/cbr-desersi/data/results/predictions.csv`
- Regenerate: `C:/file/cbr-desersi/data/eval/retrieval_metrics.csv`
- Regenerate: `C:/file/cbr-desersi/data/eval/prediction_metrics.csv`

- [ ] **Step 1: Run the full pipeline**

```bash
python 01_preprocessing.py
python python 02_case_representation.py
python python 03_retrieval.py
python python 04_predict.py
python python 05_evaluation.py
```

- [ ] **Step 2: Check the resulting coverage and metrics files**

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/processed/cases.csv")
print(df.notna().mean().sort_values(ascending=False))
PY
```

- [ ] **Step 3: Summarize rubric impact**

Expected outcome: better repository/documentation score, stronger case representation score, and at least maintained retrieval/reuse/evaluation performance.

