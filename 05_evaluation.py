"""
05_evaluation.py — Tahap 5: Model Evaluation
=============================================
Sistem CBR Tindak Pidana Disersi (Pasal 87 KUHPM)
Mata Kuliah Penalaran Komputer — Semester Genap 2025/2026

Alur kerja:
1. Load queries.json (ground truth) dan retrieval_initial.json (hasil Tahap 3)
2. Evaluasi retrieval: Hit@K, Accuracy, Precision, Recall, F1-score
3. Evaluasi prediksi solusi (dari predictions.csv Tahap 4)
4. Analisis kegagalan (error analysis)
5. Simpan: data/eval/retrieval_metrics.csv dan data/eval/prediction_metrics.csv
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from cbr_text import preprocess_text

                                                              
                  
                                                              
BASE_DIR    = r"C:\file\cbr-desersi"
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")
EVAL_DIR    = os.path.join(DATA_DIR, "eval")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

CASES_JSON              = os.path.join(DATA_DIR, "processed", "cases.json")
QUERIES_JSON            = os.path.join(EVAL_DIR, "queries.json")
RETRIEVAL_INITIAL_JSON  = os.path.join(EVAL_DIR, "retrieval_initial.json")
PREDICTIONS_CSV         = os.path.join(RESULTS_DIR, "predictions.csv")

TFIDF_VEC_PATH      = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH   = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
CASE_IDS_PATH       = os.path.join(MODEL_DIR, "case_ids.pkl")

RETRIEVAL_METRICS_CSV   = os.path.join(EVAL_DIR, "retrieval_metrics.csv")
PREDICTION_METRICS_CSV  = os.path.join(EVAL_DIR, "prediction_metrics.csv")
LOG_FILE                = os.path.join(LOG_DIR, "evaluation.log")

                                                              
               
                                                              
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


                                                              
              
                                                              

def load_queries(path: str) -> list:
    """Load queries.json — list of {query_id, query, ground_truth_case_id}."""
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    logger.info("Loaded %d query dari %s", len(queries), path)
    return queries


def load_retrieval_results(path: str) -> dict:
    """
    Load retrieval_initial.json.
    Format yang didukung:
      - dict  {query_id: {top_k: [...], ground_truth: ...}}
      - list  [{query_id, top_k, ground_truth}, ...]
    Mengembalikan dict {query_id: data}.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        results = {item["query_id"]: item for item in raw}
    elif isinstance(raw, dict):
        results = raw
    else:
        raise ValueError("Format retrieval_initial.json tidak dikenali.")

    logger.info("Loaded retrieval results untuk %d query dari %s", len(results), path)
    return results


def load_cases(path: str) -> dict:
    """Load cases.json → dict {case_id: data}."""
    with open(path, "r", encoding="utf-8") as f:
        cases_list = json.load(f)
    return {c["case_id"]: c for c in cases_list}


def load_models():
    """Load TF-IDF model dari models/."""
    vectorizer   = joblib.load(TFIDF_VEC_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    case_ids     = joblib.load(CASE_IDS_PATH)
    return vectorizer, tfidf_matrix, case_ids


                                                              
                                              
                                                              

def retrieve(query: str, vectorizer, tfidf_matrix, case_ids: list, k: int = 5):
    query_vec  = vectorizer.transform([preprocess_text(query)])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_idx  = sim_scores.argsort()[::-1][:k]
    return [(case_ids[i], float(sim_scores[i])) for i in top_k_idx]


                                                              
                       
                                                              

def eval_retrieval(queries: list, retrieval_results: dict, k_values=(1, 3, 5)):
    """
    Hitung metrik retrieval untuk berbagai nilai k.

    Metrik per query:
    - Hit@K   : 1 jika ground_truth ada di top-K, else 0
    - Rank    : posisi ground_truth di top-K (0 jika tidak ditemukan)

    Metrik agregat (diperlakukan sebagai binary classification):
    - Accuracy  : proporsi query dengan Hit@K = 1
    - Precision : TP / (TP + FP) — dalam konteks IR = sama dengan accuracy di Hit@1
    - Recall    : TP / (TP + FN) — dalam konteks ini = Hit Rate
    - F1-score  : harmonic mean Precision & Recall
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUASI RETRIEVAL")
    logger.info("=" * 60)

    per_query_rows = []                    
    summary_rows   = []                   

    for k in k_values:
        y_true = []                            
        y_pred = []

        failed_queries = []

        for q in queries:
            qid = q.get("query_id") or q.get("id")

            if qid not in retrieval_results:
                logger.warning("  Query %s tidak ditemukan di retrieval results, skip.", qid)
                continue

            res     = retrieval_results[qid]
                                                                                                    
            ground_truth = (
                q.get("ground_truth_case_id")
                or q.get("ground_truth")
                or res.get("ground_truth")
            )
                                                                          
            raw_top = (
                res.get("top_5_ids")                                    
                or res.get("top_k")
                or res.get("top_k_results")
                or []
            )
                                                                     
            if raw_top and isinstance(raw_top[0], (list, tuple)):
                top_k_ids = [item[0] for item in raw_top[:k]]
            else:
                top_k_ids = raw_top[:k]

            hit = 1 if ground_truth in top_k_ids else 0
            y_true.append(1)                                                  
            y_pred.append(hit)

            rank = (top_k_ids.index(ground_truth) + 1) if hit else None

            per_query_rows.append({
                "query_id":     qid,
                "ground_truth": ground_truth,
                "k":            k,
                "hit":          hit,
                "rank":         rank if rank else "N/A",
                "top_k_ids":    ",".join(top_k_ids),
            })

            if not hit:
                failed_queries.append({
                    "query_id":     qid,
                    "ground_truth": ground_truth,
                    "retrieved":    top_k_ids,
                })

        if not y_true:
            logger.warning("Tidak ada data untuk k=%d", k)
            continue

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        hit_rate = sum(y_pred) / len(y_pred)

        logger.info("")
        logger.info("--- Metrik @ k=%d ---", k)
        logger.info("  Total query : %d", len(y_pred))
        logger.info("  Hit@%d      : %d/%d (%.1f%%)", k, sum(y_pred), len(y_pred), hit_rate * 100)
        logger.info("  Accuracy    : %.4f", acc)
        logger.info("  Precision   : %.4f", prec)
        logger.info("  Recall      : %.4f", rec)
        logger.info("  F1-score    : %.4f", f1)

        if failed_queries:
            logger.info("  Kegagalan @ k=%d:", k)
            for fq in failed_queries:
                logger.info(
                    "    [%s] ground_truth=%s | retrieved=%s",
                    fq["query_id"], fq["ground_truth"], fq["retrieved"],
                )

        summary_rows.append({
            "k":         k,
            "total_query": len(y_pred),
            "hit":       sum(y_pred),
            "hit_rate":  round(hit_rate, 4),
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1_score":  round(f1, 4),
        })

    return pd.DataFrame(per_query_rows), pd.DataFrame(summary_rows)


                                                              
                                                                        
                                                              

def eval_retrieval_live(queries: list, vectorizer, tfidf_matrix, case_ids: list,
                        k_values=(1, 3, 5)):
    """
    Jalankan ulang retrieve() untuk setiap query dan hitung metrik.
    Digunakan jika retrieval_initial.json tidak memiliki semua field yang diperlukan.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUASI RETRIEVAL (LIVE RE-RUN)")
    logger.info("=" * 60)

    live_results = {}
    for q in queries:
        qid   = q.get("query_id") or q.get("id")
        query = q.get("query") or q.get("query_text") or q.get("text") or ""
        gt    = q.get("ground_truth_case_id") or q.get("ground_truth")

        top_k_raw = retrieve(query, vectorizer, tfidf_matrix, case_ids, k=max(k_values))
        live_results[qid] = {
            "top_k":        [[cid, sim] for cid, sim in top_k_raw],
            "ground_truth": gt,
        }
        logger.info("  [%s] ground_truth=%s | top-%d=%s",
                    qid, gt,
                    max(k_values),
                    [cid for cid, _ in top_k_raw[:max(k_values)]])

    return live_results


                                                              
                             
                                                              

def eval_predictions(predictions_csv: str, cases_dict: dict):
    """
    Evaluasi kualitas prediksi solusi dari predictions.csv (Tahap 4).

    Karena prediksi solusi bersifat teks (bukan label kategori),
    evaluasi dilakukan secara:
    - Kuantitatif: apakah prediksi = pidana_pokok kasus ground_truth (exact match)
    - Kualitatif: tampilkan perbandingan
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUASI PREDIKSI SOLUSI")
    logger.info("=" * 60)

    if not os.path.exists(predictions_csv):
        logger.warning("File predictions.csv tidak ditemukan: %s", predictions_csv)
        logger.warning("Pastikan Tahap 4 (04_predict.py) sudah dijalankan lebih dulu.")
        return pd.DataFrame()

    df = pd.read_csv(predictions_csv, encoding="utf-8-sig")
    logger.info("Loaded %d prediksi dari %s", len(df), predictions_csv)

    rows = []
    for _, row in df.iterrows():
        qid         = row.get("query_id", "?")
        pred_sol    = str(row.get("predicted_solution", "")).strip()
        top5_ids    = str(row.get("top_5_case_ids", "")).split(",")
        top1_id     = top5_ids[0].strip() if top5_ids else ""

                                                 
        ref_case    = cases_dict.get(top1_id, {})
        ref_pidana  = str(ref_case.get("pidana_pokok", "")).strip()
        ref_amar    = str(ref_case.get("amar_putusan", ""))[:200].strip()

                                                               
        exact_match = (pred_sol.lower() == ref_pidana.lower()) if ref_pidana else False

        logger.info("")
        logger.info("  [%s]", qid)
        logger.info("    Prediksi       : %s", pred_sol[:100])
        logger.info("    Top-1 kasus    : %s", top1_id)
        logger.info("    Pidana top-1   : %s", ref_pidana[:100])
        logger.info("    Exact Match    : %s", "✓ YA" if exact_match else "✗ TIDAK")

        rows.append({
            "query_id":        qid,
            "top_1_case_id":   top1_id,
            "predicted_solution": pred_sol,
            "reference_pidana": ref_pidana,
            "exact_match":     exact_match,
        })

    df_out = pd.DataFrame(rows)
    exact_count = df_out["exact_match"].sum()
    logger.info("")
    logger.info("Exact Match Prediksi: %d/%d (%.1f%%)",
                exact_count, len(df_out),
                100 * exact_count / len(df_out) if len(df_out) > 0 else 0)

    return df_out


                                                              
                   
                                                              

def error_analysis(per_query_df: pd.DataFrame, queries: list, cases_dict: dict, k: int = 5):
    """
    Analisis kasus-kasus yang gagal di-retrieve pada k tertentu.
    Tampilkan pola kegagalan beserta rekomendasi.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALISIS KEGAGALAN (ERROR ANALYSIS) @ k=%d", k)
    logger.info("=" * 60)

                                                           
    if per_query_df.empty or "k" not in per_query_df.columns:
        logger.warning("  per_query_df kosong atau tidak memiliki kolom 'k', skip error analysis.")
        return

                             
    df_k    = per_query_df[per_query_df["k"] == k]
    df_fail = df_k[df_k["hit"] == 0]

    if df_fail.empty:
        logger.info("  Tidak ada kegagalan pada k=%d — Hit Rate = 100%%!", k)
        logger.info("")
        logger.info("  Rekomendasi Perbaikan (preventif):")
        logger.info("  1. Tambahkan lebih banyak kasus (>50) agar case base lebih representatif.")
        logger.info("  2. Coba IndoBERT embedding untuk representasi semantik yang lebih kaya.")
        logger.info("  3. Implementasikan BM25 sebagai alternatif TF-IDF untuk perbandingan.")
        logger.info("  4. Tingkatkan kualitas preprocessing: stemming Bahasa Indonesia (Sastrawi).")
        return

    logger.info("  Jumlah kegagalan: %d dari %d query", len(df_fail), len(df_k))
    logger.info("")

    for _, row in df_fail.iterrows():
        qid  = row["query_id"]
        gt   = row["ground_truth"]
        retr = row["top_k_ids"]

        gt_case = cases_dict.get(gt, {})
        logger.info("  Query  : %s", qid)
        logger.info("  Ground truth case  : %s (%s)", gt, gt_case.get("terdakwa", "?"))
        logger.info("  Kasus yang retrieved: %s", retr)
        logger.info("  Kemungkinan penyebab: kasus referensi memiliki teks unik atau")
        logger.info("    metadata yang tidak ter-capture oleh TF-IDF n-gram(1,2).")
        logger.info("")

    logger.info("  Rekomendasi Perbaikan:")
    logger.info("  1. Tambah fitur: ekstrak lama_disersi dan pidana_pokok sebagai fitur numerik terpisah.")
    logger.info("  2. Gunakan IndoBERT (indobenchmark/indobert-base-p1) untuk embedding semantik.")
    logger.info("  3. Perluas dataset — minimal 50 putusan untuk distribusi kelas lebih merata.")
    logger.info("  4. Terapkan stemming Bahasa Indonesia (library Sastrawi) sebelum TF-IDF.")
    logger.info("  5. Tambah query uji yang lebih beragam untuk menguji edge-case.")


                                                              
      
                                                              

def main():
    logger.info("=" * 60)
    logger.info("TAHAP 5: MODEL EVALUATION")
    logger.info("Mulai: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

                       
    queries      = load_queries(QUERIES_JSON)
    cases_dict   = load_cases(CASES_JSON)
    vectorizer, tfidf_matrix, case_ids = load_models()

                                              
    try:
        retrieval_results = load_retrieval_results(RETRIEVAL_INITIAL_JSON)
        use_live = False
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning("Tidak bisa load retrieval_initial.json (%s), menjalankan ulang retrieve().", e)
        use_live = True

                                           
    if use_live:
        retrieval_results = eval_retrieval_live(
            queries, vectorizer, tfidf_matrix, case_ids, k_values=(1, 3, 5)
        )
    else:
                                                              
        for q in queries:
            qid = q.get("query_id") or q.get("id")
            gt  = q.get("ground_truth_case_id") or q.get("ground_truth")
            if qid in retrieval_results and "ground_truth" not in retrieval_results[qid]:
                retrieval_results[qid]["ground_truth"] = gt

                                
    per_query_df, summary_df = eval_retrieval(
        queries, retrieval_results, k_values=(1, 3, 5)
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("TABEL RINGKASAN METRIK RETRIEVAL")
    logger.info("=" * 60)
    logger.info("\n%s", summary_df.to_string(index=False))

                                          
    summary_df.to_csv(RETRIEVAL_METRICS_CSV, index=False, encoding="utf-8-sig")
    logger.info("Retrieval metrics disimpan: %s", RETRIEVAL_METRICS_CSV)

                                     
    error_analysis(per_query_df, queries, cases_dict, k=5)

                                                     
    df_pred_eval = eval_predictions(PREDICTIONS_CSV, cases_dict)

    if not df_pred_eval.empty:
        df_pred_eval.to_csv(PREDICTION_METRICS_CSV, index=False, encoding="utf-8-sig")
        logger.info("Prediction metrics disimpan: %s", PREDICTION_METRICS_CSV)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Tahap 5 selesai.")
    logger.info("Output:")
    logger.info("  - %s", RETRIEVAL_METRICS_CSV)
    if not df_pred_eval.empty:
        logger.info("  - %s", PREDICTION_METRICS_CSV)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()