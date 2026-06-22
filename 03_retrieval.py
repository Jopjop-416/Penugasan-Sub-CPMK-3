"""
Tahap 3: Case Retrieval
CBR Sistem - Pidana Militer Disersi
Script: 03_retrieval.py

Fungsi:
- Load data dari cases.json
- Vectorisasi teks dengan TF-IDF
- Split data train/test (80:20)
- Training model SVM untuk klasifikasi/retrieval
- Fungsi retrieve(query, k=5) berbasis cosine similarity
- Evaluasi awal dengan query uji
- Simpan query uji ke data/eval/queries.json
"""

import json
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from datetime import datetime

from cbr_text import preprocess_text

                                                              
                  
                                                              
BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR      = BASE_DIR / "data" / "eval"
MODEL_DIR     = BASE_DIR / "models"
LOG_DIR       = BASE_DIR / "logs"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

                                                              
               
                                                              
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "retrieval.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


                                                              
           
                                                              

def load_cases() -> list:
    json_path = PROCESSED_DIR / "cases.json"
    if not json_path.exists():
        logger.error(f"File tidak ditemukan: {json_path}")
        logger.error("Jalankan dulu 02_case_representation.py")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    logger.info(f"Loaded {len(cases)} kasus dari {json_path}")
    return cases


def build_query_text(case: dict) -> str:
    """
    Gabungkan field penting untuk membentuk teks representasi kasus.
    Lebih kaya dari text_full karena menekankan field kunci.
    """
    parts = []
    if case.get("ringkasan_fakta"):
        parts.append(case["ringkasan_fakta"] * 2)
    if case.get("amar_putusan"):
        parts.append(case["amar_putusan"])
    if case.get("pasal"):
        parts.append(case["pasal"])
    if case.get("lama_disersi"):
        parts.append(f"disersi {case['lama_disersi']}")
    if case.get("pidana_pokok"):
        parts.append(case["pidana_pokok"])
    if case.get("text_full"):
        parts.append(case["text_full"][:3000])
    return " ".join(parts)


                                                              
                      
                                                              

def build_tfidf(cases: list):
    """Bangun TF-IDF vectorizer dari semua kasus."""
    logger.info("Membangun TF-IDF vectors...")

                                       
    texts = [preprocess_text(build_query_text(c)) for c in cases]
    case_ids = [c["case_id"] for c in cases]

                       
    vectorizer = TfidfVectorizer(
        max_features=5000,                                 
        ngram_range=(1, 2),                        
        min_df=1,                                             
        max_df=0.95,                                                  
        sublinear_tf=True,                           
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    logger.info(f"  Jumlah kasus  : {tfidf_matrix.shape[0]}")
    logger.info(f"  Jumlah fitur  : {tfidf_matrix.shape[1]}")

    return vectorizer, tfidf_matrix, texts, case_ids


                                                              
                                     
                                                              

def retrieve(query: str, vectorizer, tfidf_matrix, case_ids: list,
             cases: list, k: int = 5) -> list:
    """
    Retrieve top-k kasus paling mirip dengan query.

    Args:
        query     : teks query kasus baru
        vectorizer: TF-IDF vectorizer yang sudah di-fit
        tfidf_matrix: matrix TF-IDF semua kasus
        case_ids  : list case_id
        cases     : list dict semua kasus (untuk ambil info)
        k         : jumlah kasus yang dikembalikan

    Returns:
        list of dict berisi case_id, similarity score, dan info kasus
    """
                         
    query_processed = preprocess_text(query)

                            
    query_vec = vectorizer.transform([query_processed])

                                                           
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

                          
    top_k_idx = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_k_idx:
        case = cases[idx]
        results.append({
            "case_id"       : case_ids[idx],
            "similarity"    : round(float(similarities[idx]), 4),
            "terdakwa"      : case.get("terdakwa", ""),
            "lama_disersi"  : case.get("lama_disersi", ""),
            "pidana_pokok"  : case.get("pidana_pokok", ""),
            "pidana_tambahan": case.get("pidana_tambahan", ""),
            "no_perkara"    : case.get("no_perkara", ""),
        })

    return results


                                                              
                                
                                                              

def train_svm(tfidf_matrix, cases: list):
    """
    Training SVM untuk klasifikasi berdasarkan label pidana_pokok.
    Label = kategori hukuman (misal: '1 tahun', '1.5 tahun', dll).
    """
    logger.info("\nMempersiapkan label untuk SVM...")

                                  
    labels = []
    valid_idx = []

    for i, case in enumerate(cases):
        pidana = case.get("pidana_pokok", "")
        if not pidana:
            continue

                                                            
        if "tahun" in pidana.lower():
            label = "penjara_dengan_pecat"  if re.search(r"pecat|dipecat|diberhentikan", pidana.lower()) else "penjara_tanpa_pecat"
        else:
            label = "penjara_tanpa_pecat"

        labels.append(label)
        valid_idx.append(i)

    from collections import Counter
    label_counts = Counter(labels)

                                                                          
    majority = label_counts.most_common(1)[0][0]
    labels = [l if label_counts[l] >= 2 else majority for l in labels]
    label_counts = Counter(labels)

    if len(set(labels)) < 2:
        logger.warning("Label kurang beragam untuk SVM, skip training SVM.")
        return None, None, None

                                                
    X = tfidf_matrix[valid_idx]
    y = np.array(labels)

    logger.info(f"Distribusi label:")
    for label, count in label_counts.items():
        logger.info(f"  {label}: {count} kasus")

                            
                                                            
    min_class_count = min(label_counts.values())
    use_stratify = y if min_class_count >= 2 and len(set(labels)) > 1 else None

    if len(y) < 5:
        logger.warning("Data terlalu sedikit untuk split, gunakan semua sebagai train.")
        X_train, X_test = X, X
        y_train, y_test = y, y
        test_idx = valid_idx
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, valid_idx,
            test_size=0.2,
            random_state=42,
            stratify=use_stratify
        )

    logger.info(f"Split data: train={X_train.shape[0]}, test={X_test.shape[0]}")

                  
    logger.info("Training LinearSVC...")
    svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    svm.fit(X_train, y_train)

                       
    train_acc = svm.score(X_train, y_train)
    test_acc  = svm.score(X_test, y_test)
    logger.info(f"SVM Train Accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
    logger.info(f"SVM Test Accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)")

    return svm, y_test, svm.predict(X_test)


                                                              
                
                                                              

def create_test_queries(cases: list) -> list:
    """
    Buat query uji dari sebagian kasus yang ada.
    Ground truth = case_id yang paling relevan.
    """
    queries = []
    complete_cases = [c for c in cases if c.get("terdakwa") and c.get("lama_disersi")]

    for case in complete_cases:
        if len(queries) >= 7:
            break

        query_text = ""
        if case.get("ringkasan_fakta"):
            query_text = case["ringkasan_fakta"][:300]
        elif case.get("text_full"):
            full = case["text_full"]
            mid = len(full) // 4
            query_text = full[mid:mid + 400]

        if not query_text:
            continue

        queries.append({
            "query_id": f"q_{len(queries) + 1:03d}",
            "query_text": query_text,
            "ground_truth_case_id": case["case_id"],
            "ground_truth_terdakwa": case.get("terdakwa", ""),
            "ground_truth_lama_disersi": case.get("lama_disersi", ""),
            "ground_truth_pidana": case.get("pidana_pokok", ""),
        })

    return queries

                                                              
      
                                                              

def main():
    logger.info(f"{'='*60}")
    logger.info(f"TAHAP 3: CASE RETRIEVAL")
    logger.info(f"Mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}")

                  
    cases = load_cases()
    if not cases:
        return

                      
    vectorizer, tfidf_matrix, texts, case_ids = build_tfidf(cases)

                     
    logger.info(f"\n{'='*40}")
    logger.info("TRAINING SVM")
    logger.info(f"{'='*40}")
    svm, y_test, y_pred = train_svm(tfidf_matrix, cases)

                     
    logger.info("\nMenyimpan model...")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(tfidf_matrix, MODEL_DIR / "tfidf_matrix.pkl")
    joblib.dump(case_ids, MODEL_DIR / "case_ids.pkl")
    if svm:
        joblib.dump(svm, MODEL_DIR / "svm_model.pkl")
    logger.info(f"Model disimpan di: {MODEL_DIR}")

                                  
    logger.info(f"\n{'='*40}")
    logger.info("MEMBUAT QUERY UJI")
    logger.info(f"{'='*40}")
    test_queries = create_test_queries(cases)

    queries_path = EVAL_DIR / "queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)
    logger.info(f"Query uji disimpan: {queries_path} ({len(test_queries)} query)")

                                               
    logger.info(f"\n{'='*40}")
    logger.info("UJI FUNGSI RETRIEVE")
    logger.info(f"{'='*40}")

    all_results = []
    hit_count   = 0

    for q in test_queries:
        logger.info(f"\nQuery [{q['query_id']}]: {q['query_text'][:80]}...")
        results = retrieve(
            q["query_text"], vectorizer, tfidf_matrix, case_ids, cases, k=5
        )

                                             
        top5_ids   = [r["case_id"] for r in results]
        is_hit     = q["ground_truth_case_id"] in top5_ids
        hit_count += int(is_hit)

        logger.info(f"  Ground truth : {q['ground_truth_case_id']} ({q['ground_truth_terdakwa']})")
        logger.info(f"  Top-5 result : {top5_ids}")
        logger.info(f"  Hit@5        : {'✓ YA' if is_hit else '✗ TIDAK'}")

        for r in results:
            logger.info(
                f"    [{r['case_id']}] sim={r['similarity']:.4f} | "
                f"{r['terdakwa'] or '?'} | {r['lama_disersi'] or '?'}"
            )

        all_results.append({
            "query_id"     : q["query_id"],
            "query_text"   : q["query_text"][:100],
            "ground_truth" : q["ground_truth_case_id"],
            "top_5_ids"    : top5_ids,
            "top_5_scores" : [r["similarity"] for r in results],
            "hit_at_5"     : is_hit,
        })

                        
    hit_rate = hit_count / len(test_queries) if test_queries else 0
    logger.info(f"\n{'='*60}")
    logger.info(f"RINGKASAN RETRIEVAL")
    logger.info(f"  Total query  : {len(test_queries)}")
    logger.info(f"  Hit@5        : {hit_count}/{len(test_queries)} ({hit_rate*100:.1f}%)")
    logger.info(f"{'='*60}")

                                    
    results_path = EVAL_DIR / "retrieval_initial.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Hasil retrieval disimpan: {results_path}")

    return vectorizer, tfidf_matrix, case_ids, cases, svm


def run_interactive_demo(vectorizer, tfidf_matrix, case_ids, cases):
    """Optional interactive demo for manual exploration."""
    print("\n" + "="*60)
    print("DEMO RETRIEVE — ketik query, ketik 'quit' untuk keluar")
    print("="*60)

    while True:
        try:
            query = input("\nMasukkan query: ").strip()
        except EOFError:
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = retrieve(query, vectorizer, tfidf_matrix, case_ids, cases, k=5)
        print(f"\nTop-5 kasus termirip:")
        print(f"{'No':<4} {'Case ID':<12} {'Similarity':<12} {'Terdakwa':<30} {'Lama Disersi'}")
        print("-" * 80)
        for i, r in enumerate(results, 1):
            print(
                f"{i:<4} {r['case_id']:<12} {r['similarity']:<12.4f} "
                f"{(r['terdakwa'] or '?'):<30} {r['lama_disersi'] or '?'}"
            )


                                                              
          
                                                              

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive retrieval demo after the pipeline finishes.",
    )
    args = parser.parse_args()

    result = main()
    if result and args.interactive:
        vectorizer, tfidf_matrix, case_ids, cases, svm = result
        run_interactive_demo(vectorizer, tfidf_matrix, case_ids, cases)
