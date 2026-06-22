"""
04_predict.py — Tahap 4: Case Solution Reuse
=============================================
Sistem CBR Tindak Pidana Disersi (Pasal 87 KUHPM)
Mata Kuliah Penalaran Komputer — Semester Genap 2025/2026

Alur kerja:
1. Load model TF-IDF dan case base dari Tahap 3
2. Fungsi retrieve() berbasis cosine similarity (sama dengan Tahap 3)
3. Ekstrak solusi (amar_putusan / pidana_pokok) dari top-k kasus
4. Algoritma prediksi: majority vote + weighted similarity
5. Demo 5 kasus baru
6. Simpan hasil ke data/results/predictions.csv
"""

import os
import json
import logging
import joblib
import pandas as pd
from datetime import datetime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from cbr_text import preprocess_text

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE_DIR    = r"C:\file\cbr-desersi"
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

CASES_JSON          = os.path.join(DATA_DIR, "processed", "cases.json")
TFIDF_VEC_PATH      = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH   = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
CASE_IDS_PATH       = os.path.join(MODEL_DIR, "case_ids.pkl")
PREDICTIONS_CSV     = os.path.join(RESULTS_DIR, "predictions.csv")
LOG_FILE            = os.path.join(LOG_DIR, "predict.log")

TOP_K = 5  # jumlah kasus mirip yang diambil

# ============================================================
# SETUP LOGGING
# ============================================================
os.makedirs(RESULTS_DIR, exist_ok=True)
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


# ============================================================
# 1. LOAD MODEL DAN DATA
# ============================================================

def load_models():
    """Load TF-IDF vectorizer, matrix, dan case_ids dari folder models/."""
    logger.info("Memuat model dari: %s", MODEL_DIR)
    vectorizer  = joblib.load(TFIDF_VEC_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    case_ids    = joblib.load(CASE_IDS_PATH)
    logger.info("  Vectorizer    : OK")
    logger.info("  TF-IDF matrix : shape %s", tfidf_matrix.shape)
    logger.info("  Case IDs      : %d kasus", len(case_ids))
    return vectorizer, tfidf_matrix, case_ids


def load_cases(cases_json: str) -> dict:
    """
    Load cases.json dan bangun dict {case_id: data_kasus}.
    Mengembalikan dict untuk akses O(1).
    """
    with open(cases_json, "r", encoding="utf-8") as f:
        cases_list = json.load(f)
    cases_dict = {c["case_id"]: c for c in cases_list}
    logger.info("Loaded %d kasus dari %s", len(cases_dict), cases_json)
    return cases_dict


# ============================================================
# 2. FUNGSI RETRIEVE (sama seperti Tahap 3)
# ============================================================

def retrieve(query: str, vectorizer, tfidf_matrix, case_ids: list, k: int = TOP_K):
    """
    Temukan top-k kasus paling mirip dengan query.

    Parameter
    ---------
    query        : teks query kasus baru
    vectorizer   : TfidfVectorizer yang sudah di-fit
    tfidf_matrix : matriks TF-IDF case base (sparse)
    case_ids     : list case_id sesuai urutan baris tfidf_matrix
    k            : jumlah kasus yang dikembalikan

    Return
    ------
    List of tuple: [(case_id, similarity_score), ...]
    """
    # 1) Pre-process & vektorisasi query
    query_vec = vectorizer.transform([preprocess_text(query)])

    # 2) Hitung cosine similarity dengan semua case vectors
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 3) Ambil top-k indeks (diurutkan descending)
    top_k_idx = sim_scores.argsort()[::-1][:k]

    results = [(case_ids[i], float(sim_scores[i])) for i in top_k_idx]
    return results


# ============================================================
# 3. EKSTRAKSI SOLUSI
# ============================================================

def extract_solution(case: dict) -> str:
    """
    Ambil teks solusi dari satu kasus.
    Prioritas: pidana_pokok → amar_putusan → fallback.
    """
    pidana_pokok  = (case.get("pidana_pokok") or "").strip()
    amar_putusan  = (case.get("amar_putusan") or "").strip()

    if pidana_pokok and pidana_pokok.lower() not in ("tidak ditemukan", "-", ""):
        return pidana_pokok
    if amar_putusan and amar_putusan.lower() not in ("tidak ditemukan", "-", ""):
        return amar_putusan[:300]  # batasi 300 karakter agar ringkas
    return "Solusi tidak tersedia"


def get_top_k_solutions(top_k_results: list, cases_dict: dict) -> list:
    """
    Dari hasil retrieve, bangun list solusi beserta skor similarity.

    Return
    ------
    List of dict: [{case_id, sim, solution, terdakwa, lama_disersi}, ...]
    """
    solutions = []
    for case_id, sim in top_k_results:
        case = cases_dict.get(case_id, {})
        solutions.append({
            "case_id":     case_id,
            "sim":         sim,
            "solution":    extract_solution(case),
            "terdakwa":    case.get("terdakwa", "?"),
            "lama_disersi": case.get("lama_disersi", "?"),
            "pidana_pokok": case.get("pidana_pokok", "?"),
        })
    return solutions


# ============================================================
# 4. ALGORITMA PREDIKSI
# ============================================================

def majority_vote(solutions: list) -> str:
    """
    Majority vote: pilih solusi yang paling banyak muncul di top-k.
    Jika semua unik, ambil yang similarity-nya tertinggi.
    """
    sol_texts = [s["solution"] for s in solutions if s["solution"] != "Solusi tidak tersedia"]
    if not sol_texts:
        return "Tidak ada solusi yang dapat diprediksi"

    counter   = Counter(sol_texts)
    most_common = counter.most_common(1)[0][0]
    return most_common


def weighted_similarity(solutions: list) -> str:
    """
    Weighted similarity: bobot = skor cosine similarity.
    Akumulasikan skor per solusi unik, pilih yang skornya tertinggi.
    """
    weight_map: dict[str, float] = {}
    for s in solutions:
        sol = s["solution"]
        if sol == "Solusi tidak tersedia":
            continue
        weight_map[sol] = weight_map.get(sol, 0.0) + s["sim"]

    if not weight_map:
        return "Tidak ada solusi yang dapat diprediksi"

    best_sol = max(weight_map, key=weight_map.get)
    return best_sol


def predict_outcome(query: str, vectorizer, tfidf_matrix, case_ids: list,
                    cases_dict: dict, k: int = TOP_K) -> dict:
    """
    Prediksi solusi untuk query kasus baru.

    Return
    ------
    dict berisi:
        top_k_results    : list (case_id, sim)
        solutions        : list detail solusi per kasus
        majority_vote    : prediksi majority vote
        weighted_sim     : prediksi weighted similarity
        final_prediction : prediksi akhir (weighted similarity diprioritaskan)
    """
    top_k_results = retrieve(query, vectorizer, tfidf_matrix, case_ids, k)
    solutions     = get_top_k_solutions(top_k_results, cases_dict)

    mv  = majority_vote(solutions)
    ws  = weighted_similarity(solutions)

    # Prediksi akhir: gunakan weighted similarity (lebih akurat secara teoretis)
    final = ws

    return {
        "top_k_results":    top_k_results,
        "solutions":        solutions,
        "majority_vote":    mv,
        "weighted_sim":     ws,
        "final_prediction": final,
    }


# ============================================================
# 5. DEMO — 5 KASUS BARU
# ============================================================

DEMO_QUERIES = [
    {
        "query_id": "new_001",
        "description": "Prajurit meninggalkan kesatuan tanpa ijin selama 30 hari",
        "query": (
            "Terdakwa meninggalkan Kesatuan tanpa ijin yang sah dari atasan yang berwenang "
            "terhitung sejak tanggal 10 Januari 2022 sampai dengan tanggal 10 Februari 2022 "
            "selama kurang lebih 30 hari. Terdakwa tidak membawa barang inventaris negara. "
            "Terdakwa menyesal dan berjanji tidak akan mengulangi perbuatannya."
        ),
    },
    {
        "query_id": "new_002",
        "description": "Prajurit disersi lebih dari 6 bulan dengan barang inventaris",
        "query": (
            "Bahwa terdakwa Prada meninggalkan Kesatuan tanpa ijin yang sah dari Dansat "
            "sejak bulan Maret 2021 hingga Oktober 2021 selama kurang lebih 7 bulan. "
            "Terdakwa membawa senjata laras panjang milik kesatuan dan tidak kembali "
            "meskipun sudah dipanggil secara resmi oleh Komandan Satuan."
        ),
    },
    {
        "query_id": "new_003",
        "description": "Prajurit disersi karena masalah keluarga selama 45 hari",
        "query": (
            "Terdakwa tidak masuk dinas tanpa ijin selama 45 hari terhitung mulai tanggal "
            "5 Maret 2022. Alasan terdakwa meninggalkan kesatuan adalah karena istri sakit "
            "dan ada masalah ekonomi keluarga. Terdakwa kemudian menyerahkan diri kepada "
            "Ankum setelah mendapat panggilan. Negara Kesatuan Republik Indonesia tidak "
            "dalam keadaan perang."
        ),
    },
    {
        "query_id": "new_004",
        "description": "Prajurit disersi singkat sekitar 20 hari",
        "query": (
            "Bahwa berdasarkan Berita Acara Pemeriksaan, terdakwa telah meninggalkan "
            "Kesatuan tanpa ijin yang sah dari Danyonif selaku Ankum terhitung sejak "
            "tanggal 1 April 2023 sampai dengan ditangkap pada tanggal 21 April 2023 "
            "selama kurang lebih 20 hari. Terdakwa tidak membawa barang inventaris negara "
            "dan mengakui perbuatannya."
        ),
    },
    {
        "query_id": "new_005",
        "description": "Prajurit disersi lebih dari 1 tahun tanpa kejelasan",
        "query": (
            "Terdakwa telah meninggalkan Kesatuan tanpa ijin yang sah sejak tanggal "
            "15 Januari 2020 dan tidak pernah kembali hingga dilakukan penangkapan "
            "pada bulan Maret 2022 sehingga masa disersi terdakwa lebih dari 2 tahun. "
            "Majelis Hakim mempertimbangkan bahwa terdakwa tidak pernah menyerahkan diri "
            "dan tidak ada itikad baik. Unsur-unsur Pasal 87 ayat (1) ke-2 KUHPM terpenuhi."
        ),
    },
]


def run_demo(vectorizer, tfidf_matrix, case_ids, cases_dict):
    """Jalankan demo predict_outcome untuk 5 kasus baru dan tampilkan hasilnya."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEMO PREDICT OUTCOME — 5 KASUS BARU")
    logger.info("=" * 60)

    rows = []
    for item in DEMO_QUERIES:
        qid   = item["query_id"]
        desc  = item["description"]
        query = item["query"]

        logger.info("")
        logger.info("Query [%s]: %s", qid, desc)
        result = predict_outcome(query, vectorizer, tfidf_matrix, case_ids, cases_dict)

        # Tampilkan top-k
        for rank, (case_id, sim) in enumerate(result["top_k_results"], 1):
            sol_detail = next(s for s in result["solutions"] if s["case_id"] == case_id)
            logger.info(
                "  #%d [%s] sim=%.4f | %s | disersi=%s | pidana=%s",
                rank, case_id, sim,
                sol_detail["terdakwa"],
                sol_detail["lama_disersi"],
                sol_detail["pidana_pokok"],
            )

        logger.info("  → Majority Vote     : %s", result["majority_vote"])
        logger.info("  → Weighted Sim      : %s", result["weighted_sim"])
        logger.info("  → PREDIKSI AKHIR    : %s", result["final_prediction"])

        # Siapkan baris untuk CSV
        top5_ids = ",".join([cid for cid, _ in result["top_k_results"]])
        rows.append({
            "query_id":           qid,
            "description":        desc,
            "predicted_solution": result["final_prediction"],
            "majority_vote":      result["majority_vote"],
            "weighted_sim":       result["weighted_sim"],
            "top_5_case_ids":     top5_ids,
            "top_5_similarities": ",".join([f"{s:.4f}" for _, s in result["top_k_results"]]),
        })

    return rows


# ============================================================
# 6. SIMPAN HASIL KE CSV
# ============================================================

def save_predictions(rows: list, output_path: str):
    """Simpan hasil prediksi ke predictions.csv."""
    df = pd.DataFrame(rows, columns=[
        "query_id", "description", "predicted_solution",
        "majority_vote", "weighted_sim",
        "top_5_case_ids", "top_5_similarities",
    ])
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("")
    logger.info("Hasil prediksi disimpan: %s (%d baris)", output_path, len(df))
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("TAHAP 4: CASE SOLUTION REUSE")
    logger.info("Mulai: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # 1. Load model dan data
    vectorizer, tfidf_matrix, case_ids = load_models()
    cases_dict = load_cases(CASES_JSON)

    # 2. Jalankan demo 5 kasus baru
    rows = run_demo(vectorizer, tfidf_matrix, case_ids, cases_dict)

    # 3. Simpan ke CSV
    save_predictions(rows, PREDICTIONS_CSV)

    # 4. Tampilkan ringkasan
    logger.info("")
    logger.info("=" * 60)
    logger.info("RINGKASAN PREDIKSI")
    for row in rows:
        logger.info("  [%s] → %s", row["query_id"], row["predicted_solution"])
    logger.info("=" * 60)
    logger.info("Tahap 4 selesai.")


if __name__ == "__main__":
    main()