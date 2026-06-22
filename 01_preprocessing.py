"""
Tahap 1: Membangun Case Base
CBR Sistem - Pidana Militer Disersi
Script: 01_preprocessing.py

Fungsi:
- Membaca semua PDF dari folder data/pdf/
- Mengekstrak teks
- Membersihkan teks (hapus header/footer/watermark)
- Menyimpan ke data/raw/case_XXX.txt
- Mencatat log ke logs/cleaning.log
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime

from pdfminer.high_level import extract_text

BASE_DIR = Path(__file__).parent         
PDF_DIR  = BASE_DIR / "data" / "pdf"   
RAW_DIR  = BASE_DIR / "data" / "raw"  
LOG_DIR  = BASE_DIR / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "cleaning.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

                                                              
                                                     
                                                              
NOISE_PATTERNS = [
                        
    r"hkam\s*", r"ahkamah Agung Repub\s*",
    r"ahkamah Agung Republik Indonesia\s*",
    r"mah Agung Republik Indonesia\s*",
    r"blik Indonesi\s*",
                      
    r"Direktori Putusan Mahkamah Agung Republik Indonesia\s*",
    r"putusan\.mahkamahagung\.go\.id\s*",
                       
    r"Disclaimer[\s\S]*?Halaman\s*\d+",
    r"Kepaniteraan Mahkamah Agung.*?(?=\n\n|\Z)",
    r"Email\s*:.*?Halaman\s*\d+",
                   
    r"Hal\s+\d+\s+dari\s+\d+\s+hal\s+Putusan\s+Nomor\s*:.*",
    r"Halaman\s+\d+\s*",
]

def clean_text(raw: str) -> str:
    """Membersihkan teks hasil ekstraksi PDF."""

    text = raw

                                                         
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)

                               
    text = re.sub(r"\r\n|\r", "\n", text)                               
    text = re.sub(r"[ \t]+", " ", text)                                        
    text = re.sub(r"\n{3,}", "\n\n", text)                                    

                                                              
    text = re.sub(r"[^\x20-\x7E\xA0-\xFF\n]", " ", text)

                         
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text.strip()


def validate_text(text: str, filename: str) -> bool:
    """
    Validasi keutuhan teks.
    Minimal harus mengandung kata kunci penting putusan disersi.
    """
    min_words = 100                                        
    word_count = len(text.split())

                                                        
    required_keywords = ["terdakwa", "desersi", "putusan", "mengadili", "militer", "pidana"]
    found = [kw for kw in required_keywords if kw.lower() in text.lower()]

    if word_count < min_words:
        logger.warning(f"[SKIP] {filename} — terlalu pendek ({word_count} kata) → kemungkinan PDF scan/gambar")
        return False

    if len(found) < 1:
        logger.warning(f"[SKIP] {filename} — tidak ada keyword ditemukan sama sekali")
        return False

    return True


def process_all_pdfs():
    """Proses semua PDF di folder PDF_DIR."""

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"Tidak ada file PDF di folder: {PDF_DIR}")
        logger.error("Pastikan semua PDF sudah ditaruh di folder data/pdf/")
        return

    logger.info(f"{'='*60}")
    logger.info(f"Mulai preprocessing — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total PDF ditemukan: {len(pdf_files)}")
    logger.info(f"{'='*60}")

    success_count = 0
    skip_count    = 0
    error_count   = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        case_id  = f"case_{idx:03d}"
        out_path = RAW_DIR / f"{case_id}.txt"

        logger.info(f"[{idx}/{len(pdf_files)}] Memproses: {pdf_path.name}")

        try:
                                   
            raw_text = extract_text(str(pdf_path))

            if not raw_text or len(raw_text.strip()) < 100:
                logger.warning(f"  → Teks kosong atau terlalu pendek, skip.")
                skip_count += 1
                continue

                            
            clean = clean_text(raw_text)

                      
            if not validate_text(clean, pdf_path.name):
                skip_count += 1
                continue

                    
            with open(out_path, "w", encoding="utf-8") as f:
                                                                  
                f.write(f"# SOURCE_FILE: {pdf_path.name}\n")
                f.write(f"# CASE_ID: {case_id}\n")
                f.write(f"# PROCESSED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# WORD_COUNT: {len(clean.split())}\n")
                f.write("# " + "="*50 + "\n\n")
                f.write(clean)

            word_count = len(clean.split())
            logger.info(f"  → OK | {word_count} kata | disimpan ke {out_path.name}")
            success_count += 1

        except Exception as e:
            logger.error(f"  → ERROR memproses {pdf_path.name}: {e}")
            error_count += 1

               
    logger.info(f"\n{'='*60}")
    logger.info(f"SELESAI PREPROCESSING")
    logger.info(f"  Berhasil : {success_count} file")
    logger.info(f"  Dilewati : {skip_count} file")
    logger.info(f"  Error    : {error_count} file")
    logger.info(f"  Total    : {len(pdf_files)} file")
    logger.info(f"  Output   : {RAW_DIR}")
    logger.info(f"  Log      : {log_file}")
    logger.info(f"{'='*60}")


                                                              
          
                                                              
if __name__ == "__main__":
    process_all_pdfs()