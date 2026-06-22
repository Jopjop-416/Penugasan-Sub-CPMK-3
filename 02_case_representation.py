import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from cbr_text import normalize_text

                                                              
                  
                                                              
BASE_DIR      = Path(__file__).parent
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOG_DIR       = BASE_DIR / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

                                                              
               
                                                              
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "representation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


                                                              
                           
                                                              

def extract_no_perkara(text: str, source_file: str = "") -> str:
    text = normalize_text(text)

    patterns = [
        r"Nomor\s*[:\-]\s*(\d+-K/PM[\w.\-/]+\d{4})",
        r"Nomor\s*[:\-]\s*(\d+-\w+/PM[\w.\-/]+)",
        r"(\d+-K/PM[\w.\-/]+\d{4})",
        r"(\d+\s*-\s*K\s*/\s*PM[\w\s.\-/]+\d{4})",
        r"Nomor\s*[:\-]\s*(\d+-K\.?\/?PM[\w.\-/]+\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    if source_file:
        m = re.search(r"putusan_(\d+)-k_pm\.([a-z]+)-(\d+)_([a-z]+)_([a-z]+)_([0-9]{4})", source_file, re.IGNORECASE)
        if m:
            nomor = m.group(1)
            bagian = m.group(2).upper()
            subbagian = m.group(3)
            wilayah = m.group(4).upper()
            tingkat = m.group(5).upper()
            tahun = m.group(6)
            return f"{nomor}-K/PM.{bagian}-{subbagian}/{wilayah}/{tingkat}/{tahun}"
    return ""

def extract_tanggal_putusan(text: str) -> str:
    text = normalize_text(text)
    bulan = r"(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember|Nopember)"
    patterns = [
        rf"(?:hari\s+\w+\s+)?tanggal\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"pada\s+tanggal\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"(\d{{1,2}}\s+{bulan}\s+\d{{4}})\s*,\s*(?:pada\s+)?(?:hari\s+\w+\s+)?",
        rf"diputuskan.*?tanggal\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"ditetapkan.*?tanggal\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"(?:Ditetapkan|Diputuskan|Diberikan)\s+di\s+[^\n,]+,\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"(?:Jakarta|Jayapura|Merauke|Timika|Sorong|Ambon|Manokwari|Makasar|Makassar),\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"Putus\s*:\s*(\d{{2}}-\d{{2}}-\d{{4}})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

                                                                                    
                                                         
    tail = text[-5000:]
    tail_patterns = [
        rf"(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        rf"(\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{4}})",
    ]
    for pat in tail_patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""

def extract_pengadilan(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"(Pengadilan Militer\s+[\w\-/]+(?:\s+\w+)?)\s+(?:yang bersidang|tersebut)",
        r"PENGADILAN MILITER\s+([\w\s\-/]+?)(?:\n|tersebut|yang)",
        r"(DILMIL\s*[\w\-/]+)",
        r"(PENGADILAN\s+MILITER\s+[\w\s\-/]+?)(?:,|\n|tersebut)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""

def extract_terdakwa(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"Nama\s+lengkap\s*:\s*([A-Z][A-Z\s]+?)(?:\n|Pangkat|NRP)",
        r"Nama\s+Lengkap\s*:\s*([A-Z][A-Z\s,\.]+?)(?:\n|Pangkat|NRP|Kesatuan)",
        r"Terdakwa\s*:\s*([A-Z][A-Z\s,\.]+?)(?:\n|NRP|Pangkat|33|28|40)",
        r"nama\s*:\s*([A-Z][A-Z\s]+?)(?:\n)",
        r"Atas\s+nama\s+terdakwa\s+([A-Z][A-Z\s,\.]+?)(?:\n|Pangkat|NRP)",
        r"(?:Terdakwa|tersebut)\s+(?:di\s+atas\s+)?yaitu\s+([A-Z][A-Z\s]+?),\s*(?:Sertu|Pratu|Praka|Serka|Serda|Kopka|Letda|Lettu)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name) > 3 and len(name) < 60:
                return name
    return ""

def extract_pangkat_nrp(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"Pangkat/NRP\s*:\s*([^\n#]+)",
        r"Pangkat\s*:\s*([^\n#]+?NRP[^\n#]+)",
        r"NRP\s*:\s*([^\n#]+)",
        r"((?:Sertu|Pratu|Praka|Serka|Serda|Kopka|Letda|Lettu|Mayor|Kolonel|Kapten)\s*/\s*\d{10,})",
        r"((?:Sertu|Pratu|Praka|Serka|Serda|Kopka|Letda|Lettu|Mayor|Kolonel|Kapten)[^\n]{0,40}NRP\s*[:\-]?\s*\d{8,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:80]
    return ""

def extract_kesatuan(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"Kesatuan\s*:\s*([^\n#]+)",
        r"Kesatuan\s*[-/]\s*([^\n#]+)",
        r"Kesatuan\s+([^\n,]+)",
        r"berkedudukan\s+di\s+([^\n,]+)",
        r"bertugas\s+pada\s+([^\n,]+)",
        r"berdinas\s+pada\s+([^\n,]+)",
        r"menjadi\s+anggota\s+([^\n,]+)",
        r"berdinas\s+(?:aktif\s+)?di\s+([\w/\s]+?)(?:\s+menjabat|\s+dengan|\n)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if len(val) < 80:
                return val

                                                                 
    for kw in ["kesatuan", "satuan", "dinas di", "bertugas di"]:
        idx = text.lower().find(kw)
        if idx != -1:
            snippet = text[idx:idx + 180]
            m = re.search(r"(?:kesatuan|satuan|dinas di|bertugas di)\s*[:\-]?\s*([^\n,.;]{3,120})", snippet, re.IGNORECASE)
            if m:
                return m.group(1).strip()
    return ""

def extract_pasal(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"(Pasal\s+87\s+ayat\s*\([^)]+\)[^.]{0,100}KUHPM)",
        r"(Pasal\s+87\s+ayat\s*\([^)]+\)\s*ke-\d+[^\n]{0,120}KUHPM)",
        r"(Pasal\s+87[^\n]{0,120}KUHPM)",
        r"diatur.*?ancam.*?(Pasal\s+\d+[^.]{0,150}KUHPM)",
        r"(Pasal\s+87[^\n]{0,100})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:220]
    return ""

def extract_lama_disersi(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"selama\s+(\d+)\s*\([^)]+\)\s*hari\s+secara\s+berturut",
        r"(\d+)\s*\([^)]+\)\s*hari\s+secara\s+berturut-turut",
        r"selama\s+(\d+)\s+hari\s+berturut",
        r"(\d+)\s+hari\s+secara\s+berturut",
        r"selama\s+(\d+)\s*\([\w\s]+\)\s*hari",
        r"kurang\s+lebih\s+(\d+)\s*hari",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return f"{m.group(1)} hari"

    tgl = extract_tanggal_disersi(text)
    if tgl["mulai"] and tgl["akhir"]:
        return f"(dari {tgl['mulai']} s/d {tgl['akhir']})"
    return ""

def extract_tanggal_disersi(text: str) -> dict:
    text = normalize_text(text)
    result = {"mulai": "", "akhir": ""}
    bulan = r"(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember|Nopember)"
    m = re.search(
        rf"sejak\s+tanggal\s+(\d{{1,2}}\s+{bulan}\s+\d{{4}})\s+sampai\s+dengan\s+(?:tanggal\s+)?(\d{{1,2}}\s+{bulan}\s+\d{{4}})",
        text, re.IGNORECASE
    )
    if m:
        result["mulai"] = m.group(1).strip()
        result["akhir"] = m.group(2).strip()
    return result

def extract_pidana_pokok(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"Pidana\s+Pokok\s*:\s*(Penjara\s+selama\s+[\w\s\(\)\-\.]+?)(?:\.|;|\n|Pidana\s+Tambahan|Membebankan|Menetapkan)",
        r"pidana\s+pokok\s*[:\-]\s*(Penjara\s+selama\s+[\w\s\(\)\-\.]+?)(?:\.|;|\n|Pidana\s+Tambahan|Membebankan|Menetapkan)",
        r"(?:menjatuhkan|dijatuhkan)\s+pidana\s+(?:penjara|kurungan)\s+selama\s+([\w\s\(\)\-\.]+?)(?:\.|;|\n|Pidana|Membebankan|Menetapkan)",
        r"(?:pidana\s+penjara|penjara)\s+selama\s+([\w\s\(\)\-\.]+?(?:tahun|bulan)[\w\s\(\)\-\.]*?)(?:\.|;|\n|Pidana|Membebankan|Menetapkan)",
        r"Penjara\s+selama\s+([\w\s\(\)\-\.]+?(?:tahun|bulan)[\w\s\(\)\-\.]*?)(?:\.|;|\n|Pidana|Membebankan|Menetapkan)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(0) if "Penjara" in m.group(0) else f"Penjara selama {m.group(1)}"
            return val.strip()[:120]
    return ""

def extract_pidana_tambahan(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"Pidana\s+Tambahan\s*:\s*([^\n.]{5,120})",
        r"pidana\s+tambahan\s*[:\-]\s*([^\n.]{5,120})",
        r"(?:dipecat|diberhentikan|diputus\s+untuk\s+dipecat|dijatuhi\s+pidana\s+tambahan)[^\n.]{0,120}",
        r"[Dd]ipecat\s+dari\s+dinas\s+(?:Militer|TNI)[^\n.]{0,60}",
        r"diberhentikan\s+dari\s+dinas\s+[Kk]emiliteran[^\n.]{0,60}",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()[:120]
    return ""

def extract_jenis_sidang(text: str) -> str:
    if re.search(r"in\s+absensia", text, re.IGNORECASE):
        return "In Absensia"
    elif re.search(r"terdakwa\s+hadir\s+di\s+persidangan", text, re.IGNORECASE):
        return "Terdakwa Hadir"
    return "In Absensia"                                               


def extract_ringkasan_fakta(text: str) -> str:
    text = normalize_text(text)
    patterns = [
        r"(?:Terdakwa\s+telah\s+pergi|Bahwa\s+Terdakwa\s+telah\s+pergi)([\s\S]{50,600})(?=Menimbang|Mengingat|MENGADILI)",
        r"(?:meninggalkan\s+Kesatuan\s+tanpa\s+ijin)([\s\S]{50,400})(?=Menimbang)",
        r"(?:Dengan\s+cara-cara\s+sebagai\s+berikut)([\s\S]{100,600})(?=Berpendapat|Menimbang)",
        r"(?:Bahwa\s+Terdakwa[\s\S]{50,400}?)(?=Menimbang|Mengingat|MENGADILI)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            summary = re.sub(r"\s+", " ", m.group(0)).strip()
            return summary[:500]

    m = re.search(r"(?:didakwa|dakwaan)([\s\S]{100,400})(?=Menimbang)", text, re.IGNORECASE)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:400]

                                                                                    
    keywords = [
        "meninggalkan kesatuan",
        "tanpa ijin",
        "tidak masuk dinas",
        "tidak hadir",
        "dakwaan",
        "bahwa terdakwa",
    ]
    lower = text.lower()
    for kw in keywords:
        idx = lower.find(kw)
        if idx != -1:
            start = max(0, idx - 180)
            end = min(len(text), idx + 500)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            if len(snippet) >= 80:
                return snippet[:400]
    return ""

def extract_amar_putusan(text: str) -> str:
    text = normalize_text(text)
    m = re.search(
        r"MENGADILI[:\s]*([\s\S]{50,1200})(?=Demikian\s+diputuskan|Hakim\s+(?:Ketua|Anggota)|Panitera|$)",
        text, re.IGNORECASE
    )
    if m:
        amar = re.sub(r"\s+", " ", m.group(1)).strip()
        return amar[:900]
    m = re.search(
        r"(Menyatakan\s+Terdakwa[\s\S]{50,700}?)(?=Demikian\s+diputuskan|Hakim\s+(?:Ketua|Anggota)|Panitera|$)",
        text, re.IGNORECASE
    )
    if m:
        amar = re.sub(r"\s+", " ", m.group(1)).strip()
        return amar[:900]

    for kw in ["MEMUTUSKAN", "MENETAPKAN", "MENGHUKUM", "Menyatakan Terdakwa"]:
        idx = text.upper().find(kw.upper())
        if idx != -1:
            snippet = re.sub(r"\s+", " ", text[idx:idx + 1200]).strip()
            if len(snippet) >= 80:
                return snippet[:900]
    return ""

def count_words(text: str) -> int:
    clean = re.sub(r"#.*?\n", "", text)
    return len(clean.split())


                                                              
                    
                                                              

def process_all_cases():
    txt_files = sorted(RAW_DIR.glob("*.txt"))

    if not txt_files:
        logger.error(f"Tidak ada file .txt di {RAW_DIR}")
        return None

    logger.info(f"{'='*60}")
    logger.info(f"Mulai Case Representation — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total file: {len(txt_files)}")
    logger.info(f"{'='*60}")

    records = []

    for idx, txt_path in enumerate(txt_files, start=1):
        logger.info(f"[{idx}/{len(txt_files)}] Memproses: {txt_path.name}")

        with open(txt_path, "r", encoding="utf-8") as f:
            text = normalize_text(f.read())

        source_match = re.search(r"# SOURCE_FILE: (.+)", text)
        source_file  = source_match.group(1).strip() if source_match else txt_path.name

        tgl_disersi = extract_tanggal_disersi(text)

        record = {
            "case_id"           : f"case_{idx:03d}",
            "source_file"       : source_file,
            "no_perkara"        : extract_no_perkara(text, source_file),
            "pengadilan"        : extract_pengadilan(text),
            "tanggal_putusan"   : extract_tanggal_putusan(text),
            "terdakwa"          : extract_terdakwa(text),
            "pangkat_nrp"       : extract_pangkat_nrp(text),
            "kesatuan"          : extract_kesatuan(text),
            "pasal"             : extract_pasal(text),
            "lama_disersi"      : extract_lama_disersi(text),
            "tgl_disersi_mulai" : tgl_disersi["mulai"],
            "tgl_disersi_akhir" : tgl_disersi["akhir"],
            "jenis_sidang"      : extract_jenis_sidang(text),
            "pidana_pokok"      : extract_pidana_pokok(text),
            "pidana_tambahan"   : extract_pidana_tambahan(text),
            "ringkasan_fakta"   : extract_ringkasan_fakta(text),
            "amar_putusan"      : extract_amar_putusan(text),
            "jumlah_kata"       : count_words(text),
            "text_full"         : re.sub(r"#.*?\n", "", text).strip(),
        }

        records.append(record)
        logger.info(
            f"  → terdakwa: {record['terdakwa'] or '?'} | "
            f"lama: {record['lama_disersi'] or '?'} | "
            f"pidana: {'Ada' if record['pidana_pokok'] else '?'}"
        )

                                  
    df = pd.DataFrame(records)
    csv_path = PROCESSED_DIR / "cases.csv"
    df.drop(columns=["text_full"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"\nCSV disimpan: {csv_path}")

                                    
    json_path = PROCESSED_DIR / "cases.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON disimpan: {json_path}")

                        
    logger.info(f"\n{'='*60}")
    logger.info("RINGKASAN KUALITAS EKSTRAKSI:")
    fields = ["no_perkara", "terdakwa", "pangkat_nrp", "kesatuan",
              "pasal", "lama_disersi", "pidana_pokok", "pidana_tambahan",
              "tgl_disersi_mulai", "amar_putusan"]
    for field in fields:
        filled = sum(1 for r in records if r[field])
        pct    = filled / len(records) * 100
        status = "OK" if pct >= 60 else "PERLU CEK"
        logger.info(f"  {field:<25}: {filled:>2}/{len(records)} ({pct:>5.1f}%) [{status}]")

    logger.info(f"\nTotal kasus: {len(records)}")
    logger.info(f"{'='*60}")

    return df


                                                              
          
                                                              
if __name__ == "__main__":
    df = process_all_cases()

    if df is not None:
        print("\n=== PREVIEW 5 KASUS ===")
        cols = ["case_id", "terdakwa", "lama_disersi", "pidana_pokok", "jenis_sidang"]
        print(df[cols].head(5).to_string(index=False))
