from __future__ import annotations

import re

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "ini",
    "itu", "atau", "dalam", "tidak", "adalah", "telah", "bahwa", "oleh",
    "sebagai", "para", "juga", "serta", "akan", "dapat", "kepada", "saat",
    "sudah", "belum", "jika", "maka", "atas", "bawah", "setelah", "sebelum",
    "namun", "tetapi", "karena", "yaitu", "antara", "tersebut", "hal",
    "demikian", "berdasarkan", "mengenai", "menurut", "sesuai",
    "merupakan", "dilakukan", "dilaksanakan", "diajukan", "ditetapkan",
    "menyatakan", "menyampaikan", "menimbang", "mengingat", "memperhatikan",
}

_NORMALIZE_MAP = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u200b": " ",
        "\u200c": " ",
        "\u200d": " ",
        "\ufeff": " ",
    }
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.translate(_NORMALIZE_MAP)
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
