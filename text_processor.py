import os
import re
import joblib

# ============================================================
# 1. LOAD MODEL CRF
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crf_bahasa.pkl")

# Pastikan file models/crf_bahasa.pkl sudah ada (hasil joblib.dump di notebook)
crf = joblib.load(MODEL_PATH)


# ============================================================
# 2. TOKENIZER  (SAMAKAN DENGAN NOTEBOOK/DOCX)
# ============================================================

# Pola ini sama seperti yang biasa dipakai temanmu:
# re.split(r'(\s+|[.,;?!()\[\]])', text)
token_pattern = re.compile(r'(\s+|[.,;?!()\[\]])')


# ============================================================
# 3. FUNGSI FITUR CRF  (PERSIS DARI TEMANMU)
# ============================================================

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),      # Apakah kapital semua? (USA, IBM)
        'word.istitle()': word.istitle(),      # Apakah awal kapital? (Jakarta)
        'word.isdigit()': word.isdigit(),      # Apakah angka?

        # Ciri Morfologi (Sangat ampuh untuk Inggris)
        'suffix_2': word[-2:],  # ed, ly
        'suffix_3': word[-3:],  # ing, ion
        'prefix_2': word[:2],   # un, re
    }

    # Melihat konteks kata SEBELUMNYA
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
        })
    else:
        features['BOS'] = True  # Awal Kalimat

    # Melihat konteks kata SETELAHNYA
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
        })
    else:
        features['EOS'] = True  # Akhir Kalimat

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


# ============================================================
# 4. FUNGSI UTAMA UNTUK DIPAKAI DI FLASK
# ============================================================

def process_text_html(text: str) -> str:
    """
    Menerima teks mentah (string),
    mem-predict label bahasa dengan CRF,
    lalu mengembalikan HTML dengan istilah asing dimiringkan (<i>...</i>).
    """

    # 1) Tokenisasi dengan pola yang sama
    raw_tokens = token_pattern.split(text)

    # buang token kosong (misalnya string kosong atau hanya spasi)
    clean_tokens = [t for t in raw_tokens if t.strip()]

    # 2) Siapkan format kalimat untuk CRF: [(token, label_dummy)]
    sent = [(tok, "O") for tok in clean_tokens]

    # 3) Ekstraksi fitur & prediksi
    X = sent2features(sent)
    y_pred = crf.predict([X])[0]   # list label per token

    # 4) Tentukan label mana yang dianggap "istilah asing"
    #    GANTI SET INI SESUAI LABEL DI DATASETMU
    #    Contoh: kalau di kolom langs pakai 'EN' untuk Inggris:
    FOREIGN_LABELS = {"EN", "en"}

    # 5) Gabungkan kembali dengan mempertahankan spasi / tanda baca
    html_tokens = []
    idx = 0

    for tok in raw_tokens:
        if tok.strip():  # ini token "kata"
            label = y_pred[idx]

            if label in FOREIGN_LABELS:
                html_tokens.append(f"<i>{tok}</i>")
            else:
                html_tokens.append(tok)

            idx += 1
        else:
            # spasi / delimiter kita biarkan apa adanya
            html_tokens.append(tok)

    return "".join(html_tokens)


# ============================================================
# 5. TEST SINGKAT (opsional, bisa dihapus)
# ============================================================

if __name__ == "__main__":
    contoh = "hallo aku anak stis yang blajar information retrieval"
    print(process_text_html(contoh))
