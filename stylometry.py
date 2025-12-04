import numpy as np
import re
import string
from collections import Counter

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

import spacy
from textblob import TextBlob
import textstat

# --- Inisialisasi global (dijalankan sekali) ---

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Daftar kata emosi sederhana (bisa kamu perbanyak lagi)
EMOTION_WORDS = {
    "happy", "joy", "sad", "angry", "fear", "afraid", "excited",
    "depressed", "love", "hate", "anxious", "nervous", "worried"
}

FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours"}

feature_name_31 =[
        'word_count',
        'unique_word_count',
        'char_count',
        'avg_word_length',
        'ttr',
        'hapax_rate',

        'sentence_count',
        'avg_sentence_len',
        'punctuation_count',
        'stopword_count',
        'complex_word_count',
        'noun_ratio',
        'verb_ratio',
        'adj_ratio',
        'adv_ratio',
        'question_count',
        'exclamation_count',
        'contraction_count',

        'polarity',
        'subjectivity',
        'vader_compound',
        'emotion_word_ratio',

        'flesch',
        'gunning_fog',

        'first_person_count',
        'second_person_count',
        'person_entities_count',
        'date_entities_count',

        'bigram_uniqueness',
        'trigram_uniqueness',
        'syntax_variety',
]


def stylometry_31(text: str):
    """
    Menghasilkan 31 fitur stylometry dari sebuah teks (bahasa Inggris).

    Urutan fitur yang dikembalikan:

    1  word_count
    2  unique_word_count
    3  char_count
    4  avg_word_length
    5  ttr
    6  hapax_rate

    7  sentence_count
    8  avg_sentence_len
    9  punctuation_count
    10 stopword_count
    11 complex_word_count (kata panjang >= 7 huruf)
    12 noun_ratio
    13 verb_ratio
    14 adj_ratio
    15 adv_ratio
    16 question_count
    17 exclamation_count
    18 contraction_count

    19 polarity (TextBlob)
    20 subjectivity (TextBlob)
    21 vader_compound
    22 emotion_word_ratio

    23 flesch_reading_ease
    24 gunning_fog

    25 first_person_count
    26 second_person_count
    27 person_entities_count
    28 date_entities_count

    29 bigram_uniqueness
    30 trigram_uniqueness
    31 syntax_variety (proporsi POS tag unik)
    """

    # Safety
    if not isinstance(text, str):
        return [0.0] * 31
    text = text.strip()
    if not text:
        return [0.0] * 31

    # --- Tokenisasi dasar (NLTK) ---
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words_alpha = [w for w in words if w.isalpha()]
    words_lower = [w.lower() for w in words_alpha]

    word_count = len(words_alpha)
    unique_word_count = len(set(words_lower))
    char_count = sum(len(w) for w in words_alpha)

    avg_word_length = char_count / word_count if word_count > 0 else 0.0
    ttr = unique_word_count / word_count if word_count > 0 else 0.0

    freqs = Counter(words_lower)
    hapax = sum(1 for _, f in freqs.items() if f == 1)
    hapax_rate = hapax / word_count if word_count > 0 else 0.0

    # --- Syntactic complexity ---
    sentence_count = len(sentences)
    if sentence_count > 0:
        sent_lengths = [
            len([w for w in word_tokenize(s) if w.isalpha()])
            for s in sentences
        ]
        avg_sentence_len = float(np.mean(sent_lengths))
    else:
        avg_sentence_len = 0.0

    punctuation_count = sum(1 for c in text if c in string.punctuation)
    stopword_count = sum(1 for w in words_lower if w in stop_words)
    complex_word_count = sum(1 for w in words_alpha if len(w) >= 7)

    # --- Analisis POS & NER dengan spaCy ---
    doc = nlp(text)

    noun_count = sum(1 for t in doc if t.pos_ == "NOUN")
    verb_count = sum(1 for t in doc if t.pos_ == "VERB")
    adj_count = sum(1 for t in doc if t.pos_ == "ADJ")
    adv_count = sum(1 for t in doc if t.pos_ == "ADV")

    token_count = len([t for t in doc if t.is_alpha])
    noun_ratio = noun_count / token_count if token_count > 0 else 0.0
    verb_ratio = verb_count / token_count if token_count > 0 else 0.0
    adj_ratio = adj_count / token_count if token_count > 0 else 0.0
    adv_ratio = adv_count / token_count if token_count > 0 else 0.0

    question_count = text.count("?")
    exclamation_count = text.count("!")
    contraction_count = len(re.findall(r"\b\w+'t\b|\b\w+'re\b|\b\w+'s\b|\b\w+'ve\b|\b\w+'ll\b|\b\w+'d\b", text.lower()))

    # --- Sentiment & subjectivity ---
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores["compound"]

    emotion_word_count = sum(1 for w in words_lower if w in EMOTION_WORDS)
    emotion_word_ratio = emotion_word_count / word_count if word_count > 0 else 0.0

    # --- Readability ---
    try:
        flesch = textstat.flesch_reading_ease(text)
    except Exception:
        flesch = 0.0

    try:
        gunning_fog = textstat.gunning_fog(text)
    except Exception:
        gunning_fog = 0.0

    # --- NER / personalisation ---
    first_person_count = sum(1 for w in words_lower if w in FIRST_PERSON)
    second_person_count = sum(1 for w in words_lower if w in SECOND_PERSON)

    person_entities_count = sum(1 for ent in doc.ents if ent.label_ == "PERSON")
    date_entities_count = sum(1 for ent in doc.ents if ent.label_ == "DATE")

    # --- Uniqueness & variety ---
    if len(words_lower) >= 2:
        bigrams = list(zip(words_lower[:-1], words_lower[1:]))
        bigram_uniqueness = len(set(bigrams)) / len(bigrams)
    else:
        bigram_uniqueness = 0.0

    if len(words_lower) >= 3:
        trigrams = list(zip(words_lower[:-2], words_lower[1:-1], words_lower[2:]))
        trigram_uniqueness = len(set(trigrams)) / len(trigrams)
    else:
        trigram_uniqueness = 0.0

    pos_tags = [t.pos_ for t in doc if t.is_alpha]
    if pos_tags:
        syntax_variety = len(set(pos_tags)) / len(pos_tags)
    else:
        syntax_variety = 0.0

    # --- Gabungkan semua ke dalam satu vector (31 dimensi) ---
    features = [
        float(word_count),
        float(unique_word_count),
        float(char_count),
        float(avg_word_length),
        float(ttr),
        float(hapax_rate),

        float(sentence_count),
        float(avg_sentence_len),
        float(punctuation_count),
        float(stopword_count),
        float(complex_word_count),
        float(noun_ratio),
        float(verb_ratio),
        float(adj_ratio),
        float(adv_ratio),
        float(question_count),
        float(exclamation_count),
        float(contraction_count),

        float(polarity),
        float(subjectivity),
        float(vader_compound),
        float(emotion_word_ratio),

        float(flesch),
        float(gunning_fog),

        float(first_person_count),
        float(second_person_count),
        float(person_entities_count),
        float(date_entities_count),

        float(bigram_uniqueness),
        float(trigram_uniqueness),
        float(syntax_variety),
    ]

    return features
