# stylometry.py
import numpy as np
from collections import Counter
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

# # Jalankan sekali saja saat pertama kali di-run
# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def stylometry_10(text: str):
    if not isinstance(text, str):
        return [0.0] * 10
    text = text.strip()
    if not text:
        return [0.0] * 10

    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words_alpha = [w.lower() for w in words if w.isalpha()]

    word_count = len(words_alpha)
    unique_word_count = len(set(words_alpha))
    ttr = unique_word_count / word_count if word_count > 0 else 0.0

    freqs = Counter(words_alpha)
    hapax = sum(1 for _, f in freqs.items() if f == 1)
    hapax_rate = hapax / word_count if word_count > 0 else 0.0

    stopword_count = sum(1 for w in words_alpha if w in stop_words)
    sentence_count = len(sentences)

    if sentence_count > 0:
        sent_lengths = [
            len([w for w in word_tokenize(s) if w.isalpha()])
            for s in sentences
        ]
        avg_sentence_len = float(np.mean(sent_lengths))
        var_sentence_len = float(np.var(sent_lengths))
    else:
        avg_sentence_len = 0.0
        var_sentence_len = 0.0

    if len(words_alpha) >= 2:
        bigrams = list(zip(words_alpha[:-1], words_alpha[1:]))
        bigram_unique = len(set(bigrams)) / len(bigrams)
    else:
        bigram_unique = 0.0

    if len(words_alpha) >= 3:
        trigrams = list(zip(words_alpha[:-2], words_alpha[1:-1], words_alpha[2:]))
        trigram_unique = len(set(trigrams)) / len(trigrams)
    else:
        trigram_unique = 0.0

    return [
        float(word_count),
        float(unique_word_count),
        float(ttr),
        float(hapax_rate),
        float(stopword_count),
        float(sentence_count),
        avg_sentence_len,
        var_sentence_len,
        float(bigram_unique),
        float(trigram_unique),
    ]
