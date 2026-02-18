import re
import gruut
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.spanish.normalizer import normalize_text
from style_bert_vits2.nlp.symbols import PUNCTUATIONS

def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    phones = []
    tones = []
    word2ph = []

    # Normalize text first (numbers to words, etc.)
    text = normalize_text(text)

    # Get words and their token counts from BERT tokenizer
    token_groups = __text_to_token_groups(text)

    for group in token_groups:
        # group is a list of tokens forming a word, e.g. ["camion", "##eta"] -> word "camioneta"
        # or ["."] -> word "."

        # Reconstruct word from tokens
        word_text = ""
        for t in group:
            if t.startswith("##"):
                word_text += t[2:]
            else:
                word_text += t

        w_phones = []
        w_tones = []

        if not group:
            continue

        # Check if punctuation
        # Assuming punctuation is a single token group
        if word_text in PUNCTUATIONS:
            w_phones = [word_text]
            w_tones = [0]
        else:
            # Phonemize word_text using gruut
            try:
                sentences = list(gruut.sentences(word_text, lang="es"))
                if sentences:
                    for sent in sentences:
                        for w in sent.words:
                            if w.phonemes:
                                w_phones.extend(w.phonemes)
            except Exception:
                pass

            w_tones = [0] * len(w_phones)

        # If word has tokens but no phones (and not handled punctuation), assign blank phones?
        # Or better, assign UNK or similar?
        # English g2p assigns UNK for unhandled chars.
        # But here we reconstructed words.
        # If gruut returns nothing (e.g. specialized symbol), we might skip or fallback.
        # If we skip phones, we must ensure word2ph accounts for tokens.
        # If phones is empty, distribute_phone will distribute 0 across tokens.
        # This means word2ph entries for these tokens will be 0.
        # This is valid for BERT feature alignment (feature unused).

        phones.extend(w_phones)
        tones.extend(w_tones)

        n_tokens = len(group)
        n_phones = len(w_phones)

        # Distribute phones to tokens
        # If n_phones < n_tokens (e.g. silent letters or subwords > phones), some tokens get 0 phones.
        # If n_phones > n_tokens, some tokens get >1 phones.
        distribution = __distribute_phone(n_phones, n_tokens)
        word2ph.extend(distribution)

    # Pad with silence
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph

def __distribute_phone(n_phone: int, n_word: int) -> list[int]:
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def __text_to_token_groups(text: str) -> list[list[str]]:
    tokenizer = bert_models.load_tokenizer(Languages.ES)
    tokens = tokenizer.tokenize(text)
    groups = []

    # Group tokens into words
    # Logic: tokens starting with ## are subwords of previous token.
    # Punctuation should be separated if not already.
    # BERT tokenizer usually handles punctuation as separate tokens.

    current_group = []
    for t in tokens:
        if t.startswith("##"):
            if current_group:
                current_group.append(t)
            else:
                # Should not happen ideally, but if starts with ##, treat as new group
                current_group = [t]
        elif t in PUNCTUATIONS:
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([t])
        else:
            if current_group:
                groups.append(current_group)
            current_group = [t]

    if current_group:
        groups.append(current_group)

    return groups
