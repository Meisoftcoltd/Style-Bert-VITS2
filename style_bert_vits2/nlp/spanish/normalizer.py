import re
from num2words import num2words

def normalize_text(text: str) -> str:
    text = replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    text = expand_numbers(text)
    return text

def replace_punctuation(text: str) -> str:
    # Basic punctuation replacement similar to English/Japanese
    # Add Spanish specific ones if any (e.g. inverted question/exclamation marks)
    # Style-Bert-VITS2 seems to use standard punctuation for symbols.

    # Inverted marks are usually kept in Spanish text but G2P might ignore them or use them for intonation.
    # We can replace them with space or keep them if our symbol set supports them.
    # The current symbols.py has PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]
    # So we should probably map inverted ones to standard or remove them.
    # Let's map them to space for now, or maybe they are handled by gruut.
    # gruut might handle inverted marks.

    REPLACE_MAP = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "…": "...",
        "···": "...",
        "・・・": "...",
        "¡": "!",  # Map inverted exclamation to normal? Or just remove. Let's keep for now if it helps tone.
                   # But symbols.py doesn't have ¡. So better replace or remove.
                   # Actually, let's just replace with space or remove,
                   # as the model likely only learned standard end-sentence punctuation.
        "¿": "?",  # Map inverted question to normal?
    }
    # However, ¡ and ¿ are start of sentence markers. Mapping them to !/? (end markers) might be confusing if placed at start.
    # Usually we can just remove them or treat them as pause.

    text = text.replace("¡", "").replace("¿", "")

    pattern = re.compile("|".join(re.escape(p) for p in REPLACE_MAP))
    text = pattern.sub(lambda x: REPLACE_MAP[x.group()], text)
    return text

def expand_numbers(text: str) -> str:
    # Use regex to find numbers and replace with num2words
    # This is a simple implementation.
    return re.sub(r'\d+', lambda x: num2words(int(x.group()), lang='es'), text)
