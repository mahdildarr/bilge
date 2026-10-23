"""Cheap, deterministic text measurements used for data cleaning and for evaluating answers.

Multilingual base models answering in Turkish often drift into English or Chinese, repeat
themselves, or stop early. These statistics make those failure modes countable:

* ``turkish_score``: share of alphabetic characters that are Latin, plus Turkish-specific letters
  and function words as positive evidence; CJK or Cyrillic text pushes it to zero.
* ``is_turkish``: a thresholded decision on top of it.
* ``repetition``: share of repeated word 3-grams (0 = no repetition).
"""
from __future__ import annotations

import re
import unicodedata

TR_LETTERS = set("çğıöşüÇĞİÖŞÜ")
TR_WORDS = {"ve", "bir", "bu", "için", "ile", "da", "de", "olarak", "daha", "gibi", "çok", "olan", "veya", "ama",
            "ise", "her", "en", "kadar", "değil", "mi", "mı", "şu", "ya", "hem", "sonra", "önce", "nasıl", "neden"}
EN_WORDS = {"the", "and", "of", "to", "is", "in", "that", "it", "for", "with", "as", "are", "this", "be", "on",
            "you", "can", "or", "by", "an", "which", "not", "have", "from"}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def script_counts(text: str) -> dict:
    c = {"latin": 0, "cjk": 0, "cyrillic": 0, "other": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            c["latin"] += 1
        elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name or "HANGUL" in name:
            c["cjk"] += 1
        elif "CYRILLIC" in name:
            c["cyrillic"] += 1
        else:
            c["other"] += 1
    return c


def turkish_score(text: str) -> float:
    """0..1; high for Turkish, low for English or non-Latin text."""
    c = script_counts(text)
    letters = sum(c.values())
    if letters == 0:
        return 0.0
    latin_share = c["latin"] / letters
    w = words(text)
    if not w:
        return 0.0
    tr = sum(x in TR_WORDS for x in w) + sum(any(ch in TR_LETTERS for ch in x) for x in w)
    en = sum(x in EN_WORDS for x in w)
    lexical = tr / max(tr + en, 1) if (tr + en) else 0.5
    return latin_share * lexical


def is_turkish(text: str, threshold: float = 0.6) -> bool:
    return turkish_score(text) >= threshold


def repetition(text: str, n: int = 3) -> float:
    """Share of word n-grams that already occurred earlier in the text."""
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def has_cjk(text: str) -> bool:
    return script_counts(text)["cjk"] > 0
