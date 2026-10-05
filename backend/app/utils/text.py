"""Text segmentation helpers.

Sentence-level MT models (NLLB, Opus-MT) translate one sentence at a time best; feeding a
whole paragraph makes them drop or repeat clauses. We split on sentence-final punctuation
(Latin, Arabic, CJK) and keep each sentence's punctuation attached.
"""

from __future__ import annotations

import re

# . ! ? … ؟ (Arabic) ۔ (Urdu) followed by whitespace; CJK 。！？ end a sentence even without
# a following space (Chinese/Japanese don't put spaces between sentences).
_SENTENCE_END = re.compile(r"(?<=[.!?…؟۔])\s+|(?<=[。！？])\s*")
# Scripts written without spaces between words: word counts mean nothing there.
_NO_SPACE_SCRIPTS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\u0e00-\u0e7f\u0e80-\u0eff\u1000-\u109f\u1780-\u17ff]")
# Unpunctuated runs from STT can be long; cap a "sentence" so the MT model stays accurate.
MAX_WORDS = 40


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    out: list[str] = []
    for part in parts:
        words = part.split()
        while len(words) > MAX_WORDS:
            out.append(" ".join(words[:MAX_WORDS]))
            words = words[MAX_WORDS:]
        if words:
            out.append(" ".join(words))
    return out


def word_count(text: str) -> int:
    """Words, or a rough equivalent for Chinese/Japanese/Thai/Lao/Burmese/Khmer (~2 chars a word)."""
    spaced = len(text.split())
    unspaced = len(_NO_SPACE_SCRIPTS.findall(text))
    return max(spaced, unspaced // 2)
