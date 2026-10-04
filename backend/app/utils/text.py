"""Text segmentation helpers.

Sentence-level MT models (NLLB, Opus-MT) translate one sentence at a time best; feeding a
whole paragraph makes them drop or repeat clauses. We split on sentence-final punctuation
(Latin, Arabic, CJK) and keep each sentence's punctuation attached.
"""

from __future__ import annotations

import re

# . ! ? … ؟ (Arabic) ۔ (Urdu) 。！？ (CJK) followed by whitespace or end of text.
_SENTENCE_END = re.compile(r"(?<=[.!?…؟۔。！？])\s+")
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
