"""Make spoken transcripts translatable.

Speech is not text: it has fillers ("ähm", "uh"), stutters ("I I I think"), false starts and
very short fragments. NLLB was trained on written sentences, so feeding it raw speech gives
literal, broken translations ("Uh. So. I think that..."). Cleaning costs microseconds and is
the cheapest translation-quality win there is.

Rules are deliberately conservative: only drop what never carries meaning.
"""

from __future__ import annotations

import re

from app.utils.text import split_sentences, word_count

# Pure hesitation sounds, per language base. Words that *can* carry meaning are not here
# (Arabic "يعني" = "I mean", English "like", German "also" = "so").
_FILLERS: dict[str, set[str]] = {
    "en": {"uh", "uhh", "um", "umm", "erm", "hmm", "mm", "mhm"},
    "de": {"äh", "ähm", "öh", "öhm", "hm", "hmm", "mhm"},
    "fr": {"euh", "heu", "bah", "hum"},
    "es": {"eh", "em", "este"},  # "este" as hesitation only when alone; see _is_filler
    "it": {"ehm", "eh", "uhm"},
    "pt": {"hã", "ãh", "uhm"},
    "nl": {"eh", "ehm", "uh", "uhm"},
    "tr": {"ıı", "ııı", "eee", "şey"},
    "ru": {"э", "ээ", "эээ", "мм", "ну-у"},
    # Not "آه"/"اه": in Egyptian Arabic that's "yes".
    "ar": {"امم", "اممم", "إمم", "ممم", "ااا", "اااا"},
}
_STANDALONE_ONLY = {"este", "şey"}  # fillers only when they're the whole fragment
_PUNCT = " .,!?;:،؛؟…-—"
_REPEAT = re.compile(r"\b(\w+)(?:[\s,]+\1\b)+", re.IGNORECASE | re.UNICODE)


def _base(lang: str) -> str:
    return lang.split("-")[0]


def remove_fillers(text: str, language: str) -> str:
    # Only the speaker's language: "er" is a filler in English but "he" in German.
    fillers = _FILLERS.get(_base(language), set())
    words = [w.strip(_PUNCT).lower() for w in text.split()]
    if words and all(w in fillers or not w for w in words):
        return ""  # nothing but hesitation ("Ähm." / "este...")
    for f in sorted(fillers - _STANDALONE_ONLY, key=len, reverse=True):
        # The filler and the commas around it: "we should, um, go" -> "we should go".
        text = re.sub(rf"[,،]?\s*(?<!\w){re.escape(f)}(?!\w)\s*[,،]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([.!?؟])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,،")


def collapse_repeats(text: str) -> str:
    """Stutters: 'I I I think' -> 'I think', 'ich, ich will' -> 'ich will'. Emphasis stays:
    a longer word said twice ('sehr sehr', 'very very') is meaning, not a stutter."""

    def fix(m: re.Match) -> str:
        word = m.group(1)
        count = len(re.findall(rf"(?<!\w){re.escape(word)}(?!\w)", m.group(0), re.IGNORECASE))
        return word if count >= 3 or len(word) <= 3 else m.group(0)

    return _REPEAT.sub(fix, text)


def clean_transcript(text: str, language: str) -> str:
    out = collapse_repeats(remove_fillers(text, language)).strip()
    if out and text.lstrip()[:1].isupper() and out[0].islower():
        out = out[0].upper() + out[1:]  # "Ähm, ich will" -> "Ich will": a sentence start again
    return out


def translation_units(text: str, *, min_words: int = 4, max_words: int = 40) -> list[str]:
    """Sentences to translate one by one, with tiny fragments merged into a neighbour.

    "Ja. Genau. Das machen wir morgen." translated piece by piece loses the context that
    makes "Genau" mean "exactly" rather than "precisely/accurately"; translated together it
    reads naturally. Long sentences still go alone so the first one can be spoken early.
    """
    units: list[str] = []
    for sentence in split_sentences(text):
        if units and (word_count(units[-1]) < min_words or word_count(sentence) < min_words) \
                and word_count(units[-1]) + word_count(sentence) <= max_words:
            units[-1] = f"{units[-1]} {sentence}"
        else:
            units.append(sentence)
    return units
