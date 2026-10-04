"""Single source of truth for languages: our code -> per-provider codes + display info.

Coverage is the intersection of what each stage supports:
    speech recognition (Whisper): 100 languages
    translation (NLLB-200):        200 languages
    natural voice (Piper):          ~50 languages, downloaded on first use
A language without a voice still works: its translation is shown as subtitles.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # our code (BCP-47-ish): "ar", "ar-EG", "de"
    name: str
    native: str
    flag: str
    whisper: str  # Whisper language code
    nllb: str  # FLORES-200 code used by NLLB
    rtl: bool = False

    @property
    def base(self) -> str:
        return self.code.split("-")[0]


def _L(code, name, native, flag, whisper, nllb, rtl=False):
    return Language(code, name, native, flag, whisper, nllb, rtl)


# Every Whisper language that NLLB can also translate (+ regional variants we care about).
_LANGS = [
    _L("ar", "Arabic (Standard)", "العربية", "🇸🇦", "ar", "arb_Arab", True),
    _L("ar-EG", "Arabic (Egyptian)", "مصري", "🇪🇬", "ar", "arz_Arab", True),
    _L("ar-MA", "Arabic (Moroccan)", "الدارجة", "🇲🇦", "ar", "ary_Arab", True),
    _L("ar-LB", "Arabic (Levantine)", "شامي", "🇱🇧", "ar", "apc_Arab", True),
    _L("ar-SA", "Arabic (Najdi/Gulf)", "خليجي", "🇸🇦", "ar", "ars_Arab", True),
    _L("en", "English", "English", "🇬🇧", "en", "eng_Latn"),
    _L("de", "German", "Deutsch", "🇩🇪", "de", "deu_Latn"),
    _L("fr", "French", "Français", "🇫🇷", "fr", "fra_Latn"),
    _L("es", "Spanish", "Español", "🇪🇸", "es", "spa_Latn"),
    _L("it", "Italian", "Italiano", "🇮🇹", "it", "ita_Latn"),
    _L("pt", "Portuguese", "Português", "🇵🇹", "pt", "por_Latn"),
    _L("ru", "Russian", "Русский", "🇷🇺", "ru", "rus_Cyrl"),
    _L("zh", "Chinese (Simplified)", "中文", "🇨🇳", "zh", "zho_Hans"),
    _L("zh-TW", "Chinese (Traditional)", "繁體中文", "🇹🇼", "zh", "zho_Hant"),
    _L("yue", "Cantonese", "粵語", "🇭🇰", "yue", "yue_Hant"),
    _L("ja", "Japanese", "日本語", "🇯🇵", "ja", "jpn_Jpan"),
    _L("ko", "Korean", "한국어", "🇰🇷", "ko", "kor_Hang"),
    _L("tr", "Turkish", "Türkçe", "🇹🇷", "tr", "tur_Latn"),
    _L("hi", "Hindi", "हिन्दी", "🇮🇳", "hi", "hin_Deva"),
    _L("nl", "Dutch", "Nederlands", "🇳🇱", "nl", "nld_Latn"),
    _L("pl", "Polish", "Polski", "🇵🇱", "pl", "pol_Latn"),
    _L("el", "Greek", "Ελληνικά", "🇬🇷", "el", "ell_Grek"),
    _L("sv", "Swedish", "Svenska", "🇸🇪", "sv", "swe_Latn"),
    _L("no", "Norwegian (Bokmål)", "Norsk bokmål", "🇳🇴", "no", "nob_Latn"),
    _L("nn", "Norwegian (Nynorsk)", "Nynorsk", "🇳🇴", "nn", "nno_Latn"),
    _L("da", "Danish", "Dansk", "🇩🇰", "da", "dan_Latn"),
    _L("fi", "Finnish", "Suomi", "🇫🇮", "fi", "fin_Latn"),
    _L("cs", "Czech", "Čeština", "🇨🇿", "cs", "ces_Latn"),
    _L("ro", "Romanian", "Română", "🇷🇴", "ro", "ron_Latn"),
    _L("uk", "Ukrainian", "Українська", "🇺🇦", "uk", "ukr_Cyrl"),
    _L("fa", "Persian", "فارسی", "🇮🇷", "fa", "pes_Arab", True),
    _L("ur", "Urdu", "اردو", "🇵🇰", "ur", "urd_Arab", True),
    _L("he", "Hebrew", "עברית", "🇮🇱", "he", "heb_Hebr", True),
    _L("id", "Indonesian", "Bahasa Indonesia", "🇮🇩", "id", "ind_Latn"),
    _L("ms", "Malay", "Bahasa Melayu", "🇲🇾", "ms", "zsm_Latn"),
    _L("vi", "Vietnamese", "Tiếng Việt", "🇻🇳", "vi", "vie_Latn"),
    _L("th", "Thai", "ไทย", "🇹🇭", "th", "tha_Thai"),
    _L("bn", "Bengali", "বাংলা", "🇧🇩", "bn", "ben_Beng"),
    _L("hu", "Hungarian", "Magyar", "🇭🇺", "hu", "hun_Latn"),
    _L("bg", "Bulgarian", "Български", "🇧🇬", "bg", "bul_Cyrl"),
    _L("ca", "Catalan", "Català", "🇪🇸", "ca", "cat_Latn"),
    _L("ta", "Tamil", "தமிழ்", "🇮🇳", "ta", "tam_Taml"),
    _L("te", "Telugu", "తెలుగు", "🇮🇳", "te", "tel_Telu"),
    _L("ml", "Malayalam", "മലയാളം", "🇮🇳", "ml", "mal_Mlym"),
    _L("kn", "Kannada", "ಕನ್ನಡ", "🇮🇳", "kn", "kan_Knda"),
    _L("mr", "Marathi", "मराठी", "🇮🇳", "mr", "mar_Deva"),
    _L("gu", "Gujarati", "ગુજરાતી", "🇮🇳", "gu", "guj_Gujr"),
    _L("pa", "Punjabi", "ਪੰਜਾਬੀ", "🇮🇳", "pa", "pan_Guru"),
    _L("as", "Assamese", "অসমীয়া", "🇮🇳", "as", "asm_Beng"),
    _L("ne", "Nepali", "नेपाली", "🇳🇵", "ne", "npi_Deva"),
    _L("si", "Sinhala", "සිංහල", "🇱🇰", "si", "sin_Sinh"),
    _L("sa", "Sanskrit", "संस्कृतम्", "🇮🇳", "sa", "san_Deva"),
    _L("hr", "Croatian", "Hrvatski", "🇭🇷", "hr", "hrv_Latn"),
    _L("sr", "Serbian", "Српски", "🇷🇸", "sr", "srp_Cyrl"),
    _L("bs", "Bosnian", "Bosanski", "🇧🇦", "bs", "bos_Latn"),
    _L("sl", "Slovenian", "Slovenščina", "🇸🇮", "sl", "slv_Latn"),
    _L("sk", "Slovak", "Slovenčina", "🇸🇰", "sk", "slk_Latn"),
    _L("mk", "Macedonian", "Македонски", "🇲🇰", "mk", "mkd_Cyrl"),
    _L("sq", "Albanian", "Shqip", "🇦🇱", "sq", "als_Latn"),
    _L("lt", "Lithuanian", "Lietuvių", "🇱🇹", "lt", "lit_Latn"),
    _L("lv", "Latvian", "Latviešu", "🇱🇻", "lv", "lvs_Latn"),
    _L("et", "Estonian", "Eesti", "🇪🇪", "et", "est_Latn"),
    _L("be", "Belarusian", "Беларуская", "🇧🇾", "be", "bel_Cyrl"),
    _L("ka", "Georgian", "ქართული", "🇬🇪", "ka", "kat_Geor"),
    _L("hy", "Armenian", "Հայերեն", "🇦🇲", "hy", "hye_Armn"),
    _L("az", "Azerbaijani", "Azərbaycanca", "🇦🇿", "az", "azj_Latn"),
    _L("kk", "Kazakh", "Қазақша", "🇰🇿", "kk", "kaz_Cyrl"),
    _L("uz", "Uzbek", "Oʻzbekcha", "🇺🇿", "uz", "uzn_Latn"),
    _L("tk", "Turkmen", "Türkmençe", "🇹🇲", "tk", "tuk_Latn"),
    _L("tg", "Tajik", "Тоҷикӣ", "🇹🇯", "tg", "tgk_Cyrl"),
    _L("mn", "Mongolian", "Монгол", "🇲🇳", "mn", "khk_Cyrl"),
    _L("tt", "Tatar", "Татарча", "🇷🇺", "tt", "tat_Cyrl"),
    _L("ba", "Bashkir", "Башҡортса", "🇷🇺", "ba", "bak_Cyrl"),
    _L("ps", "Pashto", "پښتو", "🇦🇫", "ps", "pbt_Arab", True),
    _L("sd", "Sindhi", "سنڌي", "🇵🇰", "sd", "snd_Arab", True),
    _L("ku", "Kurdish (Kurmanji)", "Kurdî", "🌐", "", "kmr_Latn"),  # no Whisper ASR: voice/text only
    _L("yi", "Yiddish", "ייִדיש", "🌐", "yi", "ydd_Hebr", True),
    _L("am", "Amharic", "አማርኛ", "🇪🇹", "am", "amh_Ethi"),
    _L("sw", "Swahili", "Kiswahili", "🇰🇪", "sw", "swh_Latn"),
    _L("so", "Somali", "Soomaali", "🇸🇴", "so", "som_Latn"),
    _L("ha", "Hausa", "Hausa", "🇳🇬", "ha", "hau_Latn"),
    _L("yo", "Yoruba", "Yorùbá", "🇳🇬", "yo", "yor_Latn"),
    _L("sn", "Shona", "chiShona", "🇿🇼", "sn", "sna_Latn"),
    _L("ln", "Lingala", "Lingála", "🇨🇩", "ln", "lin_Latn"),
    _L("mg", "Malagasy", "Malagasy", "🇲🇬", "mg", "plt_Latn"),
    _L("af", "Afrikaans", "Afrikaans", "🇿🇦", "af", "afr_Latn"),
    _L("ht", "Haitian Creole", "Kreyòl ayisyen", "🇭🇹", "ht", "hat_Latn"),
    _L("tl", "Tagalog / Filipino", "Tagalog", "🇵🇭", "tl", "tgl_Latn"),
    _L("jv", "Javanese", "Basa Jawa", "🇮🇩", "jw", "jav_Latn"),
    _L("su", "Sundanese", "Basa Sunda", "🇮🇩", "su", "sun_Latn"),
    _L("km", "Khmer", "ខ្មែរ", "🇰🇭", "km", "khm_Khmr"),
    _L("lo", "Lao", "ລາວ", "🇱🇦", "lo", "lao_Laoo"),
    _L("my", "Burmese", "မြန်မာ", "🇲🇲", "my", "mya_Mymr"),
    _L("bo", "Tibetan", "བོད་སྐད་", "🌐", "bo", "bod_Tibt"),
    _L("mi", "Māori", "Te Reo Māori", "🇳🇿", "mi", "mri_Latn"),
    _L("cy", "Welsh", "Cymraeg", "🏴", "cy", "cym_Latn"),
    _L("eu", "Basque", "Euskara", "🇪🇸", "eu", "eus_Latn"),
    _L("gl", "Galician", "Galego", "🇪🇸", "gl", "glg_Latn"),
    _L("oc", "Occitan", "Occitan", "🇫🇷", "oc", "oci_Latn"),
    _L("is", "Icelandic", "Íslenska", "🇮🇸", "is", "isl_Latn"),
    _L("fo", "Faroese", "Føroyskt", "🇫🇴", "fo", "fao_Latn"),
    _L("lb", "Luxembourgish", "Lëtzebuergesch", "🇱🇺", "lb", "ltz_Latn"),
    _L("mt", "Maltese", "Malti", "🇲🇹", "mt", "mlt_Latn"),
]

AUTO = "auto"
LANGUAGES: dict[str, Language] = {lang.code: lang for lang in _LANGS}
# Whisper code -> our default code (first row wins, so detected "ar" -> Standard Arabic)
_BY_WHISPER: dict[str, Language] = {}
for _lang in _LANGS:
    if _lang.whisper:
        _BY_WHISPER.setdefault(_lang.whisper, _lang)


class UnsupportedLanguageError(ValueError):
    pass


def get(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        raise UnsupportedLanguageError(f"Unsupported language {code!r}") from None


def from_whisper(code: str | None) -> Language | None:
    return _BY_WHISPER.get(code) if code else None


def can_listen(code: str) -> bool:
    """Can we recognise speech in this language? (needed for the speaker's language)"""
    return bool(get(code).whisper)


def same_base(a: str, b: str) -> bool:
    """'ar' and 'ar-EG' are the same spoken language for detection purposes."""
    return a.split("-")[0] == b.split("-")[0]
