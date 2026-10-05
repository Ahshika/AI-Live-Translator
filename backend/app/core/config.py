"""Engine settings. Values come from environment variables (prefix TRANSLATOR_) or defaults."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass
class Settings:
    my_language: str = "ar"
    other_language: str = "de"  # "auto" = detect

    stt_provider: str = "faster-whisper"
    # large-v3-turbo: best speed/quality balance for Arabic on an 8 GB GPU.
    # Use "small" on machines without an NVIDIA GPU.
    stt_model: str = "large-v3-turbo"
    stt_device: str = "auto"  # auto | cuda | cpu
    stt_compute_type: str = "auto"  # auto -> float16 on cuda, int8 on cpu

    translation_provider: str = "nllb"
    # folder in backend/models. "nllb-200-3.3B-ct2-int8" = high quality (needs ~4 GB of GPU memory)
    translation_model: str = "nllb-200-distilled-1.3B-ct2-int8"
    translation_device: str = "auto"

    # Auto-detect fallback: below this confidence we keep the speaker's last confident language.
    language_min_confidence: float = 0.6

    tts_provider: str = "piper"
    # Preferred voice per language, "lang=voice_id,..." (empty -> best installed voice)
    tts_voices: str = "ar=ar_JO-kareem-medium,de=de_DE-thorsten-high"
    speech_speed: float = 1.0
    # Names / terms you use ("Ahmed, Siemens, Kubernetes"): recognised and spelled correctly.
    glossary: str = ""

    input_device: str | None = None  # name substring or index; None = system default
    # Off by default: Whisper is trained on noisy speech and copes well; denoising helps a
    # really noisy room but can cost accuracy on a normal one. Auto-gain can also lift room
    # noise until the speech detector fires on it, so it's for genuinely quiet mics only.
    noise_reduction: str = "off"  # off | light | strong — cleans your mic before recognition
    auto_gain: bool = False  # raise a quiet microphone automatically
    output_device: str | None = None  # where incoming translations play (headphones)

    # Universal mode
    meeting_app: str = "system"  # "system" = every app except the translator, or zoom/teams/discord/chrome...
    latency_mode: str = "balanced"  # fast | balanced | accurate
    headphones: bool = True  # False -> stop listening to the mic while a translation plays (echo guard)
    interruptions: str = "smart"  # smart | off
    live_subtitles: bool = True
    hear_my_translation: bool = False  # also play my outgoing translation quietly in my headphones
    save_history: bool = False  # privacy: off by default; text only, never audio
    ui_language: str = "ar"  # interface language: ar | en

    def preferred_voices(self) -> dict[str, str]:
        pairs = (item.split("=", 1) for item in self.tts_voices.split(",") if "=" in item)
        return {lang.strip(): voice.strip() for lang, voice in pairs}

    def with_voice(self, language: str, voice: str | None) -> "Settings":
        """A copy where `language` (base code) is spoken by `voice` (None = automatic choice)."""
        voices = self.preferred_voices()
        voices.pop(language.split("-")[0], None)
        if voice:
            voices[language.split("-")[0]] = voice
        return self.updated(tts_voices=",".join(f"{k}={v}" for k, v in voices.items()))

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Defaults <- saved settings file <- TRANSLATOR_* environment <- explicit overrides."""
        values = dict(load_saved())
        for f in fields(cls):
            env = os.environ.get(f"TRANSLATOR_{f.name.upper()}")
            if env is not None:
                try:
                    values[f.name] = validate(f.name, env)
                except InvalidSettingError:
                    pass
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    def updated(self, **changes) -> "Settings":
        """A copy with `changes` applied; unknown keys are ignored, bad values raise InvalidSettingError."""
        known = {f.name for f in fields(self)}
        return replace(self, **{k: validate(k, v) for k, v in changes.items() if k in known})

    def save(self) -> Path:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path


# Allowed values for the settings the UI can change; anything else is rejected with a clear
# message instead of failing later, deep inside a running session.
CHOICES: dict[str, tuple[str, ...]] = {
    "stt_device": ("auto", "cuda", "cpu"),
    "translation_device": ("auto", "cuda", "cpu"),
    "latency_mode": ("fast", "balanced", "accurate"),
    "interruptions": ("smart", "off"),
    "ui_language": ("ar", "en"),
    "noise_reduction": ("off", "light", "strong"),
    "translation_model": ("nllb-200-distilled-1.3B-ct2-int8", "nllb-200-3.3B-ct2-int8"),
}
RANGES: dict[str, tuple[float, float]] = {
    "speech_speed": (0.5, 2.0),
    "language_min_confidence": (0.0, 1.0),
}


class InvalidSettingError(ValueError):
    pass


def validate(name: str, value):
    """Coerce one setting to its field's type and check it; raises InvalidSettingError."""
    known = {f.name: f.default for f in fields(Settings)}
    if name not in known:
        raise InvalidSettingError(f"unknown setting {name!r}")
    try:
        value = _coerce(known[name], value)
    except (TypeError, ValueError):
        raise InvalidSettingError(f"invalid value for {name}: {value!r}") from None
    if name in CHOICES and value not in CHOICES[name]:
        raise InvalidSettingError(f"{name} must be one of {', '.join(CHOICES[name])}")
    if name in RANGES:
        lo, hi = RANGES[name]
        if not lo <= value <= hi:
            raise InvalidSettingError(f"{name} must be between {lo} and {hi}")
    return value


def _coerce(default, value):
    if value is None:
        return None if default is None else default
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, (int, float)) and not isinstance(value, bool):
        return type(default)(value)
    if isinstance(default, str) and not isinstance(value, str):
        raise TypeError(f"expected text, got {type(value).__name__}")
    return value


def data_dir() -> Path:
    """Per-user data folder: %APPDATA%/AI Live Translator (override with TRANSLATOR_DATA_DIR)."""
    override = os.environ.get("TRANSLATOR_DATA_DIR")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "AI Live Translator"


def is_frozen() -> bool:
    """True inside the packaged (PyInstaller) app."""
    import sys

    return bool(getattr(sys, "frozen", False))


def models_dir() -> Path:
    """Where AI models live. Dev: backend/models. Installed app: %LOCALAPPDATA% (big, not roamed)."""
    override = os.environ.get("TRANSLATOR_MODELS_DIR")
    if override:
        return Path(override)
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
        return Path(base) / "AI Live Translator" / "models"
    return Path(__file__).resolve().parents[2] / "models"


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_saved() -> dict:
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    clean = {}
    for k, v in raw.items():
        try:  # one bad value (hand-edited file, older version) must not stop the app from opening
            clean[k] = validate(k, v)
        except InvalidSettingError:
            continue
    return clean
