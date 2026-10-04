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
    translation_model: str = "nllb-200-distilled-1.3B-ct2-int8"  # folder in backend/models
    translation_device: str = "auto"

    # Auto-detect fallback: below this confidence we keep the speaker's last confident language.
    language_min_confidence: float = 0.6

    tts_provider: str = "piper"
    # Preferred voice per language, "lang=voice_id,..." (empty -> best installed voice)
    tts_voices: str = "ar=ar_JO-kareem-medium,de=de_DE-thorsten-high"
    speech_speed: float = 1.0

    input_device: str | None = None  # name substring or index; None = system default
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

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Defaults <- saved settings file <- TRANSLATOR_* environment <- explicit overrides."""
        values = dict(load_saved())
        for f in fields(cls):
            env = os.environ.get(f"TRANSLATOR_{f.name.upper()}")
            if env is not None:
                values[f.name] = _coerce(f.default, env)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    def updated(self, **changes) -> "Settings":
        known = {f.name: f.default for f in fields(self)}
        clean = {k: _coerce(known[k], v) for k, v in changes.items() if k in known}
        return replace(self, **clean)

    def save(self) -> Path:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path


def _coerce(default, value):
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, (int, float)) and not isinstance(value, bool):
        return type(default)(value)
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
    known = {f.name: f.default for f in fields(Settings)}
    return {k: _coerce(known[k], v) for k, v in raw.items() if k in known}
