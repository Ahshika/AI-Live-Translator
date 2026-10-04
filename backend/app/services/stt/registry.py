"""Name -> STT provider factory, so the provider is chosen from config, not code."""

from __future__ import annotations

from typing import Callable

from app.core.config import Settings
from app.services.stt.base import SpeechToTextProvider

ProviderFactory = Callable[[Settings], SpeechToTextProvider]
_FACTORIES: dict[str, ProviderFactory] = {}


def register(name: str, factory: ProviderFactory) -> None:
    _FACTORIES[name] = factory


def available() -> list[str]:
    _register_builtins()
    return sorted(_FACTORIES)


def create(settings: Settings) -> SpeechToTextProvider:
    _register_builtins()
    try:
        factory = _FACTORIES[settings.stt_provider]
    except KeyError:
        raise ValueError(
            f"Unknown STT provider {settings.stt_provider!r}; available: {sorted(_FACTORIES)}"
        ) from None
    return factory(settings)


def _register_builtins() -> None:
    if "faster-whisper" not in _FACTORIES:
        # Imported lazily so tests / other providers don't pay for loading CTranslate2.
        from app.providers.stt.faster_whisper_provider import FasterWhisperProvider

        register(
            "faster-whisper",
            lambda s: FasterWhisperProvider(
                model=s.stt_model, device=s.stt_device, compute_type=s.stt_compute_type
            ),
        )
