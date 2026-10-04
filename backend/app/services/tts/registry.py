"""Name -> TTS provider factory, so the provider is chosen from config, not code."""

from __future__ import annotations

from typing import Callable

from app.core.config import Settings
from app.services.tts.base import TextToSpeechProvider

ProviderFactory = Callable[[Settings], TextToSpeechProvider]
_FACTORIES: dict[str, ProviderFactory] = {}


def register(name: str, factory: ProviderFactory) -> None:
    _FACTORIES[name] = factory


def available() -> list[str]:
    _register_builtins()
    return sorted(_FACTORIES)


def create(settings: Settings) -> TextToSpeechProvider:
    _register_builtins()
    try:
        factory = _FACTORIES[settings.tts_provider]
    except KeyError:
        raise ValueError(
            f"Unknown TTS provider {settings.tts_provider!r}; available: {sorted(_FACTORIES)}"
        ) from None
    return factory(settings)


def _register_builtins() -> None:
    if "piper" not in _FACTORIES:
        from app.providers.tts.piper_provider import PiperProvider

        register("piper", lambda s: PiperProvider(preferred=s.preferred_voices()))
