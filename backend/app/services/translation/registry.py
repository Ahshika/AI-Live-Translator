"""Name -> translation provider factory, so the provider is chosen from config, not code."""

from __future__ import annotations

from typing import Callable

from app.core.config import Settings
from app.services.translation.base import TranslationProvider

ProviderFactory = Callable[[Settings], TranslationProvider]
_FACTORIES: dict[str, ProviderFactory] = {}


def register(name: str, factory: ProviderFactory) -> None:
    _FACTORIES[name] = factory


def available() -> list[str]:
    _register_builtins()
    return sorted(_FACTORIES)


def create(settings: Settings) -> TranslationProvider:
    _register_builtins()
    try:
        factory = _FACTORIES[settings.translation_provider]
    except KeyError:
        raise ValueError(
            f"Unknown translation provider {settings.translation_provider!r}; available: {sorted(_FACTORIES)}"
        ) from None
    return factory(settings)


def _register_builtins() -> None:
    if "nllb" not in _FACTORIES:
        from app.providers.translation.nllb_provider import NLLBProvider

        register("nllb", lambda s: NLLBProvider(model=s.translation_model, device=s.translation_device))
