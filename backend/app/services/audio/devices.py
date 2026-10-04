"""Audio device discovery (WASAPI-first on Windows)."""

from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool
    is_default_output: bool


def _host_api_name(index: int) -> str:
    return sd.query_hostapis(index)["name"]


def list_devices(host_api: str | None = "Windows WASAPI") -> list[AudioDevice]:
    """List devices, by default only those exposed through WASAPI.

    Windows exposes every physical device several times (MME, DirectSound, WASAPI, WDM-KS).
    WASAPI is the low-latency modern API and the one that supports loopback capture,
    so we standardise on it and hide the duplicates.
    """
    default_in, default_out = sd.default.device
    devices: list[AudioDevice] = []
    for idx, d in enumerate(sd.query_devices()):
        api = _host_api_name(d["hostapi"])
        if host_api and api != host_api:
            continue
        devices.append(
            AudioDevice(
                index=idx,
                name=d["name"],
                host_api=api,
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_sample_rate=d["default_samplerate"],
                is_default_input=idx == default_in,
                is_default_output=idx == default_out,
            )
        )
    if not devices and host_api:
        return list_devices(host_api=None)
    return devices


def input_devices(host_api: str | None = "Windows WASAPI") -> list[AudioDevice]:
    return [d for d in list_devices(host_api) if d.max_input_channels > 0]


def default_input_device(host_api: str | None = "Windows WASAPI") -> AudioDevice | None:
    """The system default mic, mapped to its entry under the requested host API."""
    devices = input_devices(host_api)
    if not devices:
        return None
    from app.services.audio.virtual_mic import is_virtual_device

    real = [d for d in devices if not is_virtual_device(d.name)]
    if not real:
        return None
    default_name = sd.query_devices(kind="input")["name"]
    for d in real:  # installers like VB-CABLE sometimes make the cable the default mic: skip it
        if d.name == default_name or default_name.startswith(d.name) or d.name.startswith(default_name):
            return d
    return real[0]


def find_device(query: str | int, *, kind: str = "input") -> AudioDevice:
    """Find a device by index or (case-insensitive) name substring."""
    pool = input_devices() if kind == "input" else [d for d in list_devices() if d.max_output_channels > 0]
    if isinstance(query, int) or str(query).isdigit():
        for d in pool:
            if d.index == int(query):
                return d
        raise LookupError(f"No {kind} device with index {query}")
    matches = [d for d in pool if str(query).lower() in d.name.lower()]
    if not matches:
        raise LookupError(f"No {kind} device matching {query!r}")
    return matches[0]
