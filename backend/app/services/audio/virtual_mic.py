"""The virtual microphone: where the translated voice goes so meeting apps can 'hear' it.

How it works (VB-CABLE, free, signed driver by VB-Audio):
    the cable adds two devices that are wired together inside Windows:
        "CABLE Input"   — a speaker   <- we PLAY the translated voice here
        "CABLE Output"  — a microphone -> you SELECT this as the mic in Zoom/Teams/Discord/Meet
    so the other side hears only the translation, never your original voice.

We don't ship our own driver: a kernel audio driver must be signed by Microsoft (EV code
signing certificate + attestation) — not possible on a zero budget. Any other virtual cable
(VoiceMeeter, Virtual Audio Cable) works too; we look for them by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.audio.devices import AudioDevice, list_devices
from app.services.audio.playback import AudioPlayer
from app.services.tts.base import AudioChunk

# (playback device we write to, the microphone the meeting app should select)
KNOWN_CABLES = [
    ("CABLE Input", "CABLE Output"),  # VB-CABLE
    ("CABLE-A Input", "CABLE-A Output"),  # VB-CABLE A+B
    ("VoiceMeeter Input", "VoiceMeeter Output"),
    ("VoiceMeeter Aux Input", "VoiceMeeter Aux Output"),
    ("Line 1 (Virtual Audio Cable)", "Line 1 (Virtual Audio Cable)"),
]
VB_CABLE_URL = "https://vb-audio.com/Cable/"


class VirtualMicUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class VirtualCable:
    playback: AudioDevice  # we play into this
    mic_name: str  # what the user picks in the meeting app


def find_virtual_cable(devices: list[AudioDevice] | None = None) -> VirtualCable | None:
    devs = devices if devices is not None else list_devices()
    outputs = [d for d in devs if d.max_output_channels > 0]
    inputs = [d for d in devs if d.max_input_channels > 0]
    for play_name, mic_name in KNOWN_CABLES:
        for d in outputs:
            if d.name.lower().startswith(play_name.lower()):
                return VirtualCable(d, _actual_mic_name(inputs, mic_name))
    # VB-CABLE 2.x names its playback side "Speakers (2- VB-Audio Virtual Cable)" (plus a
    # "CABLE In 16 Ch" variant) — match on the driver name and prefer the stereo endpoint.
    vb = [d for d in outputs if "vb-audio virtual cable" in d.name.lower()]
    if vb:
        vb.sort(key=lambda d: ("16 ch" in d.name.lower(), d.name))
        return VirtualCable(vb[0], _actual_mic_name(inputs, "CABLE Output"))
    return None


def _actual_mic_name(inputs: list[AudioDevice], expected: str) -> str:
    """The exact name Windows shows for the cable's microphone side (e.g. 'CABLE Output (2- VB-...)')."""
    for d in inputs:
        if d.name.lower().startswith(expected.lower()):
            return d.name
    return expected


VIRTUAL_MARKERS = ("vb-audio", "cable output", "cable input", "voicemeeter", "virtual audio cable")


def is_virtual_device(name: str) -> bool:
    """A virtual cable must never be used as the user's real microphone: we'd translate our
    own translation forever."""
    return any(m in name.lower() for m in VIRTUAL_MARKERS)


class VirtualMicrophone:
    """Sink that speaks into the virtual cable; optionally lets you monitor it quietly."""

    def __init__(self, cable: VirtualCable | None = None, *, monitor: AudioPlayer | None = None,
                 monitor_gain: float = 0.25):
        cable = cable or find_virtual_cable()
        if cable is None:
            raise VirtualMicUnavailableError(
                f"No virtual microphone found. Install VB-CABLE (free) from {VB_CABLE_URL}, then restart.")
        self.cable = cable
        self.player = AudioPlayer(cable.playback)
        self.monitor, self.monitor_gain = monitor, monitor_gain

    @property
    def mic_name(self) -> str:
        return self.cable.mic_name

    def start(self) -> None:
        self.player.start()

    def close(self) -> None:
        self.player.close()

    def __call__(self, chunk: AudioChunk) -> None:
        self.player.enqueue(chunk)
        if self.monitor is not None:
            self.monitor.enqueue(AudioChunk(chunk.samples * self.monitor_gain, chunk.sample_rate))

    def stop(self) -> None:
        """Interrupt what's being said (the other person started talking)."""
        self.player.stop()

    @property
    def speaking(self) -> bool:
        return self.player.queued_seconds > 0
