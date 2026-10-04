import os
import subprocess
import sys
import time

import numpy as np
import pytest

from app.services.audio import process_capture as pc

windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def P(pid, parent, exe):
    return pc.ProcessInfo(pid, parent, exe)


def test_find_app_root_picks_top_of_the_tree():
    procs = [P(1, 0, "explorer.exe"), P(10, 1, "Zoom.exe"), P(11, 10, "Zoom.exe"), P(12, 10, "zWebview2Agent.exe"),
             P(20, 1, "chrome.exe"), P(21, 20, "chrome.exe")]
    assert pc.find_app_root("zoom", procs).pid == 10
    assert pc.find_app_root("chrome", procs).pid == 20
    assert pc.find_app_root("discord", procs) is None


@windows
def test_list_processes_sees_ourselves():
    assert any(p.pid == os.getpid() for p in pc.list_processes())


@windows
def test_unknown_app_not_running():
    with pytest.raises(pc.MeetingAudioUnavailableError, match="not running"):
        pc.ProcessAudioCapture("definitely-not-running-app.exe")


def _record(cap, seconds=1.0):
    cap.start()
    first = cap.read_frame(timeout=5)  # the helper needs a moment to start; time from the first frame
    frames, end = ([first] if first is not None else []), time.time() + seconds
    while time.time() < end:
        f = cap.read_frame(timeout=0.5)
        if f is not None:
            frames.append(f)
    cap.stop()
    return np.concatenate(frames) if frames else np.zeros(0, np.float32)


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0


# These make a quiet beep on the speakers, so they're opt-in.
audio_test = pytest.mark.skipif(os.environ.get("RUN_AUDIO_TESTS") != "1", reason="set RUN_AUDIO_TESTS=1 (plays a sound)")


@windows
@audio_test
def test_app_mode_captures_the_target_process():
    tone = subprocess.Popen([sys.executable, "-c",
                             "import numpy as np,sounddevice as sd;r=48000;t=np.arange(r*3)/r;"
                             "sd.play((0.03*np.sin(2*np.pi*440*t)).astype('float32'),r);sd.wait()"])
    try:
        time.sleep(0.8)
        audio = _record(pc.ProcessAudioCapture(pid=tone.pid))
        assert len(audio) / 16_000 > 0.7 and _rms(audio) > 0.01
        assert abs(np.argmax(np.abs(np.fft.rfft(audio))) * 16_000 / len(audio) - 440) < 5
    finally:
        tone.kill()


@windows
@audio_test
def test_system_mode_never_hears_our_own_voice():
    """The echo-loop guarantee: what the engine itself plays is excluded from capture."""
    import sounddevice as sd

    r = 48_000
    t = np.arange(r * 2) / r
    sd.play((0.03 * np.sin(2 * np.pi * 440 * t)).astype("float32"), r)  # played by THIS process
    try:
        time.sleep(0.3)
        audio = _record(pc.ProcessAudioCapture())
        assert len(audio) / 16_000 > 0.7  # the stream runs...
        assert _rms(audio) < 0.002  # ...but our own sound is not in it
    finally:
        sd.stop()
