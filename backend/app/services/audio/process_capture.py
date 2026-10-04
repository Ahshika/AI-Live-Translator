"""Capture what other apps play (the meeting's voices) — per process, via our native helper.

native/ProcessLoopback (C#, Windows' official process-loopback API, Win10 2004+/Win11):
  * app mode:    capture ONLY a meeting app's process tree (Zoom.exe, Teams, Discord, browser)
  * system mode: capture EVERYTHING EXCEPT this engine's own process tree

System mode is how we avoid the echo loop: our translated voice plays from our process,
so it is never captured and re-translated — no matter which device it plays on.
"""

from __future__ import annotations

import ctypes
import os
import queue
import subprocess
import sys
import threading
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.audio.format import FRAME_SAMPLES, StreamResampler

ROOT = Path(__file__).resolve().parents[4]
_CANDIDATES = [
    ROOT / "native" / "ProcessLoopback" / "publish" / "ProcessLoopback.exe",
    ROOT / "native" / "ProcessLoopback" / "bin" / "Release" / "net10.0-windows" / "win-x64" / "ProcessLoopback.exe",
    Path(getattr(sys, "_MEIPASS", "")) / "ProcessLoopback.exe",  # packaged app (_internal)
    Path(sys.executable).parent / "ProcessLoopback.exe",
]

# Meeting apps we know how to find. The browser entries cover Google Meet / web calls.
KNOWN_APPS: dict[str, tuple[str, ...]] = {
    "zoom": ("Zoom.exe",),
    "teams": ("ms-teams.exe", "Teams.exe"),
    "discord": ("Discord.exe",),
    "skype": ("Skype.exe",),
    "whatsapp": ("WhatsApp.exe", "WhatsApp.Root.exe"),
    "telegram": ("Telegram.exe",),
    "chrome": ("chrome.exe",),
    "edge": ("msedge.exe",),
    "firefox": ("firefox.exe",),
    "brave": ("brave.exe",),
    "opera": ("opera.exe",),
}


class MeetingAudioUnavailableError(RuntimeError):
    pass


def helper_path() -> Path:
    for p in _CANDIDATES:
        if p.exists():
            return p
    raise MeetingAudioUnavailableError(
        "ProcessLoopback.exe not found — build it: dotnet publish native/ProcessLoopback -c Release -o "
        "native/ProcessLoopback/publish")


def supported() -> bool:
    return sys.platform == "win32" and sys.getwindowsversion().build >= 19041


# ---- process discovery (Toolhelp32, no extra dependencies) ----------------------------
class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    parent: int
    exe: str


def list_processes() -> list[ProcessInfo]:
    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    entry = _PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    out = []
    try:
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            out.append(ProcessInfo(entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile))
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return out


def find_app_root(app: str, processes: list[ProcessInfo] | None = None) -> ProcessInfo | None:
    """The top-most process of a running app (its tree contains the audio child processes)."""
    exes = {e.lower() for e in KNOWN_APPS.get(app.lower(), (app,))}
    procs = processes if processes is not None else list_processes()
    by_pid = {p.pid: p for p in procs}
    matches = [p for p in procs if p.exe.lower() in exes]
    roots = [p for p in matches if by_pid.get(p.parent) is None or by_pid[p.parent].exe.lower() not in exes]
    return min(roots, key=lambda p: p.pid) if roots else None


def running_meeting_apps() -> list[str]:
    procs = list_processes()
    return [name for name in KNOWN_APPS if find_app_root(name, procs)]


# ---- the AudioSource -----------------------------------------------------------------
class ProcessAudioCapture:
    """AudioSource of another app's audio (app=...) or of all audio except ours (app=None)."""

    RATE = 48_000

    def __init__(self, app: str | None = None, *, pid: int | None = None, max_queue_seconds: float = 5.0):
        if not supported():
            raise MeetingAudioUnavailableError("Per-app audio capture needs Windows 10 2004+ or Windows 11")
        self.app = app
        if pid is not None:
            self.pid, self.mode, self.description = pid, "include", f"pid {pid}"
        elif app:
            root = find_app_root(app)
            if root is None:
                raise MeetingAudioUnavailableError(f"{app} is not running")
            self.pid, self.mode, self.description = root.pid, "include", f"{root.exe} (pid {root.pid})"
        else:
            self.pid, self.mode, self.description = os.getpid(), "exclude", "all apps except the translator"
        self._proc: subprocess.Popen | None = None
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=int(max_queue_seconds * 50))
        self._pending = np.zeros(0, np.float32)
        self.error: str | None = None
        self.dropped = 0
        self._resampler = StreamResampler(self.RATE)

    def start(self) -> None:
        exe = helper_path()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen([str(exe), "--pid", str(self.pid), "--mode", self.mode,
                                       "--rate", str(self.RATE)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        threading.Thread(target=self._read_stdout, daemon=True, name="loopback-pcm").start()
        threading.Thread(target=self._read_stderr, daemon=True, name="loopback-log").start()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _read_stderr(self) -> None:
        for line in self._proc.stderr:
            text = line.decode(errors="replace").strip()
            if text.startswith("ERROR"):
                self.error = text

    def _read_stdout(self) -> None:
        bytes_per_block = self.RATE * 4 // 50  # 20 ms of s16 stereo
        stream = self._proc.stdout
        while True:
            data = stream.read(bytes_per_block)
            if not data:
                return
            pcm = np.frombuffer(data[: len(data) // 4 * 4], dtype=np.int16).reshape(-1, 2)
            block = self._resampler(pcm)
            try:
                self._frames.put_nowait(block)
            except queue.Full:
                self.dropped += 1

    def read_frame(self, timeout: float | None = 1.0) -> np.ndarray | None:
        while len(self._pending) < FRAME_SAMPLES:
            try:
                self._pending = np.concatenate([self._pending, self._frames.get(timeout=timeout)])
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    raise MeetingAudioUnavailableError(self.error or "audio capture helper exited")
                return None
        frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
        return frame
