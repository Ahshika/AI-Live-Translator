"""Local engine server: REST for commands + WebSocket for live events, UI served as static files.

Security: binds to 127.0.0.1 only, and every request needs the per-launch random token the
desktop window was opened with (other local programs / web pages can't drive the engine).
No API keys exist anywhere: every model runs locally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core import languages
from app.core.config import InvalidSettingError, Settings
from app.engine import doctor
from app.engine.history import HistoryStore
from app.engine.session import SessionError, TranslationSession
from app.pipeline.factory import Providers

log = logging.getLogger(__name__)
UI_DIR = Path(__file__).resolve().parent / "ui"


class Engine:
    """Owns settings, the (single) translation session, history and event fan-out."""

    def __init__(self, settings: Settings | None = None):
        self.settings = _without_virtual_devices(settings or Settings.from_env())
        self.session: TranslationSession | None = None
        self.providers: Providers | None = None
        self.history = HistoryStore(enabled=self.settings.save_history)
        self.last_error: dict | None = None
        self.installer = None  # AssetInstaller while first-run downloads are running
        self._clients: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    # -- events --------------------------------------------------------------
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: dict[str, Any]) -> None:
        """Called from engine threads; delivered to every connected UI."""
        if event.get("type") == "final" and "message" in event:
            self.history.add(event["message"])
        if event.get("type") == "error":
            self.last_error = event
        if event.get("type") == "setup_progress" and event.get("finished"):
            self.providers = None  # new models / GPU libraries: load fresh on next start
        if self._loop is None:
            return
        data = json.dumps(event, ensure_ascii=False, default=str)
        for q in list(self._clients):
            self._loop.call_soon_threadsafe(_offer, q, data)

    # -- session control -----------------------------------------------------
    def start(self, demo: bool = False) -> None:
        with self._lock:
            if self.session and self.session.state in ("starting", "running", "paused"):
                return
            self.last_error = None
            if self.providers is None:
                self.publish({"type": "status", "state": "loading_models"})
                try:
                    self.providers = Providers.load(self.settings)
                except Exception as exc:
                    log.exception("loading models failed")
                    self.publish({"type": "status", "state": "idle"})
                    raise SessionError("models", str(exc)) from exc
            self.history.new_session()
            if demo:
                self.session = self._demo_session()
            else:
                self.session = TranslationSession(self.settings, on_event=self.publish, providers=self.providers)
            self.session.start()

    def _demo_session(self) -> TranslationSession:
        """Try everything without a meeting or virtual mic: a recorded Arabic speaker 'talks'
        into the mic, a German one 'answers' in the meeting; you hear both translations."""
        from app.engine.session import NullSink
        from app.services.audio.file_source import FileAudioSource
        from app.services.audio.playback import AudioPlayer

        samples = Path(__file__).resolve().parent / "assets" / "samples"
        demo_settings = self.settings.updated(my_language="ar", other_language="de", headphones=True)
        phones = AudioPlayer(self.settings.output_device)
        return TranslationSession(
            demo_settings, on_event=self.publish, providers=self.providers,
            mic=FileAudioSource(samples / "ar_explain_project.wav", realtime=True, tail_silence_s=30),
            meeting=FileAudioSource(samples / "de_question.wav", realtime=True, lead_silence_s=9, tail_silence_s=30),
            outgoing_sink=NullSink(forward=phones.enqueue), headphones=phones)

    def stop(self) -> None:
        with self._lock:
            if self.session:
                self.session.stop()

    def settings_initialised(self) -> bool:
        """Has the user's machine-specific setup (e.g. GPU vs CPU speech model) been saved yet?"""
        from app.core.config import settings_path

        return settings_path().exists()

    def update_settings(self, changes: dict) -> Settings:
        new = _without_virtual_devices(self.settings.updated(**changes))
        model_keys = {"stt_model", "stt_device", "translation_model", "translation_device", "tts_voices"}
        if any(getattr(new, k) != getattr(self.settings, k) for k in model_keys):
            self.providers = None  # reload models on next start
        self.settings = new
        self.settings.save()
        self.history.enabled = bool(new.save_history)
        return new

    def state(self) -> dict:
        s = self.session
        return {
            "state": s.state if s else "idle",
            "settings": self.settings.to_dict(),
            "mic_for_meeting": s.mic_name_for_meeting if s else None,
            "mic_muted": bool(s and s.mic_muted),
            "conversation": [m.__dict__ for m in s.history] if s else [],
            "last_error": self.last_error,
        }


def _offer(q: asyncio.Queue, data: str) -> None:
    """Deliver to one UI; a client that stopped reading loses its oldest events, not the engine."""
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(data)


def _without_virtual_devices(settings: Settings) -> Settings:
    """Picking the virtual cable as OUR mic/speaker is an easy mistake (its names look like a
    mic and a speaker) and breaks everything: we'd translate our own output, and the other
    person's translation would be sent back into the meeting. Fall back to Windows' defaults."""
    from app.services.audio.virtual_mic import is_virtual_device

    fixes = {k: None for k in ("input_device", "output_device")
             if getattr(settings, k) and is_virtual_device(getattr(settings, k))}
    if not fixes:
        return settings
    log.warning("ignoring virtual cable chosen as %s", ", ".join(fixes))
    fixed = settings.updated(**fixes)
    try:
        fixed.save()
    except OSError:
        pass
    return fixed


def create_app(engine: Engine | None = None, token: str | None = None) -> FastAPI:
    engine = engine or Engine()
    token = token or secrets.token_urlsafe(24)
    @asynccontextmanager
    async def lifespan(_app):
        engine.attach_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="AI Live Translator", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.engine, app.state.token = engine, token

    def auth(request: Request) -> None:
        supplied = request.headers.get("x-token") or request.query_params.get("token")
        if not supplied or not secrets.compare_digest(supplied, token):
            raise HTTPException(401, "bad token")

    # -- read ---------------------------------------------------------------
    @app.get("/api/state", dependencies=[Depends(auth)])
    def get_state():
        return engine.state()

    @app.get("/api/languages", dependencies=[Depends(auth)])
    def get_languages():
        from app.providers.tts.piper_provider import PiperProvider

        tts = engine.providers.tts if engine.providers else PiperProvider()
        try:
            tts.load()
            voiced = tts.downloadable_languages() | {v.language for v in tts.voices()} \
                if hasattr(tts, "downloadable_languages") else {v.language for v in tts.voices()}
        except Exception:
            voiced = set()
        return [{"code": l.code, "name": l.name, "native": l.native, "flag": l.flag, "rtl": l.rtl,
                 "can_listen": bool(l.whisper), "has_voice": l.base in voiced}
                for l in languages.LANGUAGES.values()]

    @app.get("/api/devices", dependencies=[Depends(auth)])
    def get_devices():
        from app.services.audio import process_capture as pc
        from app.services.audio.devices import list_devices
        from app.services.audio.virtual_mic import find_virtual_cable, is_virtual_device

        devs = list_devices()
        cable = find_virtual_cable(devs)
        return {
            # The cable belongs in the MEETING app's settings, never in ours.
            "inputs": [d.name for d in devs if d.max_input_channels > 0 and not is_virtual_device(d.name)],
            "outputs": [d.name for d in devs if d.max_output_channels > 0 and not is_virtual_device(d.name)],
            "virtual_mic": cable.mic_name if cable else None,
            "meeting_apps": pc.running_meeting_apps(),
            "known_apps": list(pc.KNOWN_APPS),
        }

    @app.get("/api/doctor", dependencies=[Depends(auth)])
    def get_doctor():
        return doctor.summary(doctor.run_checks(engine.settings))

    @app.get("/api/setup", dependencies=[Depends(auth)])
    def get_setup():
        from app.engine import assets

        st = assets.status(engine.settings)
        st["downloading"] = engine.installer is not None and engine.installer.running
        return st

    @app.post("/api/setup/download", dependencies=[Depends(auth)])
    def start_download():
        from app.engine import assets

        if engine.installer is None or not engine.installer.running:
            if not engine.settings_initialised():
                engine.update_settings({"stt_model": assets.recommended_stt_model()})
            engine.installer = assets.AssetInstaller(engine.settings, on_progress=engine.publish)
            engine.installer.start()
        return {"ok": True}

    @app.get("/api/history", dependencies=[Depends(auth)])
    def get_history(limit: int = 200):
        return engine.history.recent(limit)

    @app.delete("/api/history", dependencies=[Depends(auth)])
    def clear_history():
        engine.history.clear()
        return {"ok": True}

    # -- write --------------------------------------------------------------
    @app.put("/api/settings", dependencies=[Depends(auth)])
    async def put_settings(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "expected an object")
        for key in ("my_language", "other_language"):
            if key in body and body[key] != languages.AUTO and body[key] not in languages.LANGUAGES:
                raise HTTPException(400, f"unknown language {body[key]!r}")
        if body.get("my_language") == languages.AUTO:
            raise HTTPException(400, "your own language can't be auto")
        if engine.session and engine.session.state in ("starting", "running", "paused"):
            raise HTTPException(409, "stop translation before changing settings")
        try:
            return engine.update_settings(body).to_dict()
        except InvalidSettingError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/session/{action}", dependencies=[Depends(auth)])
    async def session_action(action: str, speaker: str = Query("other")):
        loop = asyncio.get_running_loop()
        s = engine.session
        try:
            if action in ("start", "demo"):
                await loop.run_in_executor(None, engine.start, action == "demo")
            elif action == "stop":
                await loop.run_in_executor(None, engine.stop)
            elif action in ("pause", "resume") and s:
                s.set_paused(action == "pause")
            elif action in ("mute", "unmute") and s:
                s.set_mic_muted(action == "mute")
            elif action == "replay" and s and speaker in ("me", "other"):
                ok = await loop.run_in_executor(None, s.replay_last, speaker)
                return {"ok": ok}
            else:
                raise HTTPException(400, f"can't {action} now")
        except SessionError as exc:
            engine.publish({"type": "error", "code": exc.code, "message": str(exc)})
            return JSONResponse({"ok": False, "code": exc.code, "message": str(exc)}, status_code=409)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("session %s failed", action)
            engine.publish({"type": "error", "code": "engine", "message": str(exc)})
            return JSONResponse({"ok": False, "code": "engine", "message": str(exc)}, status_code=500)
        return {"ok": True, **engine.state()}

    # -- live events ----------------------------------------------------------
    @app.websocket("/ws")
    async def ws(websocket: WebSocket, t: str = Query("")):
        if not secrets.compare_digest(t, token):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        engine._clients.add(q)
        try:
            await websocket.send_text(json.dumps({"type": "hello", **engine.state()}, ensure_ascii=False, default=str))
            while True:
                await websocket.send_text(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            engine._clients.discard(q)

    # -- UI -------------------------------------------------------------------
    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

        @app.get("/")
        def index():
            return FileResponse(UI_DIR / "index.html")

    return app
