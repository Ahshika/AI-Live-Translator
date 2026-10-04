import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.engine.history import HistoryStore
from app.main import Engine, create_app

TOKEN = "test-token"
H = {"x-token": TOKEN}


@pytest.fixture
def client():
    engine = Engine(Settings())
    with TestClient(create_app(engine, TOKEN)) as c:
        c.engine = engine
        yield c


def test_every_endpoint_requires_the_token(client):
    for path in ("/api/state", "/api/languages", "/api/doctor", "/api/history"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"x-token": "wrong"}).status_code == 401
    assert client.post("/api/session/start").status_code == 401
    assert client.put("/api/settings", json={"my_language": "en"}).status_code == 401


def test_state_and_languages(client):
    st = client.get("/api/state", headers=H).json()
    assert st["state"] == "idle" and st["settings"]["my_language"] == "ar"
    langs = client.get("/api/languages", headers=H).json()
    codes = {l["code"]: l for l in langs}
    assert len(langs) >= 100 and codes["ar-EG"]["rtl"] and codes["de"]["can_listen"]
    assert codes["ku"]["can_listen"] is False


def test_settings_update_is_validated_and_saved(client):
    r = client.put("/api/settings", headers=H, json={"other_language": "fr", "headphones": "false", "bogus": 1})
    assert r.status_code == 200 and r.json()["other_language"] == "fr" and r.json()["headphones"] is False
    assert Settings.from_env().other_language == "fr"  # persisted (in the test's temp data dir)
    assert client.put("/api/settings", headers=H, json={"my_language": "xx"}).status_code == 400
    assert client.put("/api/settings", headers=H, json=["nope"]).status_code == 400


def test_session_errors_are_reported_not_crashed(client, monkeypatch):
    from app.engine.session import SessionError

    def boom(demo=False):
        raise SessionError("virtual_mic_missing", "No virtual microphone found")

    monkeypatch.setattr(client.engine, "start", boom)
    r = client.post("/api/session/start", headers=H)
    assert r.status_code == 409 and r.json()["code"] == "virtual_mic_missing"
    assert client.post("/api/session/pause", headers=H).status_code == 400  # nothing running


def test_websocket_requires_token_and_says_hello(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?t=wrong") as ws:
            ws.receive_text()
    with client.websocket_connect(f"/ws?t={TOKEN}") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello" and hello["state"] == "idle"
        client.engine.publish({"type": "partial", "direction": "outgoing", "text": "أنا"})
        assert json.loads(ws.receive_text())["text"] == "أنا"


def test_history_off_by_default_and_text_only(tmp_path):
    store = HistoryStore(path=tmp_path / "h.db")
    msg = {"speaker": "me", "source_language": "ar", "source_text": "مرحبا", "target_language": "de",
           "translated_text": "Hallo", "timestamp": 1.0}
    store.add(msg)
    assert store.recent() == [] and not (tmp_path / "h.db").exists()  # nothing written when off
    store.enabled = True
    store.add(msg)
    assert [r["translated_text"] for r in store.recent()] == ["Hallo"]
    store.clear()
    assert store.recent() == []


def test_ui_is_served(client):
    assert "المترجم الفوري" in client.get("/").text


def test_real_server_websocket_works():
    """Regression: uvicorn silently refuses WebSockets when no WS library is installed —
    TestClient doesn't notice, the real app shows no live text. Use a real server."""
    import socket
    import threading
    import time

    import uvicorn
    from websockets.sync.client import connect

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    engine = Engine(Settings())
    server = uvicorn.Server(uvicorn.Config(create_app(engine, TOKEN), host="127.0.0.1", port=port,
                                           log_level="error", log_config=None))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.02)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws?t={TOKEN}") as ws:
            assert json.loads(ws.recv(timeout=5))["type"] == "hello"
            engine.publish({"type": "final", "direction": "incoming", "translated_text": "مرحبا"})
            assert json.loads(ws.recv(timeout=5))["translated_text"] == "مرحبا"
    finally:
        server.should_exit = True
        t.join(5)


def _dev(i, name, inp, out):
    from app.services.audio.devices import AudioDevice

    return AudioDevice(i, name, "Windows WASAPI", inp, out, 48000.0, False, False)


def test_vb_cable_2x_names_are_recognised():
    """Regression: VB-CABLE 2.1.5.8 calls its playback side 'Speakers (2- VB-Audio Virtual Cable)'."""
    from app.services.audio.virtual_mic import find_virtual_cable, is_virtual_device

    devs = [_dev(18, "Speakers (2- VB-Audio Virtual Cable)", 0, 2), _dev(19, "Speakers (Realtek(R) Audio)", 0, 2),
            _dev(20, "CABLE In 16 Ch (2- VB-Audio Virtual Cable)", 0, 2),
            _dev(22, "CABLE Output (2- VB-Audio Virtual Cable)", 2, 0), _dev(24, "Microphone (USB-Audio-1.0)", 1, 0)]
    cable = find_virtual_cable(devs)
    assert cable.playback.index == 18 and cable.mic_name == "CABLE Output (2- VB-Audio Virtual Cable)"
    old = find_virtual_cable([_dev(1, "CABLE Input (VB-Audio Virtual Cable)", 0, 2), _dev(2, "CABLE Output (VB-Audio Virtual Cable)", 2, 0)])
    assert old.playback.index == 1
    assert find_virtual_cable([_dev(19, "Speakers (Realtek(R) Audio)", 0, 2)]) is None
    assert is_virtual_device("CABLE Output (2- VB-Audio Virtual Cable)") and not is_virtual_device("Microphone (USB-Audio-1.0)")


def test_default_mic_never_picks_the_virtual_cable(monkeypatch):
    from app.services.audio import devices

    devs = [_dev(22, "CABLE Output (2- VB-Audio Virtual Cable)", 2, 0), _dev(24, "Microphone (USB-Audio-1.0)", 1, 0)]
    monkeypatch.setattr(devices, "input_devices", lambda host_api=None: devs)
    monkeypatch.setattr(devices.sd, "query_devices", lambda kind=None: {"name": "CABLE Output (2- VB-Audio Virtual Cable)"})
    assert devices.default_input_device().name == "Microphone (USB-Audio-1.0)"


def test_virtual_cable_never_becomes_our_mic_or_speaker():
    """Regression: a user picked 'CABLE Output' as mic and the cable as speaker in our app."""
    bad = Settings(input_device="CABLE Output (2- VB-Audio Virtual Cable)",
                   output_device="Speakers (2- VB-Audio Virtual Cable)")
    engine = Engine(bad)
    assert engine.settings.input_device is None and engine.settings.output_device is None
    assert Settings.from_env().input_device is None  # the fix was saved
    with TestClient(create_app(engine, TOKEN)) as c:
        r = c.put("/api/settings", headers=H, json={"output_device": "Speakers (2- VB-Audio Virtual Cable)"})
        assert r.json()["output_device"] is None
        r = c.put("/api/settings", headers=H, json={"output_device": "Speakers (USB-Audio-1.0)"})
        assert r.json()["output_device"] == "Speakers (USB-Audio-1.0)"
