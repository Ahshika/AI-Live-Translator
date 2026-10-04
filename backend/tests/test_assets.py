import io
import zipfile
from pathlib import Path

from app.core.config import Settings
from app.engine import assets


def test_plan_on_an_empty_machine(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSLATOR_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(assets, "has_nvidia_gpu", lambda: False)
    st = assets.status(Settings(stt_model="small"))
    assert not st["ready"]
    assert [c["id"] for c in st["components"]] == ["stt", "mt", "voices"]  # no GPU -> no CUDA download
    assert st["components"][0]["size_mb"] == 480
    assert assets.recommended_stt_model() == "small"


def test_installer_downloads_missing_models_and_extracts_cuda(tmp_path, monkeypatch):
    models = tmp_path / "models"
    monkeypatch.setenv("TRANSLATOR_MODELS_DIR", str(models))
    monkeypatch.setattr(assets, "has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(assets, "is_frozen", lambda: True)
    monkeypatch.setattr(assets, "_hf_files", lambda repo, dest: (
        [(f"https://hf/{repo}/model.bin", str(dest / "model.bin")),
         (f"https://hf/{repo}/config.json", str(dest / "config.json"))], 10))
    monkeypatch.setattr(assets, "_wheel_url", lambda pkg: f"https://pypi/{pkg}-1.0-py3-none-win_amd64.whl")

    def fake_download(url, dest, progress=None, **kw):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".whl"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                stem = "cublas" if "cublas" in url else "cudnn"
                z.writestr(f"nvidia/{stem}/bin/{stem}64_9.dll", b"dll")
                z.writestr(f"nvidia/{stem}/include/x.h", b"h")
            dest.write_bytes(buf.getvalue())
        else:
            dest.write_bytes(b"x" * 5)
        if progress:
            progress(5, 5)
        return dest

    monkeypatch.setattr(assets, "download", fake_download)

    class FakeTTS:
        def __init__(self, **kw):
            pass

        def ensure_voice(self, lang, progress=None):
            (models / "piper").mkdir(parents=True, exist_ok=True)
            for v in ("ar_JO-kareem-medium", "de_DE-thorsten-high"):
                (models / "piper" / f"{v}.onnx").write_bytes(b"v")

    import app.providers.tts.piper_provider as pp
    monkeypatch.setattr(pp, "PiperProvider", FakeTTS)

    events = []
    inst = assets.AssetInstaller(Settings(), on_progress=events.append)
    inst._run()  # synchronously
    assert inst.error is None, inst.error
    assert assets.status(Settings())["ready"]
    assert (models / "faster-whisper-large-v3-turbo" / "model.bin").exists()
    cuda = assets.cuda_dir()
    assert sorted(p.name for p in cuda.glob("*.dll")) == ["cublas64_9.dll", "cudnn64_9.dll"]
    assert not list(cuda.glob("*.h")) and not list((cuda / "downloads").glob("*.whl"))
    assert events[-1] == {"type": "setup_progress", "component": "all", "finished": True}


def test_installer_reports_network_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSLATOR_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(assets, "has_nvidia_gpu", lambda: False)

    def offline(*a, **k):
        raise ConnectionError("no internet")

    monkeypatch.setattr(assets, "_hf_files", offline)
    events = []
    inst = assets.AssetInstaller(Settings(stt_model="small"), on_progress=events.append)
    inst._run()
    assert inst.error == "no internet"
    assert events[-1]["finished"] is False and events[-1]["error"] == "no internet"


def test_voices_count_as_installed_whatever_quality_was_downloaded(tmp_path, monkeypatch):
    """Regression: settings preferred thorsten-high, the downloader fetched thorsten-medium,
    and the first-run screen then reported 'voices missing' forever."""
    monkeypatch.setenv("TRANSLATOR_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(assets, "has_nvidia_gpu", lambda: False)
    piper = tmp_path / "models" / "piper"
    piper.mkdir(parents=True)
    for v in ("ar_JO-kareem-medium", "de_DE-thorsten-medium"):
        (piper / f"{v}.onnx").write_bytes(b"v")
    comps = {c["id"]: c for c in assets.status(Settings())["components"]}
    assert comps["voices"]["installed"]


def test_preferred_voice_is_the_one_downloaded(tmp_path, monkeypatch):
    import json

    from app.providers.tts import piper_provider as pp

    vdir = tmp_path / "piper"
    vdir.mkdir()
    catalog = {k: {"key": k, "quality": q, "num_speakers": 1, "language": {"family": "de"},
                   "files": {f"de/{k}.onnx": {}, f"de/{k}.onnx.json": {}}}
               for k, q in (("de_DE-thorsten-high", "high"), ("de_DE-thorsten-medium", "medium"))}
    (vdir / "voices.json").write_text(json.dumps(catalog), encoding="utf-8")
    got = []

    def fake_download(url, dest, progress=None, **kw):
        got.append(Path(dest).name)
        Path(dest).write_text(json.dumps({"language": {"family": "de"}, "audio": {"quality": "high"}})
                              if str(dest).endswith(".json") else "x", encoding="utf-8")
        return Path(dest)

    monkeypatch.setattr(pp, "download", fake_download)
    tts = pp.PiperProvider(voices_dir=vdir, preferred={"de": "de_DE-thorsten-high"})
    assert tts.ensure_voice("de") == "de_DE-thorsten-high"
    assert "de_DE-thorsten-high.onnx" in got and "de_DE-thorsten-medium.onnx" not in got
