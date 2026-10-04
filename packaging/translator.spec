# PyInstaller spec — builds dist/AI Live Translator/ (one folder, windowed).
# Build:  backend\.venv\Scripts\pyinstaller packaging\translator.spec --noconfirm --distpath build\dist --workpath build\work
#
# Deliberately NOT bundled (downloaded on first run instead, see app/engine/assets.py):
#   * AI models (~3 GB)            -> %LOCALAPPDATA%\AI Live Translator\models
#   * NVIDIA CUDA/cuDNN (~2 GB)    -> %LOCALAPPDATA%\AI Live Translator\cuda   (GPU machines only)
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
BACKEND = ROOT / "backend"

datas = [
    (str(BACKEND / "app" / "ui"), "app/ui"),
    (str(BACKEND / "app" / "assets"), "app/assets"),
]
datas += collect_data_files("faster_whisper")            # silero VAD model
datas += collect_data_files("piper")                     # espeak-ng data + Arabic tashkeel model
datas += collect_data_files("webview")

binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")
binaries += [(str(ROOT / "native" / "ProcessLoopback" / "publish" / "ProcessLoopback.exe"), ".")]

hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("app")
    + ["websockets", "websockets.legacy", "clr_loader", "pythonnet", "webview.platforms.winforms",
       "webview.platforms.edgechromium"]
)

excludes = ["nvidia", "PIL", "torch", "tensorflow", "matplotlib", "tkinter", "pytest", "IPython", "PyInstaller"]

a = Analysis(
    [str(BACKEND / "app" / "desktop.py")],
    pathex=[str(BACKEND)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
# Belt and braces: never ship NVIDIA runtime DLLs that slipped in through dependencies,
# except ctranslate2's own small cudnn loader stub.
a.binaries = [b for b in a.binaries if not (b[0].lower().startswith("nvidia") or "\\nvidia\\" in b[1].lower())]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI Live Translator",
    icon=str(ROOT / "packaging" / "icon.ico"),
    console=False,
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="AI Live Translator")
