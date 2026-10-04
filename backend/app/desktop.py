"""Desktop entry point: start the local engine server and open the app window.

The window is the system's Edge WebView2 (built into Windows 11) via pywebview, so the
app stays small: no bundled browser. `--browser` opens the default browser instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import socket
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make `app` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _setup_logging() -> None:
    from app.core.config import data_dir

    log_dir = data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "engine.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="AI Live Translator")
    p.add_argument("--browser", action="store_true", help="open in the default browser instead of a window")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--token", help="fixed access token (development only)")
    p.add_argument("--no-window", action="store_true", help="only run the engine server")
    p.add_argument("--data-dir", help="use this folder for settings/history/logs (testing)")
    p.add_argument("--models-dir", help="use this folder for AI models (testing)")
    args = p.parse_args(argv)
    if args.data_dir:
        os.environ["TRANSLATOR_DATA_DIR"] = args.data_dir
    if args.models_dir:
        os.environ["TRANSLATOR_MODELS_DIR"] = args.models_dir
    _setup_logging()

    import uvicorn

    from app.main import Engine, create_app

    token = args.token or secrets.token_urlsafe(24)
    port = args.port or _free_port()
    engine = Engine()
    app = create_app(engine, token)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", log_config=None))
    thread = threading.Thread(target=server.run, daemon=True, name="engine-server")
    thread.start()
    while not server.started:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}/?token={token}"

    try:
        if args.no_window:
            print(url, flush=True)
            while thread.is_alive():
                time.sleep(0.5)
        elif args.browser:
            import webbrowser

            webbrowser.open(url)
            while thread.is_alive():
                time.sleep(0.5)
        else:
            try:
                import webview

                webview.create_window("المترجم الفوري — AI Live Translator", url, width=1180, height=780,
                                      min_size=(720, 560))
                webview.start()
            except Exception:
                # No WebView2 runtime (older Windows 10): fall back to the default browser.
                logging.getLogger(__name__).exception("window failed; opening the browser instead")
                import webbrowser

                webbrowser.open(url)
                while thread.is_alive():
                    time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()  # release the mic / virtual mic before exiting
        server.should_exit = True
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
