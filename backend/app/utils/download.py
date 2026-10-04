"""Resumable HTTP download (slow / flaky connections are the norm for our users)."""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)
Progress = Callable[[int, int | None], None]  # (bytes done, total or None)


def download(url: str, dest: Path, *, retries: int = 20, progress: Progress | None = None,
             timeout: float = 60) -> Path:
    """Download url -> dest via dest.part, resuming with HTTP Range after any failure."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        done = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "ai-live-translator"})
        if done:
            req.add_header("Range", f"bytes={done}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if done and resp.status != 206:  # server ignored Range: start over
                    done = 0
                    part.unlink(missing_ok=True)
                length = resp.headers.get("Content-Length")
                total = done + int(length) if length else None
                with open(part, "ab") as f:
                    while chunk := resp.read(256 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            part.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists():  # already complete
                part.replace(dest)
                return dest
            if 400 <= exc.code < 500:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
        log.warning("download %s failed (attempt %d): %s", url, attempt + 1, last_error)
        time.sleep(min(30, 2 ** attempt))
    raise ConnectionError(f"Could not download {url}: {last_error}")
