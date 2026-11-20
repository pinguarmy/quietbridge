"""Verified, resumable downloads of the stock model files; never installs or replaces an app.

One implementation serves ``python -m quietbridge.assets --download``,
``python -m quietbridge.signs.assets --download`` and the page's Setup panel
(``SetupDownloads``, used by quietbridge.server). Only the fixed URLs below are
ever fetched: nothing here accepts a URL or a path from a request.

Every file is written to ``<name>.part`` (the same name setup.sh and setup.ps1
use, so either can resume the other's partial file), checked for its exact size
and SHA-256, and only then renamed to its final name. An existing final file is
never overwritten.

Lip-reading files go to lipflow.paths.models_dir() (the checkout's models/ when
running from source), the only place lipflow.vsr and lipflow.face load them
from. The stock gesture model goes to quietbridge.signs.assets.default_model_path().
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import http.client
import importlib.util
import ipaddress
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "quietbridge-model-download/0.1 (verified model downloader; urllib)"
CHUNK = 1024 * 1024
PROGRESS_BYTES = 4 * 1024 * 1024  # report progress at least this often (or every 0.2 s)
# Without a known size, a response may not grow beyond this (bounded disk use).
MAX_UNKNOWN_BYTES = 2 * 1024 ** 3
MAX_REDIRECTS = 5

LIPS_LICENSE = ("The lip-reading weights are derived from the LRS3 dataset and are for non-commercial "
                "research use only (see NOTICE). Their authors' licence limits them to research and private "
                "study, and the copy downloaded here is a third-party mirror that states no licence of its "
                "own, so check that your use fits. Also included: Google's MediaPipe face landmarker and the "
                "Auto-AVSR tokenizer, under their publishers' terms.")
GESTURES_LICENSE = ("Google's MediaPipe gesture recognizer model, downloaded from Google under the terms of "
                    "its MediaPipe model card. It is not redistributed with QuietBridge.")


@dataclass(frozen=True)
class Asset:
    name: str      # relative name shown to people, e.g. "vsr/model.pth"
    url: str       # fixed upstream URL (HTTPS)
    size: int      # exact expected bytes
    sha256: str    # pinned content hash
    min_bytes: int = 1  # a present file below this is a truncated legacy download
    location: str = "models"  # "models" (lipflow.paths.MODELS) or "gesture"

    def path(self) -> Path:
        if self.location == "gesture":
            from .signs import assets as sign_assets
            return sign_assets.default_model_path()
        from lipflow import paths
        return Path(paths.MODELS) / self.name  # read at call time: tests and env overrides


# Must list the same URLs and destinations as scripts/models.txt (a test checks this).
# Sizes and hashes: Hugging Face's X-Linked-Size / X-Linked-ETag (LFS SHA-256) for the
# .pth files, the Git blob SHA-1 for the .json files cross-checked against local copies,
# and Google Cloud Storage's MD5 for the face landmarker, all verified on 2026-11-20.
LIPS_ASSETS = (
    Asset("vsr/model.json", "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json",
          5512, "e062e8c58579f54cb51087e639e674781189809905db4162efe0fe1325356d4f", 100),
    Asset("vsr/model.pth", "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth",
          1001908942, "e740cef369abeabd0ba2c18e37a0661342e1d94d432d6caa77755a11821d8fe3", 900_000_000),
    Asset("lm/model.json", "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json",
          951, "6e2d2004e4066af3e5a3d74c53b0d648fd53904da07178671d1cbe28a206d548", 100),
    Asset("lm/model.pth", "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth",
          215076928, "c75aa39020dec98f432c8689b145d3f4cc407d4daa90a0c64202386c34f83c18", 200_000_000),
    Asset("lm/unigram5000.model",
          "https://raw.githubusercontent.com/mpc001/auto_avsr/main/spm/unigram/unigram5000.model",
          327113, "2c1d648ccf5fdce6612ecbea2ffbd1cab5aabc90458781186175c3911c4bdb1e", 100_000),
    Asset("face_landmarker.task",
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          3758596, "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff", 3_000_000),
)


def _gesture_asset() -> Asset:
    from .signs import assets as sign_assets
    return Asset("gesture_recognizer.task", sign_assets.MODEL_URL, sign_assets.MODEL_SIZE,
                 sign_assets.MODEL_SHA256, 1, "gesture")


GROUPS = {
    "lips": {"label": "Lip reading", "license_note": LIPS_LICENSE},
    "gestures": {"label": "Stock hand gestures", "license_note": GESTURES_LICENSE},
}

# Kept for callers of the earlier module: relative name -> URL of the lip-reading files.
ASSETS = {asset.name: asset.url for asset in LIPS_ASSETS}


def group_assets(group_id: str) -> tuple[Asset, ...]:
    if group_id == "lips":
        return LIPS_ASSETS
    if group_id == "gestures":
        return (_gesture_asset(),)
    raise KeyError(group_id)


def size_note(total: int) -> str:
    if total >= 1e9:
        text = f"about {total / 1e9:.1f} GB"
    elif total >= 1e6:
        text = f"about {round(total / 1e6)} MB"
    else:
        text = f"about {max(1, round(total / 1e3))} KB"
    return text + ", one time"


# --------------------------------------------------------------------------- errors


class DownloadError(ValueError):
    """A download failed; ``str()`` is short and says what to do. ``retry`` marks temporary failures."""

    def __init__(self, message, retry=False, retry_after=None):
        super().__init__(message)
        self.retry = retry
        self.retry_after = retry_after


class DownloadCancelled(Exception):
    """Stopped on request; the .part file is kept for resuming."""


class _LocalFileError(Exception):
    def __init__(self, path, exc):
        super().__init__(str(exc))
        self.path = path
        self.exc = exc


def _host(url):
    return urllib.parse.urlsplit(url).hostname or "the download server"


def _local_message(path, exc, needed=None):
    if getattr(exc, "errno", None) == errno.ENOSPC:
        extra = f" (needs about {needed / 1e9:.1f} GB)" if needed and needed >= 1e8 else ""
        return f"Not enough disk space to save {Path(path).name}{extra}. Free some space, then try again."
    reason = getattr(exc, "strerror", None) or str(exc)
    return f"Could not save {path}: {reason}. Check that the folder is writable, then try again."


def _network_message(tracker):
    host, origin = tracker["host"], tracker.get("origin")
    via = f" (where {origin} sent the download)" if origin and origin != host else ""
    return f"Could not reach {host}{via}. Check your connection, then try again."


def _http_message(name, host, code):
    if code in (401, 403, 407):
        return (f"{host} refused the download (HTTP {code}). A proxy or firewall may be blocking it; "
                "try another network, then try again.")
    if code == 404:
        return f"{name} is no longer available from {host} (HTTP 404). Update QuietBridge, then try again."
    if code == 429 or code >= 500:
        return f"{host} is busy (HTTP {code}). Wait a few minutes, then try again."
    return f"{host} answered HTTP {code} for {name}. Try again later."


# --------------------------------------------------------------------------- one attempt


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow only HTTPS redirects to public host names; remember the last host for messages."""

    max_redirections = MAX_REDIRECTS

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlsplit(newurl)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https":
            raise DownloadError(f"{_host(req.full_url)} redirected the download to an insecure "
                                "(non-HTTPS) address, so it was refused.")
        literal = None
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            pass
        if not host or host == "localhost" or host.endswith(".localhost") or literal is not None:
            raise DownloadError(f"{_host(req.full_url)} redirected the download to a local or numeric "
                                "address, so it was refused.")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            self.tracker["host"] = host
        return new


def _open(url, headers, timeout, tracker):
    opener = urllib.request.build_opener(_Redirects(tracker))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    return opener.open(request, timeout=timeout)


def _declared_total(response, offset, resume):
    """Full file size the server committed to, or None if it did not say."""
    if resume:
        match = re.fullmatch(r"bytes \d+-\d+/(\d+)", response.headers.get("Content-Range", "").strip())
        if match:
            return int(match.group(1))
    length = response.headers.get("Content-Length", "").strip()
    if length.isdigit():
        return (offset if resume else 0) + int(length)
    return None


def _hash_prefix(part, offset):
    digest = hashlib.sha256()
    try:
        with part.open("rb") as existing:
            remaining = offset
            while remaining:
                block = existing.read(min(CHUNK, remaining))
                if not block:
                    break
                digest.update(block)
                remaining -= len(block)
    except OSError as exc:
        raise _LocalFileError(part, exc) from exc
    return digest


def download(url, path, *, name=None, expected_size=None, sha256=None, progress=None, cancel=None,
             timeout=30, _restarted=False):
    """Fetch url to path through a resumable .part file; install only a complete file.

    One attempt (``fetch`` adds retries). urllib does not raise when a
    connection closes before Content-Length bytes arrive, so the byte count is
    checked before the .part file is renamed. With ``expected_size`` the body
    may never exceed it; with ``sha256`` the content must match.
    ``progress(bytes_so_far)`` is called as data arrives; a set ``cancel``
    event stops the transfer and keeps the .part file.
    """
    path = Path(path)
    name = name or path.name
    part = path.with_suffix(path.suffix + ".part")
    tracker = {"host": _host(url), "origin": _host(url)}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        offset = part.stat().st_size if part.exists() else 0
    except OSError as exc:
        raise _LocalFileError(path.parent, exc) from exc
    if expected_size is not None and offset > expected_size:
        part.unlink(missing_ok=True)  # cannot be a prefix of the expected file
        offset = 0
    if expected_size is not None and offset == expected_size and sha256:
        if _hash_prefix(part, offset).hexdigest() == sha256:
            part.replace(path)  # an earlier run received everything but was stopped before renaming
            return path
        part.unlink(missing_ok=True)
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        response = _open(url, headers, timeout, tracker)
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code == 416 and offset and not _restarted:
            # The partial file cannot be continued (e.g. it is already complete
            # but unverified, or the upstream file changed): start again.
            part.unlink()
            return download(url, path, name=name, expected_size=expected_size, sha256=sha256,
                            progress=progress, cancel=cancel, timeout=timeout, _restarted=True)
        retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
        raise DownloadError(_http_message(name, tracker["host"], exc.code),
                            retry=exc.code in (408, 425, 429) or exc.code >= 500,
                            retry_after=int(retry_after) if retry_after.strip().isdigit() else None) from exc
    except DownloadError:
        raise
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise DownloadError(_network_message(tracker), retry=True) from exc
    with response:
        status = getattr(response, "status", 200)
        resume = bool(offset) and status == 206 and \
            response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
        if status == 206 and not resume:
            # A partial body for another range must never become the whole file.
            if _restarted:
                raise DownloadError(f"{name}: the server returned an unexpected byte range. Try again later.",
                                    retry=True)
            part.unlink(missing_ok=True)
            return download(url, path, name=name, expected_size=expected_size, sha256=sha256,
                            progress=progress, cancel=cancel, timeout=timeout, _restarted=True)
        declared = _declared_total(response, offset, resume)
        if expected_size is not None and declared is not None and declared != expected_size:
            part.unlink(missing_ok=True)
            raise DownloadError(f"{tracker['host']} offered a different {name} ({declared} bytes, expected "
                                f"{expected_size}), so it was not installed. Update QuietBridge, then try again.")
        limit = expected_size if expected_size is not None else (declared or MAX_UNKNOWN_BYTES)
        count = offset if resume else 0
        digest = _hash_prefix(part, offset) if (resume and sha256) else hashlib.sha256()
        if progress:
            progress(count)
        last_report, last_count = time.monotonic(), count
        try:
            out = part.open("ab" if resume else "wb")
        except OSError as exc:
            raise _LocalFileError(part, exc) from exc
        with out:
            while True:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled(name)
                try:
                    chunk = response.read(CHUNK)
                except (OSError, http.client.HTTPException, AttributeError, ValueError) as exc:
                    if cancel is not None and cancel.is_set():
                        raise DownloadCancelled(name) from exc
                    raise DownloadError(f"{name}: the connection to {tracker['host']} was interrupted "
                                        f"after {count} bytes; try again to resume.", retry=True) from exc
                if not chunk:
                    break
                if count + len(chunk) > limit:
                    out.close()
                    part.unlink(missing_ok=True)
                    raise DownloadError(f"{tracker['host']} sent more data than expected for {name}, so it "
                                        "was discarded. Try again later.")
                try:
                    out.write(chunk)
                except OSError as exc:
                    raise _LocalFileError(part, exc) from exc
                digest.update(chunk)
                count += len(chunk)
                now = time.monotonic()
                if progress and (now - last_report >= 0.2 or count - last_count >= PROGRESS_BYTES):
                    progress(count)
                    last_report, last_count = now, count
    if progress:
        progress(count)
    expected = expected_size if expected_size is not None else declared
    if expected is not None and count != expected:
        # Keep the .part file: rerunning resumes from the bytes received.
        raise DownloadError(f"{name}: incomplete download ({count} of {expected} bytes); "
                            "try again to resume.", retry=True)
    if count == 0:
        raise DownloadError(f"{name}: the server returned an empty file. Try again later.", retry=True)
    if sha256 and digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{name} did not match the expected file, so it was discarded. Try again; if "
                            "this repeats, the download source has changed.")
    if expected is None:
        print(f"  {name}: server did not declare a size; completeness not verified", flush=True)
    try:
        if path.exists():
            raise FileExistsError(errno.EEXIST, "a file with this name appeared meanwhile; not replacing it")
        part.replace(path)
    except OSError as exc:
        raise _LocalFileError(path, exc) from exc
    return path


# --------------------------------------------------------------------------- retries


def fetch(asset: Asset, *, path=None, progress=None, cancel=None, retries=4, backoff=2.0, max_wait=30.0,
          sleep=None):
    """Download one asset with bounded retries (exponential backoff, Retry-After honoured).

    Raises DownloadError with a short, actionable message, or DownloadCancelled.
    An existing final file is returned untouched (never overwritten).
    """
    path = Path(path) if path is not None else asset.path()
    if path.is_file():
        return path
    cancel = cancel if cancel is not None else threading.Event()
    sleep = sleep if sleep is not None else cancel.wait
    wait = backoff
    for attempt in range(1, retries + 1):
        if cancel.is_set():
            raise DownloadCancelled(asset.name)
        try:
            return download(asset.url, path, name=asset.name, expected_size=asset.size, sha256=asset.sha256,
                            progress=progress, cancel=cancel)
        except _LocalFileError as exc:
            raise DownloadError(_local_message(exc.path, exc.exc, asset.size)) from exc
        except DownloadError as exc:
            if not exc.retry or attempt == retries:
                raise
            delay = min(exc.retry_after if exc.retry_after is not None else wait, max_wait)
        sleep(delay)
        wait = min(wait * 2, max_wait)
    raise DownloadCancelled(asset.name)  # unreachable: the loop returns or raises


# --------------------------------------------------------------------------- status


def _size(path):
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def asset_status(asset: Asset):
    """("present" | "missing" | "short", bytes on disk incl. a resumable .part)."""
    path = asset.path()
    size = _size(path)
    if size is not None:
        return ("present" if size >= asset.min_bytes else "short"), size
    part = _size(path.with_suffix(path.suffix + ".part")) or 0
    return "missing", min(part, asset.size)


def group_disk_status(group_id: str):
    assets = group_assets(group_id)
    done = total = 0
    short = []
    missing = []
    for asset in assets:
        state, size = asset_status(asset)
        total += asset.size
        if state == "present":
            done += asset.size
        elif state == "short":
            short.append(asset)
        else:
            done += size
            missing.append(asset)
    return {"assets": assets, "done": done, "total": total, "short": short, "missing": missing}


def group_ready(group_id: str) -> bool:
    """Every file of the group is present (by size; hashes were checked when downloaded)."""
    disk = group_disk_status(group_id)
    return not disk["missing"] and not disk["short"]


def _short_message(asset):
    path = asset.path()
    return (f"{path} is smaller than expected (an interrupted download?). Move it away, "
            "then try again.")


class SetupDownloads:
    """The page's model downloads: one worker thread, groups queued in order.

    Only the fixed group ids "lips" and "gestures" are accepted. ``on_ready``
    is called with a group id after every file of that group was installed.
    """

    def __init__(self, on_ready=None, fetcher=None, log=None):
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._active = None
        self._cancel = threading.Event()
        self._errors: dict[str, str] = {}
        self._progress: dict[str, tuple[str, int]] = {}  # group -> (asset name, bytes of that file)
        self._thread = None
        self._closed = False
        self._on_ready = on_ready
        self._fetch = fetcher or fetch
        self._log = log or (lambda message: print(message, file=sys.stderr, flush=True))

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None or bool(self._queue)

    def start(self, group_id: str):
        group_assets(group_id)  # KeyError for an unknown id
        with self._lock:
            if self._closed:
                raise RuntimeError("QuietBridge is closing")
            if group_id == self._active or group_id in self._queue:
                return  # idempotent while already downloading
            self._errors.pop(group_id, None)
            self._queue.append(group_id)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="quietbridge-model-download",
                                                daemon=True)
                self._thread.start()

    def cancel(self, group_id: str):
        group_assets(group_id)
        with self._lock:
            if group_id in self._queue:
                self._queue.remove(group_id)
            if self._active == group_id:
                self._cancel.set()

    def close(self, timeout=5.0):
        with self._lock:
            self._closed = True
            self._queue.clear()
            self._cancel.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self):
        while True:
            with self._lock:
                if self._closed or not self._queue:
                    self._active = None
                    return
                group_id = self._active = self._queue.pop(0)
                self._cancel = cancel = threading.Event()
            try:
                completed = self._download_group(group_id, cancel)
            except DownloadCancelled:
                completed = False
            except DownloadError as exc:
                completed = False
                with self._lock:
                    self._errors[group_id] = str(exc)
                self._log(f"QuietBridge: {GROUPS[group_id]['label']} download failed: {exc}")
            except Exception as exc:  # never let the worker die silently
                completed = False
                with self._lock:
                    self._errors[group_id] = f"The download stopped unexpectedly ({exc.__class__.__name__}). Try again."
                self._log("QuietBridge: unexpected download failure:\n" + traceback.format_exc())
            finally:
                with self._lock:
                    self._progress.pop(group_id, None)
                    self._active = None
            if completed and self._on_ready is not None:
                try:
                    self._on_ready(group_id)
                except Exception:
                    self._log("QuietBridge: download completion hook failed:\n" + traceback.format_exc())

    def _download_group(self, group_id, cancel):
        disk = group_disk_status(group_id)
        if disk["short"]:
            raise DownloadError(_short_message(disk["short"][0]))
        needed = sum(asset.size - asset_status(asset)[1] for asset in disk["missing"])
        if disk["missing"]:
            folder = disk["missing"][0].path().parent
            probe = folder
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            try:
                free = shutil.disk_usage(probe).free
            except OSError:
                free = None
            if free is not None and free < needed + 50 * 1024 * 1024:
                raise DownloadError(f"Not enough disk space in {probe}: about {needed / 1e9:.2f} GB is "
                                    "needed. Free some space, then try again.")
        # Resolve every destination once, when the job starts: a later change of the
        # models folder (an override, or a test restoring it) must never redirect a
        # running download into another directory.
        targets = [(asset, asset.path()) for asset in disk["missing"]]
        for asset, destination in targets:
            def report(count, name=asset.name):
                with self._lock:
                    self._progress[group_id] = (name, count)
            self._fetch(asset, path=destination, progress=report, cancel=cancel)
            if cancel.is_set():
                raise DownloadCancelled(asset.name)
        return True

    def group_state(self, group_id: str) -> dict:
        info = GROUPS[group_id]
        disk = group_disk_status(group_id)
        with self._lock:
            downloading = group_id == self._active or group_id in self._queue
            error = self._errors.get(group_id, "")
            progress = self._progress.get(group_id)
        done = disk["done"]
        if progress is not None:
            for asset in disk["missing"]:
                if asset.name == progress[0]:
                    done += max(0, min(progress[1], asset.size) - asset_status(asset)[1])
        if downloading:
            status, error = "downloading", ""
        elif not disk["missing"] and not disk["short"]:
            status, error = "ready", ""
        elif error:
            status = "error"
        elif disk["short"]:
            status, error = "error", _short_message(disk["short"][0])
        else:
            status = "missing"
        return {"id": group_id, "label": info["label"], "status": status,
                "bytes_done": int(min(done, disk["total"])), "bytes_total": int(disk["total"]),
                "error": error, "size_note": size_note(disk["total"]),
                "license_note": info["license_note"]}

    def ready(self, group_id: str) -> bool:
        return group_ready(group_id)


# --------------------------------------------------------------------------- command line


def download_group_cli(group_id: str, out=None) -> bool:
    """Fetch a group's missing files with printed progress. True when everything is present."""
    out = out or sys.stdout
    disk = group_disk_status(group_id)
    if disk["short"]:
        print(f"  {_short_message(disk['short'][0])}", file=out, flush=True)
        return False
    ok = True
    for asset in disk["missing"]:
        print(f"Downloading {asset.name} ({size_note(asset.size).removesuffix(', one time')})…",
              file=out, flush=True)
        marks = {"next": 64 * 1024 * 1024}

        def report(count, name=asset.name):
            if count >= marks["next"]:
                print(f"  {name}: {count // (1024 * 1024)} MiB", file=out, flush=True)
                marks["next"] = (count // (64 * 1024 * 1024) + 1) * 64 * 1024 * 1024
        try:
            fetch(asset, progress=report, sleep=time.sleep)
        except DownloadError as exc:
            print(f"  {asset.name}: download failed: {exc}", file=out, flush=True)
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--download", action="store_true",
                        help="Fetch missing lip-reading files (~1.2 GB); research-use weights, see NOTICE")
    args = parser.parse_args()
    from lipflow import paths
    print(f"Model folder: {paths.MODELS}", flush=True)
    if args.download:
        print(LIPS_LICENSE, flush=True)
        download_group_cli("lips")
    missing = []
    missing_modules = []
    for asset in LIPS_ASSETS:
        state, _ = asset_status(asset)
        print(f"{'OK     ' if state == 'present' else 'MISSING'} models/{asset.name}", flush=True)
        if state != "present":
            missing.append(asset.name)
    for module in ["cv2", "numpy", "torch", "mediapipe"]:
        present = importlib.util.find_spec(module) is not None
        print(f"{'OK     ' if present else 'MISSING'} Python module {module}")
        if not present:
            missing_modules.append(module)
    print("Sign recognizer:", "present" if (ROOT / "quietbridge/signs/recognizer.py").is_file() else "missing")
    print("Stock gesture model: python -m quietbridge.signs.assets --download (or the Setup panel in the app)")
    print("No inference, camera access or accuracy test was performed by this check.")
    if missing or missing_modules:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
