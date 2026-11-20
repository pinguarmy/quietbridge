"""The shared model downloader and the page's download worker, against local fakes (no internet).

Real downloads from the fixed hosts are exercised by hand, not here.
"""
import hashlib
import http.server
import os
import socketserver
import threading
import time
from pathlib import Path

import pytest

from lipflow import paths
from quietbridge import assets
from quietbridge.assets import Asset, DownloadCancelled, DownloadError, SetupDownloads

BODY = os.urandom(300_000)
SHA = hashlib.sha256(BODY).hexdigest()
ROOT = Path(__file__).resolve().parents[2]


class Handler(http.server.BaseHTTPRequestHandler):
    hits: dict = {}
    ranges: list = []

    def log_message(self, *args):
        pass

    def _send(self, code, body=b"", headers=(), length=True):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        if length:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        n = self.hits[self.path] = self.hits.get(self.path, 0) + 1
        rng = self.headers.get("Range")
        self.ranges.append((self.path, rng))
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        port = self.server.server_port
        if self.path in ("/ok", "/cutoff"):
            if self.path == "/cutoff" and n == 1:  # promise everything, send a third, hang up
                self.send_response(200)
                self.send_header("Content-Length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY[:100_000])
                self.wfile.flush()
                self.connection.shutdown(2)
                return
            if start:
                return self._send(206, BODY[start:], [("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")])
            return self._send(200, BODY)
        if self.path == "/429":
            if n == 1:
                return self._send(429, b"slow down", [("Retry-After", "7")])
            return self._send(200, BODY)
        if self.path == "/503":
            return self._send(503, b"busy")
        if self.path == "/404":
            return self._send(404, b"gone")
        if self.path == "/403":
            return self._send(403, b"no")
        if self.path == "/redirect-http":
            return self._send(302, b"", [("Location", f"http://127.0.0.1:{port}/ok")])
        if self.path == "/redirect-local":
            return self._send(302, b"", [("Location", f"https://127.0.0.1:{port}/ok")])
        if self.path == "/oversize":  # no Content-Length: more bytes than the pinned size
            self.send_response(200)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(BODY + b"extra" * 1000)
            self.close_connection = True
            return
        if self.path == "/declared":
            return self._send(200, BODY + b"x")
        if self.path == "/wrong":
            return self._send(200, bytes(len(BODY)))
        return self._send(404)


@pytest.fixture
def web():
    Handler.hits = {}
    Handler.ranges = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr(assets, "CHUNK", 16 * 1024)
    monkeypatch.setattr(assets, "PROGRESS_BYTES", 32 * 1024)


def asset(url, size=len(BODY), sha=SHA):
    return Asset("vsr/model.pth", url, size, sha, 1)


class Sleeps(list):
    def __call__(self, seconds):
        self.append(seconds)


def test_complete_file_is_installed_atomically(web, tmp_path):
    target = tmp_path / "vsr" / "model.pth"
    seen = []
    assert assets.fetch(asset(web + "/ok"), path=target, progress=seen.append) == target
    assert target.read_bytes() == BODY and not target.with_suffix(".pth.part").exists()
    assert seen[-1] == len(BODY)


def test_truncated_transfer_is_retried_and_resumed(web, tmp_path):
    target = tmp_path / "model.pth"
    sleeps = Sleeps()
    assets.fetch(asset(web + "/cutoff"), path=target, sleep=sleeps)
    assert target.read_bytes() == BODY
    assert Handler.ranges[-1] == ("/cutoff", "bytes=100000-")  # resumed, not restarted
    assert sleeps == [2.0]


def test_retry_after_is_honoured_and_capped(web, tmp_path):
    sleeps = Sleeps()
    assets.fetch(asset(web + "/429"), path=tmp_path / "m.pth", sleep=sleeps, max_wait=5)
    assert sleeps == [5]  # Retry-After: 7, capped at max_wait


def test_busy_server_gives_up_after_bounded_retries(web, tmp_path):
    sleeps = Sleeps()
    with pytest.raises(DownloadError, match=r"127\.0\.0\.1 is busy \(HTTP 503\)\. Wait a few minutes"):
        assets.fetch(asset(web + "/503"), path=tmp_path / "m.pth", sleep=sleeps, retries=4)
    assert sleeps == [2.0, 4.0, 8.0] and Handler.hits["/503"] == 4


@pytest.mark.parametrize("path, message", [
    ("/404", "no longer available"),
    ("/403", r"refused the download \(HTTP 403\)\. A proxy or firewall"),
])
def test_permanent_http_errors_are_not_retried(web, tmp_path, path, message):
    sleeps = Sleeps()
    with pytest.raises(DownloadError, match=message):
        assets.fetch(asset(web + path), path=tmp_path / "m.pth", sleep=sleeps)
    assert sleeps == [] and Handler.hits[path] == 1


@pytest.mark.parametrize("path, message", [
    ("/redirect-http", "insecure"),
    ("/redirect-local", "local or numeric address"),
])
def test_unsafe_redirects_are_refused(web, tmp_path, path, message):
    with pytest.raises(DownloadError, match=message):
        assets.fetch(asset(web + path), path=tmp_path / "m.pth", sleep=Sleeps())
    assert Handler.hits.get("/ok") is None  # never followed
    assert not (tmp_path / "m.pth").exists()


def test_body_beyond_the_pinned_size_is_discarded(web, tmp_path):
    target = tmp_path / "m.pth"
    with pytest.raises(DownloadError, match="more data than expected"):
        assets.fetch(asset(web + "/oversize"), path=target, sleep=Sleeps())
    assert not target.exists() and not target.with_suffix(".pth.part").exists()


def test_declared_size_mismatch_is_refused_before_reading(web, tmp_path):
    with pytest.raises(DownloadError, match="offered a different"):
        assets.fetch(asset(web + "/declared"), path=tmp_path / "m.pth", sleep=Sleeps())
    assert not (tmp_path / "m.pth").exists()


def test_wrong_content_is_discarded(web, tmp_path):
    target = tmp_path / "m.pth"
    with pytest.raises(DownloadError, match="did not match the expected file"):
        assets.fetch(asset(web + "/wrong"), path=target, sleep=Sleeps())
    assert not target.exists() and not target.with_suffix(".pth.part").exists()


def test_cancel_keeps_the_partial_file_and_a_later_run_resumes(web, tmp_path):
    target = tmp_path / "m.pth"
    cancel = threading.Event()

    def progress(count):
        if count >= 50_000:
            cancel.set()
    with pytest.raises(DownloadCancelled):
        assets.fetch(asset(web + "/ok"), path=target, progress=progress, cancel=cancel)
    part = target.with_suffix(".pth.part")
    kept = part.stat().st_size
    assert 0 < kept < len(BODY) and not target.exists()
    assets.fetch(asset(web + "/ok"), path=target)
    assert target.read_bytes() == BODY
    assert Handler.ranges[-1] == ("/ok", f"bytes={kept}-")


def test_complete_but_unrenamed_partial_is_verified_and_installed(web, tmp_path):
    target = tmp_path / "m.pth"
    target.with_suffix(".pth.part").write_bytes(BODY)
    assets.fetch(asset(web + "/ok"), path=target)
    assert target.read_bytes() == BODY and Handler.hits == {}


def test_existing_file_is_never_overwritten(web, tmp_path):
    target = tmp_path / "m.pth"
    target.write_bytes(b"mine")
    assert assets.fetch(asset(web + "/ok"), path=target) == target
    assert target.read_bytes() == b"mine" and Handler.hits == {}


def test_unwritable_destination_is_an_actionable_error(web, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(DownloadError, match="Could not save .*Check that the folder is writable"):
        assets.fetch(asset(web + "/ok"), path=blocker / "m.pth", sleep=Sleeps())


class _RefusingProxy(socketserver.StreamRequestHandler):
    """Answers CONNECT with 403 the way the network proxy here refuses a blocked host."""

    def handle(self):
        while self.rfile.readline() not in (b"\r\n", b""):
            pass
        self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")


def test_proxy_refusal_names_the_host(monkeypatch, tmp_path):
    proxy = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _RefusingProxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{proxy.server_address[1]}"
    for name in ("https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, url)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "")
    sleeps = Sleeps()
    try:
        with pytest.raises(DownloadError) as excinfo:
            assets.fetch(asset("https://models.example.test/model.pth"), path=tmp_path / "m.pth",
                         sleep=sleeps, retries=2)
    finally:
        proxy.shutdown()
        proxy.server_close()
    assert str(excinfo.value) == "Could not reach models.example.test. Check your connection, then try again."
    assert sleeps == [2.0]


# ----------------------------------------------------------------- manifest


def test_fixed_list_matches_scripts_models_txt():
    entries = []
    for line in (ROOT / "scripts" / "models.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            url, dest, minimum = line.split()
            entries.append((url, dest, int(minimum)))
    assert entries == [(a.url, f"models/{a.name}", a.min_bytes) for a in assets.LIPS_ASSETS]
    for a in (*assets.LIPS_ASSETS, assets.group_assets("gestures")[0]):
        assert a.url.startswith("https://") and a.size >= a.min_bytes and len(a.sha256) == 64
    # github.com/<repo>/raw/ only redirects to raw.githubusercontent.com, and is refused by some proxies.
    assert not any(url.startswith("https://github.com/") for url, _, _ in entries)


# ----------------------------------------------------------------- the page's worker


REAL_LIPS_ASSETS = assets.LIPS_ASSETS  # the shipped table, before any fixture shrinks it


def shrink_assets(monkeypatch):
    """Use small stand-in sizes for the worker tests.

    The fake fetchers create a file of each asset's size. Extending a file to a
    gigabyte is instant on Linux but writes real zeros on Windows, which made
    these tests take seconds, time out and leave a worker running.
    """
    import dataclasses
    monkeypatch.setattr(assets, "LIPS_ASSETS", tuple(
        dataclasses.replace(a, size=min(a.size, 4096), min_bytes=min(a.min_bytes, 512)) for a in REAL_LIPS_ASSETS))


@pytest.fixture(autouse=True)
def close_download_workers(monkeypatch):
    """Join every download worker before the temporary models folder is undone."""
    created = []
    original = SetupDownloads.__init__

    def tracking(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(SetupDownloads, "__init__", tracking)
    yield
    for manager in created:
        manager.close(timeout=10)


@pytest.fixture
def model_dirs(tmp_path, monkeypatch):
    shrink_assets(monkeypatch)
    monkeypatch.setattr(paths, "MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("QUIETBRIDGE_GESTURE_MODEL", str(tmp_path / "cache" / "gesture.task"))
    return tmp_path


def sparse(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)  # a sparse file: the right size without writing a gigabyte


class FakeFetcher:
    def __init__(self, gate=None, error=None):
        self.calls = []
        self.gate = gate
        self.error = error
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, asset, path=None, progress=None, cancel=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append(asset.name)
            if self.error is not None:
                raise self.error
            if progress:
                progress(asset.size // 2)
            if self.gate is not None:
                while not self.gate.wait(0.01):
                    if cancel.is_set():
                        raise DownloadCancelled(asset.name)
            sparse(path or asset.path(), asset.size)
        finally:
            with self.lock:
                self.active -= 1


def wait_idle(manager, timeout=5):
    deadline = time.monotonic() + timeout
    while manager.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not manager.busy


def test_group_state_contract(model_dirs):
    manager = SetupDownloads(fetcher=FakeFetcher())
    lips = manager.group_state("lips")
    assert set(lips) == {"id", "label", "status", "bytes_done", "bytes_total", "error", "size_note", "license_note"}
    assert lips["status"] == "missing" and lips["bytes_done"] == 0
    # The page's totals come from the asset table; the shipped table adds up to about 1.2 GB.
    assert lips["bytes_total"] == sum(a.size for a in assets.LIPS_ASSETS)
    assert lips["size_note"] == assets.size_note(lips["bytes_total"])
    assert sum(a.size for a in REAL_LIPS_ASSETS) == 1221078042
    assert assets.size_note(1221078042) == "about 1.2 GB, one time"
    assert "non-commercial research use only" in lips["license_note"]
    gestures = manager.group_state("gestures")
    assert gestures["label"] == "Stock hand gestures" and gestures["bytes_total"] == 8373440
    assert gestures["size_note"] == "about 8 MB, one time"


def test_download_installs_the_group_and_reports_ready(model_dirs):
    ready = []
    fetcher = FakeFetcher()
    manager = SetupDownloads(on_ready=ready.append, fetcher=fetcher)
    sparse(Path(paths.MODELS) / "vsr" / "model.json", assets.LIPS_ASSETS[0].size)  # present files are skipped
    manager.start("lips")
    wait_idle(manager)
    assert fetcher.calls == [a.name for a in assets.LIPS_ASSETS if a.name != "vsr/model.json"]
    state = manager.group_state("lips")
    assert state["status"] == "ready" and state["bytes_done"] == state["bytes_total"] and ready == ["lips"]


def test_one_worker_queues_groups_and_start_is_idempotent(model_dirs):
    gate = threading.Event()
    fetcher = FakeFetcher(gate=gate)
    manager = SetupDownloads(fetcher=fetcher)
    manager.start("gestures")
    manager.start("gestures")
    manager.start("lips")
    deadline = time.monotonic() + 2
    while not fetcher.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.group_state("gestures")["status"] == "downloading"
    assert manager.group_state("gestures")["bytes_done"] == 8373440 // 2  # live progress
    assert manager.group_state("lips")["status"] == "downloading"  # queued
    gate.set()
    wait_idle(manager)
    assert fetcher.max_active == 1
    assert fetcher.calls[0] == "gesture_recognizer.task" and fetcher.calls.count("gesture_recognizer.task") == 1
    assert len(fetcher.calls) == 1 + len(assets.LIPS_ASSETS)


def test_cancel_stops_without_an_error(model_dirs):
    gate = threading.Event()
    manager = SetupDownloads(fetcher=FakeFetcher(gate=gate))
    manager.start("gestures")
    time.sleep(0.05)
    manager.cancel("gestures")
    wait_idle(manager)
    state = manager.group_state("gestures")
    assert state["status"] == "missing" and state["error"] == ""


def test_failure_is_reported_and_cleared_by_trying_again(model_dirs):
    logs = []
    fetcher = FakeFetcher(error=DownloadError("Could not reach huggingface.co. Check your connection, then try again."))
    manager = SetupDownloads(fetcher=fetcher, log=logs.append)
    manager.start("lips")
    wait_idle(manager)
    state = manager.group_state("lips")
    assert state["status"] == "error" and state["error"].startswith("Could not reach huggingface.co")
    assert logs and "Could not reach" in logs[0]
    fetcher.error = None
    manager.start("lips")
    assert manager.group_state("lips")["error"] == ""
    wait_idle(manager)
    assert manager.group_state("lips")["status"] == "ready"


def test_unexpected_exception_never_kills_the_worker_silently(model_dirs):
    logs = []
    fetcher = FakeFetcher(error=RuntimeError("boom"))
    manager = SetupDownloads(fetcher=fetcher, log=logs.append)
    manager.start("gestures")
    wait_idle(manager)
    assert manager.group_state("gestures")["error"] == "The download stopped unexpectedly (RuntimeError). Try again."
    assert "Traceback" in logs[0]
    fetcher.error = None
    manager.start("gestures")
    wait_idle(manager)
    assert manager.group_state("gestures")["status"] == "ready"


def test_undersized_existing_file_is_an_error_and_never_replaced(model_dirs):
    short = Path(paths.MODELS) / "vsr" / "model.pth"
    sparse(short, 100)  # below the stand-in minimum of 512 bytes
    fetcher = FakeFetcher()
    manager = SetupDownloads(fetcher=fetcher)
    assert "smaller than expected" in manager.group_state("lips")["error"]
    manager.start("lips")
    wait_idle(manager)
    assert fetcher.calls == [] and short.stat().st_size == 100
    assert manager.group_state("lips")["status"] == "error"


def test_unknown_group_is_refused(model_dirs):
    manager = SetupDownloads(fetcher=FakeFetcher())
    for bad in ("../models", "https://example.com/x", "", "LIPS"):
        with pytest.raises(KeyError):
            manager.start(bad)
        with pytest.raises(KeyError):
            manager.cancel(bad)


def test_close_cancels_and_joins(model_dirs):
    gate = threading.Event()
    manager = SetupDownloads(fetcher=FakeFetcher(gate=gate))
    manager.start("lips")
    time.sleep(0.05)
    started = time.monotonic()
    manager.close()
    assert time.monotonic() - started < 2 and not manager.busy
    with pytest.raises(RuntimeError):
        manager.start("lips")


def test_command_line_uses_the_same_downloader(model_dirs, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(assets, "fetch", lambda a, **kw: calls.append(a.name) or sparse(a.path(), a.size))
    assert assets.download_group_cli("lips") is True
    assert calls == [a.name for a in assets.LIPS_ASSETS]
    monkeypatch.setattr(assets, "fetch", lambda a, **kw: (_ for _ in ()).throw(DownloadError("Could not reach x.")))
    (Path(paths.MODELS) / "face_landmarker.task").unlink()
    assert assets.download_group_cli("lips") is False
    assert "face_landmarker.task: download failed: Could not reach x." in capsys.readouterr().out
