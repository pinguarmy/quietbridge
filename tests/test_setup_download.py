"""setup.sh's downloader against a local HTTP server: 429 + Retry-After, resume after a cut-off
transfer, a server that ignores Range, permanent errors, truncated files, and the user agent.

setup.ps1 has the same logic in PowerShell; it is not exercised here (no PowerShell on Linux CI)."""
import http.server
import os
import shutil
import subprocess
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = os.path.join(ROOT, "setup.sh")
BODY = bytes(range(256)) * 400  # 100 KiB

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash") or not shutil.which("curl"),
    reason="setup.sh needs bash and curl (setup.ps1 covers Windows)")


class Handler(http.server.BaseHTTPRequestHandler):
    hits: dict = {}
    agents: list = []

    def log_message(self, *args):
        pass

    def _send(self, code, body=b"", headers=()):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        n = self.hits[self.path] = self.hits.get(self.path, 0) + 1
        self.agents.append(self.headers.get("User-Agent", ""))
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        if self.path == "/flaky429":
            if n == 1:
                return self._send(429, b"slow down", [("Retry-After", "1")])
            return self._send(200, BODY)
        if self.path in ("/cutoff", "/norange"):
            if n == 1:  # promise the whole file, send half, hang up
                self.send_response(200)
                self.send_header("Content-Length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY[:len(BODY) // 2])
                self.wfile.flush()
                self.connection.shutdown(2)
                return
            if self.path == "/cutoff" and start:
                return self._send(206, BODY[start:],
                                  [("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")])
            return self._send(200, BODY)
        if self.path == "/always503":
            return self._send(503, b"busy")
        if self.path == "/tiny":
            return self._send(200, b"<html>")
        return self._send(404, b"no")


@pytest.fixture()
def server():
    Handler.hits, Handler.agents = {}, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(tmp_path, url, minimum=1, retries=3):
    dest = tmp_path / "out" / "file.bin"
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(LIPFLOW_DOWNLOAD_RETRIES=str(retries), LIPFLOW_DOWNLOAD_BACKOFF="0", NO_PROXY="127.0.0.1")
    script = f'source "{SETUP}"; get "$1" "$2" "$3"'
    proc = subprocess.run(["bash", "-c", script, "setup-test", url, str(dest), str(minimum)],
                          capture_output=True, text=True, env=env, timeout=60)
    return proc, dest


def test_429_honours_retry_after_then_succeeds(server, tmp_path):
    proc, dest = get(tmp_path, server + "/flaky429")
    assert proc.returncode == 0, proc.stderr
    assert dest.read_bytes() == BODY
    assert Handler.hits["/flaky429"] == 2
    assert "Retry-After 1" in proc.stderr
    assert not (tmp_path / "out" / "file.bin.part").exists()
    assert not (tmp_path / "out" / "file.bin.headers").exists()


def test_cut_off_transfer_resumes_with_a_range_request(server, tmp_path):
    proc, dest = get(tmp_path, server + "/cutoff")
    assert proc.returncode == 0, proc.stderr
    assert dest.read_bytes() == BODY and Handler.hits["/cutoff"] == 2


def test_server_ignoring_range_restarts_from_scratch(server, tmp_path):
    proc, dest = get(tmp_path, server + "/norange")
    assert proc.returncode == 0, proc.stderr
    assert dest.read_bytes() == BODY
    assert "can't resume" in proc.stderr


def test_permanent_http_error_fails_at_once_and_names_the_url(server, tmp_path):
    proc, dest = get(tmp_path, server + "/missing")
    assert proc.returncode == 1 and not dest.exists()
    assert Handler.hits["/missing"] == 1
    assert server + "/missing" in proc.stderr


def test_temporary_errors_give_up_after_the_retry_budget(server, tmp_path):
    proc, dest = get(tmp_path, server + "/always503", retries=3)
    assert proc.returncode == 1 and not dest.exists()
    assert Handler.hits["/always503"] == 3
    assert "giving up after 3 attempts: " + server + "/always503" in proc.stderr


def test_too_small_download_is_rejected(server, tmp_path):
    proc, dest = get(tmp_path, server + "/tiny", minimum=100, retries=2)
    assert proc.returncode == 1 and not dest.exists()
    assert "expected at least 100" in proc.stderr


def test_existing_file_is_not_downloaded_again(server, tmp_path):
    dest = tmp_path / "out" / "file.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"done")
    proc, _ = get(tmp_path, server + "/flaky429")
    assert proc.returncode == 0 and not Handler.hits


def test_user_agent_is_generic(server, tmp_path):
    get(tmp_path, server + "/flaky429")
    assert Handler.agents and all(a.startswith("lipflow-setup/0.1") for a in Handler.agents)
    assert not any("@" in a or "github" in a.lower() for a in Handler.agents)


def test_unknown_option_is_rejected():
    proc = subprocess.run(["bash", SETUP, "--sampels"], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2 and "unknown option" in proc.stderr


def _launcher(function, home, checkout, xdg=None):
    env = {**os.environ, "HOME": str(home)}
    env.pop("XDG_DATA_HOME", None)
    if xdg is not None:
        env["XDG_DATA_HOME"] = str(xdg)
    return subprocess.run(["bash", "-c", f'source "{SETUP}"; {function} "$1"', "setup-test", str(checkout)],
                          capture_output=True, text=True, env=env, timeout=30)


def test_linux_launcher_is_written_idempotently_with_a_quoted_exec(tmp_path):
    (tmp_path / "home" / ".local" / "share").mkdir(parents=True)
    checkout = tmp_path / 'my "apps" $HOME %u'
    for _ in range(2):
        proc = _launcher("write_linux_launcher", tmp_path / "home", checkout)
        assert proc.returncode == 0, proc.stderr
    desktop = tmp_path / "home" / ".local" / "share" / "applications" / "quietbridge.desktop"
    lines = desktop.read_text().splitlines()
    assert lines[0] == "[Desktop Entry]" and "Terminal=false" in lines and "Name=QuietBridge" in lines
    exe = str(checkout / ".venv" / "bin" / "quietbridge")
    quoted = exe.replace('"', '\\\\"').replace("$", "\\\\$").replace("%", "%%")
    assert f'Exec="{quoted}"' in lines
    assert list(desktop.parent.iterdir()) == [desktop]  # no temporary files left behind


def test_linux_launcher_honours_xdg_data_home_and_skips_without_it(tmp_path):
    (tmp_path / "xdg").mkdir()
    assert _launcher("write_linux_launcher", tmp_path / "home", tmp_path, tmp_path / "xdg").returncode == 0
    assert (tmp_path / "xdg" / "applications" / "quietbridge.desktop").is_file()
    proc = _launcher("write_linux_launcher", tmp_path / "nohome", tmp_path, tmp_path / "missing")
    assert proc.returncode == 0 and "skipping" in proc.stdout
    assert not (tmp_path / "missing").exists() and not (tmp_path / "nohome").exists()


def test_macos_command_launcher_starts_quietbridge_from_its_folder(tmp_path):
    for _ in range(2):
        assert _launcher("write_macos_command", tmp_path, tmp_path).returncode == 0
    command = tmp_path / "QuietBridge.command"
    assert os.access(command, os.X_OK)
    text = command.read_text()
    assert text.startswith("#!/bin/bash\n") and 'cd "$(dirname "$0")" && exec ./.venv/bin/quietbridge' in text


def test_no_app_is_documented_for_every_launcher():
    help_text = subprocess.run(["bash", SETUP, "--help"], capture_output=True, text=True, timeout=30).stdout
    assert "--no-app" in help_text and "quietbridge.desktop" in help_text and "QuietBridge.command" in help_text
