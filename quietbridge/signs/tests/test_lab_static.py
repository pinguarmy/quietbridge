"""The recorder serves the shared design system next to lab.html.

Loopback HTTP against a fake engine: checks routes, types and headers only.
"""
from __future__ import annotations

import http.client
import threading

import pytest

from quietbridge.signs import lab
from quietbridge.signs.lab import create_server
from quietbridge.signs.tests.test_sequence_lab import FakeEngine


@pytest.fixture()
def server(tmp_path):
    srv = create_server(engine=FakeEngine(), port=0, data_root=tmp_path / "seq")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def _get(server, path, host=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path, headers={"Host": host or f"127.0.0.1:{port}"})
        response = conn.getresponse()
        return response, response.read()
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("path", "name", "content_type"),
    [
        ("/design.css", "design.css", "text/css; charset=utf-8"),
        ("/design.css?v=1", "design.css", "text/css; charset=utf-8"),
        ("/favicon.svg", "favicon.svg", "image/svg+xml"),
    ],
)
def test_shared_design_files_are_served_with_security_headers(server, path, name, content_type):
    response, body = _get(server, path)
    assert response.status == 200
    assert body == (lab.UI_DIR / name).read_bytes()
    assert response.getheader("Content-Type") == content_type
    assert response.getheader("X-Content-Type-Options") == "nosniff"
    assert response.getheader("Cache-Control") == "no-store"
    csp = response.getheader("Content-Security-Policy")
    assert "default-src 'self'" in csp and "style-src 'self'" in csp and "img-src 'self'" in csp


def test_csp_script_policy_is_unchanged(server):
    response, _ = _get(server, "/design.css")
    csp = response.getheader("Content-Security-Policy")
    assert "script-src 'self' 'unsafe-inline';" in csp
    assert "http" not in csp and "*" not in csp


def test_static_routes_check_host_and_do_not_serve_other_files(server):
    port = server.server_address[1]
    response, _ = _get(server, "/design.css", host=f"evil.example:{port}")
    assert response.status == 403
    for path in ("/style.css", "/app.js", "/../ui/design.css", "/ui/design.css", "/design.css/"):
        response, body = _get(server, path)
        assert response.status == 404, path
        assert b"QuietBridge design system" not in body


def test_missing_design_file_fails_visibly(server, tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "UI_DIR", tmp_path / "missing")
    response, body = _get(server, "/design.css")
    assert response.status == 404
    assert b"design.css not found" in body


def test_lab_page_links_the_served_files():
    html = lab.LAB_HTML.read_text(encoding="utf-8")
    assert '<link rel="stylesheet" href="/design.css">' in html
    assert '<link rel="icon" href="/favicon.svg" type="image/svg+xml">' in html
    for path in lab.STATIC_FILES:
        assert (lab.UI_DIR / lab.STATIC_FILES[path][0]).is_file()


def test_port_in_use_reports_the_bind_error(tmp_path):
    """A busy port must surface as OSError, not an AttributeError from cleanup."""
    import socket

    from quietbridge.signs import lab

    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    try:
        with pytest.raises(OSError):
            lab.create_server(engine=object(), port=busy.getsockname()[1], data_root=tmp_path)
    finally:
        busy.close()
