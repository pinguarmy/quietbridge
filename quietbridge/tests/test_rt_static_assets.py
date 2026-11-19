"""Software contract: the loopback server serves the shared design system and icon.

Checks HTTP routing, MIME types and security headers only. Says nothing about
how the pages look or how usable they are.
"""
import http.client
import threading
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from quietbridge import server as server_module
from quietbridge.runtime import Runtime
from quietbridge.server import make_server

CSP = "default-src 'self'; img-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'"


@contextmanager
def serving():
    runtime = Runtime(rehearsal=True, vocabulary={})
    runtime.speaker.close()
    runtime.speaker = Mock(available=True)
    server = make_server(runtime, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.close()


def get(server, path):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response, response.read()
    finally:
        conn.close()


@pytest.mark.parametrize("path, mime", [
    ("/design.css", "text/css; charset=utf-8"),
    ("/favicon.svg", "image/svg+xml; charset=utf-8"),
    ("/style.css", "text/css; charset=utf-8"),
])
def test_static_assets_are_served_with_type_and_hardening_headers(path, mime):
    with serving() as server:
        response, body = get(server, path)
    assert response.status == 200
    assert response.getheader("Content-Type") == mime
    assert body == (server_module.UI / path[1:]).read_bytes()
    assert response.getheader("X-Content-Type-Options") == "nosniff"
    assert response.getheader("Cache-Control") == "no-store"
    assert response.getheader("Content-Security-Policy") == CSP


@pytest.mark.parametrize("page", ["/", "/settings"])
def test_pages_reference_only_assets_the_server_serves(page):
    with serving() as server:
        _, html = get(server, page)
        for asset in ("/design.css", "/style.css", "/favicon.svg", "/app.js"):
            assert f'"{asset}"'.encode() in html
            response, _ = get(server, asset)
            assert response.status == 200, asset


def test_unlisted_ui_files_are_not_served():
    with serving() as server:
        for path in ("/design.css.bak", "/../server.py", "/ui/design.css", "/favicon.ico"):
            response, _ = get(server, path)
            assert response.status == 404, path
