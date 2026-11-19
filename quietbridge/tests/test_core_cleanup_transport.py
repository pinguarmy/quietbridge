"""Cleanup transport: the API key stays with the configured origin; the worker survives."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from quietbridge.cleanup import CleanupClient
from quietbridge.tests.test_cleanup import Response


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_redirect_never_forwards_api_key_to_another_origin():
    seen = []

    class Sink(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            body = json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET

        def log_message(self, *args):
            pass

    sink = _serve(Sink)

    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/collect")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    redirect = _serve(Redirect)
    client = CleanupClient()
    try:
        client.configure({"base_url": f"http://127.0.0.1:{redirect.server_port}/v1",
                          "api_key": "test-only-secret", "model": "m"})
        with pytest.raises(ValueError, match="HTTP 302"):
            client.test()
        assert seen == []
    finally:
        client.close()
        for server in (sink, redirect):
            server.shutdown()
            server.server_close()


def test_failing_callback_does_not_stop_the_worker():
    client = CleanupClient()
    results = []
    done = threading.Event()
    client.configure({"model": "test", "api_key": "test-only", "enabled": True})

    def callback(generation, revision, result):
        results.append((generation, result))
        if generation == 1:
            raise RuntimeError("consumer bug")
        done.set()

    try:
        with patch("quietbridge.cleanup.urlopen", side_effect=[Response("first."), Response("second.")]):
            assert client.submit(1, 1, ["first"], [], callback)
            deadline = threading.Event()
            for _ in range(200):
                if results:
                    break
                deadline.wait(0.01)
            assert client.submit(2, 2, ["second"], [], callback)
            assert done.wait(2)
        assert [generation for generation, _ in results] == [1, 2]
        assert results[1][1]["suggestion"] == "second."
    finally:
        client.close()


@pytest.mark.parametrize("output", ["Please\nwait.", "Please wait.", "Please \t wait."])
def test_accepted_suggestion_is_whitespace_normalized(output):
    client = CleanupClient()
    results = []
    done = threading.Event()
    client.configure({"model": "test", "api_key": "test-only", "enabled": True})
    try:
        with patch("quietbridge.cleanup.urlopen", return_value=Response(output)):
            assert client.submit(1, 1, ["Please wait"], [], lambda *args: (results.append(args), done.set()))
            assert done.wait(2)
        assert results[0][2]["status"] == "ready"
        assert results[0][2]["suggestion"] == "Please wait."
    finally:
        client.close()
