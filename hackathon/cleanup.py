"""Optional, in-memory OpenAI-compatible cleanup with one bounded worker."""
from __future__ import annotations

import json
import queue
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

MAX_RESPONSE = 512 * 1024
TEST_PROMPT = "Please return exactly OK."


def normalize_base_url(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A cleanup base URL is required")
    value = value.strip()
    try:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("Cleanup base URL cannot contain credentials, query, or fragment")
        if parsed.scheme.lower() not in {"https", "http"} or not parsed.hostname:
            raise ValueError("Cleanup base URL must use HTTPS")
        if parsed.scheme.lower() == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("HTTP cleanup is allowed only for localhost or 127.0.0.1")
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            path = path[:-len("/chat/completions")]
        if not path.endswith("/v1"):
            path += "/v1"
        return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))
    except ValueError as exc:
        if str(exc).startswith("Cleanup") or str(exc) == "A cleanup base URL is required":
            raise
        raise ValueError("Invalid cleanup base URL") from None


class CleanupClient:
    def __init__(self):
        self._lock = threading.RLock()
        self._config = {"base_url": "https://api.openai.com/v1", "model": "", "enabled": False}
        self._api_key = ""
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="demo-cleanup", daemon=True)
        self._worker.start()

    def public_config(self):
        with self._lock:
            return {**self._config, "configured": bool(self._api_key)}

    def configure(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        with self._lock:
            enabled = data.get("enabled", self._config["enabled"])
            model = data.get("model", self._config["model"])
            if not isinstance(enabled, bool) or not isinstance(model, str):
                raise ValueError("Cleanup enabled must be boolean and model must be text")
            model = model.strip()
            base_url = normalize_base_url(data.get("base_url", self._config["base_url"]))
            api_key = data.get("api_key", self._api_key)
            if not isinstance(api_key, str):
                raise ValueError("Cleanup API key must be text")
            if not api_key:
                api_key = self._api_key
            if enabled and not api_key:
                raise ValueError("Add an API key before enabling cleanup")
            if enabled and not model:
                raise ValueError("Choose a model before enabling cleanup")
            self._config = {"base_url": base_url, "model": model, "enabled": enabled}
            self._api_key = api_key
            return self.public_config()

    def submit(self, generation, revision, candidates, labels, callback):
        with self._lock:
            if not self._config["enabled"] or self._stop.is_set():
                return False
            config = {**self._config, "api_key": self._api_key}
        job = (generation, revision, list(candidates), list(labels), config, callback)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                return False
        return True

    def test(self):
        with self._lock:
            config = {**self._config, "api_key": self._api_key}
        if not config["api_key"] or not config["model"]:
            raise ValueError("Configure an API key and model before testing cleanup")
        if self._chat(config, TEST_PROMPT).strip() != "OK":
            raise ValueError("Cleanup test returned an unexpected response")
        return {"ok": True}

    def _chat(self, config, prompt):
        messages = [{"role": "user", "content": prompt}]
        if prompt != TEST_PROMPT:
            messages.insert(0, {"role": "system", "content":
                "Clean up the best candidate without changing its meaning. Return only the cleaned text."})
        body = json.dumps({"model": config["model"], "messages": messages, "stream": False}).encode()
        request = Request(config["base_url"] + "/chat/completions", data=body,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + config["api_key"]})
        try:
            with urlopen(request, timeout=6) as response:
                payload = response.read(MAX_RESPONSE + 1)
        except HTTPError as exc:
            raise ValueError(f"Cleanup service returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise ValueError("Cleanup service could not be reached") from None
        if len(payload) > MAX_RESPONSE:
            raise ValueError("Cleanup response exceeded 512 KiB")
        try:
            content = json.loads(payload)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ValueError("Cleanup service returned an invalid response") from None
        if not isinstance(content, str):
            raise ValueError("Cleanup service returned an invalid response")
        return content

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                return
            generation, revision, candidates, labels, config, callback = job
            started = time.perf_counter()
            try:
                prompt = json.dumps({"candidates": [text[:2000] for text in candidates[:10]],
                                     "gesture_labels": labels[:10]}, ensure_ascii=False)
                suggestion = self._chat(config, prompt).strip()
                result = {"status": "ready" if suggestion else "idle", "suggestion": suggestion, "error": ""}
            except Exception as exc:
                result = {"status": "error", "suggestion": "", "error": str(exc) if isinstance(exc, ValueError) else "Cleanup failed"}
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            if not self._stop.is_set():
                callback(generation, revision, result)

    def close(self):
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=7)
