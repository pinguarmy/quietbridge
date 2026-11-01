"""Explicit, on-device speech using the native macOS voice."""
import shutil
import subprocess
import threading


class Speaker:
    def __init__(self):
        self._process = None
        self._lock = threading.Lock()

    @property
    def available(self):
        return shutil.which("say") is not None

    def speak(self, text: str):
        if not text.strip() or len(text) > 2000:
            raise ValueError("Choose a message of 1–2000 characters")
        if not self.available:
            raise RuntimeError("Native speech requires macOS and the say command")
        with self._lock:
            self._stop()
            # stdin keeps recognized text out of command-line options and shell interpretation.
            self._process = subprocess.Popen(["say"], stdin=subprocess.PIPE)
            try:
                self._process.stdin.write(text.encode("utf-8"))
                self._process.stdin.close()
            except BrokenPipeError:
                raise RuntimeError("Native speech could not start")

    def _stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1)

    def close(self):
        with self._lock:
            self._stop()
