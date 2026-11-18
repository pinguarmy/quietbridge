"""Explicit, on-device speech through a local, offline system voice.

Backends, chosen per platform from what is installed:

- macOS: ``say``.
- Windows: Windows PowerShell (or ``pwsh``) with ``System.Speech``.
- Linux: ``espeak-ng``, ``espeak``, then Speech Dispatcher's ``spd-say``.

The message is only ever written to the child's stdin. It is never placed in
argv or a shell string, so recognized text cannot become options or commands.
"""
import base64
import shutil
import subprocess
import sys
import threading

MAX_CHARS = 2000

# Fixed script: the message arrives on stdin as UTF-8 and is never interpolated.
# Reading raw stdin bytes with an explicit UTF-8 decoder keeps non-ASCII text
# intact even if the console code page cannot be changed.
_POWERSHELL_SCRIPT = """\
$ErrorActionPreference = 'Stop'
try { [Console]::InputEncoding = New-Object System.Text.UTF8Encoding $false } catch {}
$reader = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), (New-Object System.Text.UTF8Encoding $false))
$text = $reader.ReadToEnd()
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $synth.SetOutputToDefaultAudioDevice(); $synth.Speak($text) } finally { $synth.Dispose() }
"""
# -EncodedCommand (UTF-16LE base64) avoids any command-line quoting of the script.
_POWERSHELL_ENCODED = base64.b64encode(_POWERSHELL_SCRIPT.encode("utf-16-le")).decode("ascii")

_INSTALL_HINTS = {
    "darwin": "Native speech on macOS requires the built-in say command (/usr/bin/say)",
    "win32": ("Native speech on Windows requires Windows PowerShell (powershell.exe) "
              "with System.Speech, or PowerShell 7 (pwsh)"),
    "linux": ("Native speech on Linux requires espeak-ng (for example: sudo apt install espeak-ng), "
              "espeak, or Speech Dispatcher's spd-say"),
}


def _platform_key(platform):
    if platform.startswith("linux") or platform.startswith("freebsd"):
        return "linux"
    return platform


def _detect(platform, which):
    """Return ``(backend, executable)`` or ``(None, None)``."""
    key = _platform_key(platform)
    if key == "darwin":
        candidates = [("say", "say")]
    elif key == "win32":
        candidates = [("powershell", "powershell.exe"), ("powershell", "powershell"), ("powershell", "pwsh")]
    else:
        candidates = [("espeak-ng", "espeak-ng"), ("espeak", "espeak"), ("spd-say", "spd-say")]
    for backend, name in candidates:
        path = which(name)
        if path:
            return backend, path
    return None, None


def _command(backend, executable):
    if backend == "say":
        return [executable]
    if backend == "powershell":
        return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", _POWERSHELL_ENCODED]
    if backend in ("espeak-ng", "espeak"):
        return [executable, "--stdin"]
    if backend == "spd-say":
        # -e/--pipe-mode reads stdin line by line; -w keeps the process alive until spoken.
        return [executable, "-e", "-w"]
    raise ValueError(f"Unknown speech backend {backend!r}")


def _payload(backend, text):
    if backend == "spd-say":
        # spd-say pipe mode treats a line starting with "!-!" as a raw SSIP command
        # (speechd src/clients/say/say.c). Send one line that cannot start with "!".
        # 2000 chars is at most 8000 UTF-8 bytes, below its 16384-byte line buffer.
        line = " ".join(text.split())
        if line.startswith("!"):
            line = " " + line
        return (line + "\n").encode("utf-8")
    return text.encode("utf-8")


class Speaker:
    def __init__(self, platform=None, which=None):
        self._platform = platform
        self._which = which
        self._detected = None
        self._process = None
        self._lock = threading.Lock()

    def _resolve(self):
        if self._detected is None:
            platform = self._platform if self._platform is not None else sys.platform
            which = self._which if self._which is not None else shutil.which
            self._detected = _detect(platform, which)
        return self._detected

    def refresh(self):
        """Forget the cached backend so the next query detects it again."""
        self._detected = None

    @property
    def available(self) -> bool:
        return self._resolve()[0] is not None

    @property
    def backend(self):
        return self._resolve()[0]

    @property
    def speaking(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def _unavailable_message(self):
        platform = self._platform if self._platform is not None else sys.platform
        key = _platform_key(platform)
        return _INSTALL_HINTS.get(key, f"No local speech backend is supported on {platform}") + "."

    def speak(self, text: str):
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
            raise ValueError("Choose a message of 1–2000 characters")
        backend, executable = self._resolve()
        if backend is None:
            raise RuntimeError(self._unavailable_message())
        kwargs = {"stdin": subprocess.PIPE}
        if backend == "spd-say":
            kwargs["stdout"] = subprocess.DEVNULL  # pipe mode echoes stdin to stdout
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        with self._lock:
            self._stop()
            try:
                process = subprocess.Popen(_command(backend, executable), **kwargs)
            except OSError as exc:
                raise RuntimeError(f"Native speech could not start ({backend}): {exc}") from exc
            self._process = process
            try:
                process.stdin.write(_payload(backend, text))
                process.stdin.close()
            except (BrokenPipeError, OSError) as exc:
                self._stop()
                raise RuntimeError(f"Native speech could not start ({backend})") from exc

    def _stop(self):
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            stdin = getattr(process, "stdin", None)
            if stdin is not None:
                try:
                    stdin.close()
                except (OSError, ValueError):
                    pass
            if self._process is process:
                self._process = None

    def stop(self):
        """Stop the current utterance, if any. Safe to call repeatedly from any thread."""
        with self._lock:
            self._stop()

    def close(self):
        with self._lock:
            self._stop()
