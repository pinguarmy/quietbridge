"""Explicit, on-device speech through a local, offline system voice.

Backends, chosen per platform from what is installed:

- macOS: ``say``.
- Windows: Windows PowerShell 5.1 (``powershell.exe``) with ``System.Speech``.
  PowerShell 7 (``pwsh``) is a last resort; it normally cannot load
  ``System.Speech`` and then reports an actionable error instead of failing silently.
- Linux: ``espeak-ng``, ``espeak``, then Speech Dispatcher's ``spd-say``.

The message is only ever written to the child's stdin. It is never placed in
argv or a shell string, so recognized text cannot become options or commands.

A speech process that exits with an error after it was started is reported via
``Speaker.last_error``. A clean exit is not proof that anything was audible.
"""
import base64
import os
import shutil
import subprocess
import sys
import tempfile
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
try { Add-Type -AssemblyName System.Speech } catch {
  [Console]::Error.WriteLine('QUIETBRIDGE_NO_SYSTEM_SPEECH: System.Speech cannot be loaded by this PowerShell.')
  exit 3
}
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $synth.SetOutputToDefaultAudioDevice(); $synth.Speak($text) } finally { $synth.Dispose() }
"""
# -EncodedCommand (UTF-16LE base64) avoids any command-line quoting of the script.
_POWERSHELL_ENCODED = base64.b64encode(_POWERSHELL_SCRIPT.encode("utf-16-le")).decode("ascii")
_NO_SYSTEM_SPEECH_EXIT = 3
_NO_SYSTEM_SPEECH_MARKER = "QUIETBRIDGE_NO_SYSTEM_SPEECH"
_NO_SYSTEM_SPEECH_MESSAGE = (
    "Speech did not play: this PowerShell cannot load System.Speech (PowerShell 7 does not include it). "
    "Windows PowerShell 5.1 (powershell.exe) is needed; it is part of Windows. Check that "
    "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe exists, then restart QuietBridge.")
# Only the end of a failing voice's error output is kept for the message.
_STDERR_TAIL_BYTES = 4096
_STDERR_DETAIL_CHARS = 200

_INSTALL_HINTS = {
    "darwin": "Native speech on macOS requires the built-in say command (/usr/bin/say)",
    "win32": ("Native speech on Windows requires Windows PowerShell 5.1 (powershell.exe) "
              "with System.Speech, which is part of Windows"),
    "linux": ("Native speech on Linux requires espeak-ng (for example: sudo apt install espeak-ng), "
              "espeak, or Speech Dispatcher's spd-say"),
}


def _platform_key(platform):
    if platform.startswith("linux") or platform.startswith("freebsd"):
        return "linux"
    return platform


def _windows_powershell_path():
    """Windows PowerShell 5.1's fixed location, used when it is not on PATH."""
    root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or "C:\\Windows"
    return "\\".join([root.rstrip("\\/"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"])


def _detect(platform, which):
    """Return ``(backend, executable)`` or ``(None, None)``."""
    key = _platform_key(platform)
    if key == "darwin":
        candidates = [("say", "say")]
    elif key == "win32":
        # Windows PowerShell 5.1 has System.Speech; pwsh normally does not, so
        # it is tried last and fails with an actionable message.
        candidates = [("powershell", "powershell.exe"), ("powershell", "powershell"),
                      ("powershell", _windows_powershell_path()), ("powershell", "pwsh")]
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


class _Utterance:
    """One started speech process and its bounded error output."""

    def __init__(self, process, backend, stderr_file):
        self.process = process
        self.backend = backend
        self.stderr_file = stderr_file
        self.stopped = False  # stopped on purpose: never an error
        self.dismissed = False
        self.error = None

    def close_stderr(self):
        stderr_file, self.stderr_file = self.stderr_file, None
        if stderr_file is not None:
            try:
                stderr_file.close()
            except (OSError, ValueError):
                pass

    def stderr_tail(self):
        if self.stderr_file is None:
            return ""
        try:
            self.stderr_file.flush()
            self.stderr_file.seek(0, os.SEEK_END)
            size = self.stderr_file.tell()
            self.stderr_file.seek(max(0, size - _STDERR_TAIL_BYTES))
            data = self.stderr_file.read(_STDERR_TAIL_BYTES)
        except (OSError, ValueError):
            return ""
        return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)


def _failure_message(backend, returncode, stderr):
    if backend == "powershell" and (returncode == _NO_SYSTEM_SPEECH_EXIT
                                    or _NO_SYSTEM_SPEECH_MARKER in stderr):
        return _NO_SYSTEM_SPEECH_MESSAGE
    status = (f"was stopped by signal {-returncode}" if returncode < 0
              else f"exited with status {returncode}")
    detail = " ".join(line.strip() for line in stderr.splitlines() if line.strip())
    if len(detail) > _STDERR_DETAIL_CHARS:
        detail = "…" + detail[-(_STDERR_DETAIL_CHARS - 1):]
    message = (f"Speech did not play: the {backend} voice {status}. "
               "Check that a speaker or headphones are connected and that the system voice works, "
               "then try Speak again.")
    return f"{message} Details: {detail}" if detail else message


class Speaker:
    def __init__(self, platform=None, which=None):
        self._platform = platform
        self._which = which
        self._detected = None
        self._process = None
        self._utterance = None
        self._lock = threading.Lock()
        # Short-held lock for the error record; never held while waiting on a process.
        self._error_lock = threading.Lock()

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

    @property
    def last_error(self) -> str:
        """Why the last started utterance failed, or "" (also while it runs).

        "" after a clean exit means only that the voice reported no error; it
        is not evidence that anything was heard.
        """
        with self._error_lock:
            utterance = self._utterance
            if utterance is None or utterance.stopped or utterance.dismissed:
                return ""
            if utterance.error is None:
                returncode = utterance.process.poll()
                if returncode is None or returncode == 0 or not isinstance(returncode, int):
                    return ""
                utterance.error = _failure_message(utterance.backend, returncode,
                                                   utterance.stderr_tail())
                utterance.close_stderr()
            return utterance.error

    def clear_error(self):
        """Dismiss a reported failure (for example when a new message starts)."""
        with self._error_lock:
            if self._utterance is not None:
                self._utterance.dismissed = True

    def _unavailable_message(self):
        platform = self._platform if self._platform is not None else sys.platform
        key = _platform_key(platform)
        return _INSTALL_HINTS.get(key, f"No local speech backend is supported on {platform}") + "."

    def speak(self, text: str):
        """Start speaking, replacing any current utterance. Returns once started."""
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
            # Error output goes to an anonymous temporary file, not a pipe: no
            # reader thread, and a voice (or a daemon it starts) can never block on it.
            try:
                stderr_file = tempfile.TemporaryFile()
            except OSError:
                stderr_file = None
            kwargs["stderr"] = stderr_file if stderr_file is not None else subprocess.DEVNULL
            try:
                process = subprocess.Popen(_command(backend, executable), **kwargs)
            except OSError as exc:
                if stderr_file is not None:
                    stderr_file.close()
                raise RuntimeError(f"Native speech could not start ({backend}): {exc}") from exc
            utterance = _Utterance(process, backend, stderr_file)
            with self._error_lock:
                previous, self._utterance = self._utterance, utterance
                if previous is not None:
                    previous.close_stderr()
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
        with self._error_lock:
            utterance = self._utterance
            if utterance is not None and utterance.process is process and utterance.error is None:
                # Marked before terminate(): a concurrent last_error must not
                # report our own stop as a failure.
                utterance.stopped = True
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
            with self._error_lock:
                if self._utterance is not None:
                    self._utterance.close_stderr()
