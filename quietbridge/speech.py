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

Speaking rate ("slow" | "normal" | "fast") and voice apply to the next
utterance. "normal" and the default voice ("") add no options, so the command
is exactly the system default. A voice id is used only after it was found in
the backend's own voice list, travels as a single argv element (never through
a shell) or, for PowerShell, in an environment variable read as data; the
PowerShell script itself is fixed. Voice lists come from ``say -v ?``,
``espeak-ng --voices`` and System.Speech ``GetInstalledVoices()``; they are
cached, time-bounded, and a failure yields an empty list. spd-say gets the
rate only: its ``-L`` listing prints fixed-width columns in which voice names
may contain spaces, so it cannot be parsed safely and the system voice is used.
"""
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

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
try {
  # Rate and voice arrive as data in environment variables, never in this script.
  $rate = $env:QUIETBRIDGE_SPEECH_RATE
  if ($rate) { $synth.Rate = [int]$rate }
  $voice = $env:QUIETBRIDGE_SPEECH_VOICE
  if ($voice) {
    try { $synth.SelectVoice($voice) } catch {
      [Console]::Error.WriteLine('QUIETBRIDGE_VOICE_UNAVAILABLE: the chosen voice cannot be used.')
      exit 4
    }
  }
  $synth.SetOutputToDefaultAudioDevice(); $synth.Speak($text)
} finally { $synth.Dispose() }
"""
_POWERSHELL_VOICES_SCRIPT = """\
$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch {}
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  foreach ($v in $synth.GetInstalledVoices()) {
    if ($v.Enabled) { [Console]::Out.WriteLine($v.VoiceInfo.Name + "`t" + $v.VoiceInfo.Culture.Name) }
  }
} finally { $synth.Dispose() }
"""
# -EncodedCommand (UTF-16LE base64) avoids any command-line quoting of the script.
_POWERSHELL_ENCODED = base64.b64encode(_POWERSHELL_SCRIPT.encode("utf-16-le")).decode("ascii")
_POWERSHELL_VOICES_ENCODED = base64.b64encode(_POWERSHELL_VOICES_SCRIPT.encode("utf-16-le")).decode("ascii")
_POWERSHELL_RATE_ENV = "QUIETBRIDGE_SPEECH_RATE"
_POWERSHELL_VOICE_ENV = "QUIETBRIDGE_SPEECH_VOICE"
_NO_SYSTEM_SPEECH_EXIT = 3
_VOICE_UNAVAILABLE_EXIT = 4
_VOICE_UNAVAILABLE_MARKER = "QUIETBRIDGE_VOICE_UNAVAILABLE"
_VOICE_UNAVAILABLE_MESSAGE = ("Speech did not play: the chosen voice cannot be used any more. "
                              "Choose another voice (or the system default) in Settings, then try Speak again.")

RATES = ("slow", "normal", "fast")
# Per backend: the option and its value for slow / fast. "normal" adds nothing.
# say and espeak take words per minute (their defaults are about 175-200 and 175),
# spd-say -100..100, System.Speech -10..10.
_RATE_ARGS = {
    "say": ("-r", {"slow": "140", "fast": "240"}),
    "espeak-ng": ("-s", {"slow": "130", "fast": "220"}),
    "espeak": ("-s", {"slow": "130", "fast": "220"}),
    "spd-say": ("-r", {"slow": "-40", "fast": "40"}),
}
_POWERSHELL_RATES = {"slow": "-3", "fast": "3"}
VOICE_LIST_TIMEOUT_S = 5.0
_VOICE_RETRY_AFTER_FAILURE_S = 60.0
_VOICE_OUTPUT_LIMIT = 256 * 1024
_MAX_VOICES = 500
_VOICE_ID_MAX = 128
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


def safe_voice_id(voice):
    """A voice id that can only ever be one plain argv element or environment value."""
    return (isinstance(voice, str) and 0 < len(voice) <= _VOICE_ID_MAX and voice == voice.strip()
            and not voice.startswith("-") and voice.isprintable())


def _command(backend, executable, rate="normal", voice=""):
    """argv for one utterance. Text never appears here; voice only if safe_voice_id()."""
    if rate not in RATES:
        raise ValueError(f"Unknown speaking rate {rate!r}")
    if voice and (backend == "spd-say" or not safe_voice_id(voice)):
        raise ValueError("This voice cannot be used here")
    options = []
    if backend in _RATE_ARGS and rate != "normal":
        flag, values = _RATE_ARGS[backend]
        options += [flag, values[rate]]
    if voice and backend != "powershell":
        options += ["-v", voice]
    if backend == "say":
        return [executable, *options]
    if backend == "powershell":
        # Rate and voice go in the environment (_environment); the script is fixed.
        return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", _POWERSHELL_ENCODED]
    if backend in ("espeak-ng", "espeak"):
        return [executable, *options, "--stdin"]
    if backend == "spd-say":
        # -e/--pipe-mode reads stdin line by line; -w keeps the process alive until spoken.
        return [executable, *options, "-e", "-w"]
    raise ValueError(f"Unknown speech backend {backend!r}")


def _environment(backend, rate="normal", voice=""):
    """The child's environment for PowerShell (rate/voice as data); None = inherit unchanged."""
    if backend != "powershell":
        return None
    if voice and not safe_voice_id(voice):
        raise ValueError("This voice cannot be used here")
    env = {k: v for k, v in os.environ.items() if k.upper() not in (_POWERSHELL_RATE_ENV, _POWERSHELL_VOICE_ENV)}
    if rate != "normal":
        env[_POWERSHELL_RATE_ENV] = _POWERSHELL_RATES[rate]
    if voice:
        env[_POWERSHELL_VOICE_ENV] = voice
    return env


def _voices_command(backend, executable):
    """argv that lists installed voices, or None when the backend has no safe listing."""
    if backend == "say":
        return [executable, "-v", "?"]
    if backend in ("espeak-ng", "espeak"):
        return [executable, "--voices"]
    if backend == "powershell":
        return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                _POWERSHELL_VOICES_ENCODED]
    return None  # spd-say: see the module docstring


# macOS: "Alex                en_US    # Most people recognize me by my voice."
# Names may contain spaces and parentheses ("Eddy (English (US))").
_SAY_LINE = re.compile(r"^(?P<name>\S.*?)\s+(?P<lang>[a-z]{2,3}(?:[_-][A-Za-z0-9]+)*)\s+#")


def _parse_voices(backend, output):
    voices = []
    seen = set()

    def add(voice_id, label):
        if safe_voice_id(voice_id) and voice_id not in seen and len(voices) < _MAX_VOICES:
            seen.add(voice_id)
            voices.append({"id": voice_id, "label": label[:200]})

    for line in output.splitlines():
        if backend == "say":
            match = _SAY_LINE.match(line)
            if match:
                add(match.group("name"), f"{match.group('name')} ({match.group('lang')})")
        elif backend in ("espeak-ng", "espeak"):
            # "Pty Language Age/Gender VoiceName File Other Languages"; -v takes the language.
            fields = line.split()
            if len(fields) >= 4 and fields[0].isdigit():
                add(fields[1], f"{fields[3].replace('_', ' ')} ({fields[1]})")
        elif backend == "powershell":
            name, _, culture = line.rstrip("\r").partition("\t")
            if name:
                add(name, f"{name} ({culture})" if culture else name)
    return voices


def list_voices(backend, executable, run=None, timeout=VOICE_LIST_TIMEOUT_S):
    """Installed voices as [{"id", "label"}]; [] on any failure (never raises)."""
    command = _voices_command(backend, executable) if backend else None
    if command is None:
        return []
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL,
              "timeout": timeout}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        result = (run or subprocess.run)(command, **kwargs)
        if getattr(result, "returncode", 1) != 0:
            return []
        output = result.stdout or b""
        if isinstance(output, bytes):
            output = output[:_VOICE_OUTPUT_LIMIT].decode("utf-8", errors="replace")
        return _parse_voices(backend, str(output)[:_VOICE_OUTPUT_LIMIT])
    except Exception:  # timeout, missing executable, anything: an empty list
        return []


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
    if backend == "powershell" and (returncode == _VOICE_UNAVAILABLE_EXIT
                                    or _VOICE_UNAVAILABLE_MARKER in stderr):
        return _VOICE_UNAVAILABLE_MESSAGE
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
    def __init__(self, platform=None, which=None, run=None):
        self._platform = platform
        self._which = which
        self._run = run  # subprocess.run replacement for the voice listing (tests)
        self._detected = None
        self._process = None
        self._utterance = None
        self._lock = threading.Lock()
        # Short-held lock for the error record; never held while waiting on a process.
        self._error_lock = threading.Lock()
        self._rate = "normal"
        self._voice = ""  # preferred voice; used only once found in the voice list
        self._settings_lock = threading.Lock()
        self._voices = None  # cached list, or None before the first listing finished
        self._voices_failed_at = None
        self._voices_thread = None

    def _resolve(self):
        if self._detected is None:
            platform = self._platform if self._platform is not None else sys.platform
            which = self._which if self._which is not None else shutil.which
            self._detected = _detect(platform, which)
        return self._detected

    def refresh(self):
        """Forget the cached backend (and its voice list) so the next query detects it again."""
        self._detected = None
        with self._settings_lock:
            self._voices = None
            self._voices_failed_at = None

    # ----------------------------------------------------------- rate and voice

    def _load_voices(self):
        backend, executable = self._resolve()
        voices = list_voices(backend, executable, run=self._run)
        with self._settings_lock:
            self._voices = voices
            self._voices_failed_at = time.monotonic() if not voices else None

    def voices(self, wait=False, timeout=VOICE_LIST_TIMEOUT_S + 1):
        """Installed voices [{"id", "label"}]; never raises.

        Without ``wait`` this never blocks: the first call starts a background
        listing and returns [] until it finished. An empty result is retried
        after a minute (a voice service may start later).
        """
        with self._settings_lock:
            voices = self._voices
            stale = voices is not None and not voices and self._voices_failed_at is not None and \
                time.monotonic() - self._voices_failed_at > _VOICE_RETRY_AFTER_FAILURE_S
            if voices is not None and not stale:
                return [dict(v) for v in voices]
            thread = self._voices_thread
            if thread is None or not thread.is_alive():
                thread = threading.Thread(target=self._load_voices, name="quietbridge-voice-list", daemon=True)
                self._voices_thread = thread
                thread.start()
        if wait:
            thread.join(timeout)
            with self._settings_lock:
                return [dict(v) for v in (self._voices or [])]
        return [dict(v) for v in (voices or [])]

    def _effective_voice(self, wait=False):
        voice = self._voice
        if not voice:
            return ""
        return voice if any(v["id"] == voice for v in self.voices(wait=wait)) else ""

    def settings(self):
        """{"rate", "voice"}: the voice is "" (system default) until it is confirmed installed."""
        return {"rate": self._rate, "voice": self._effective_voice(wait=False)}

    def set_rate(self, rate):
        if rate not in RATES:
            raise ValueError("Speaking rate must be slow, normal or fast")
        self._rate = rate

    def set_voice(self, voice, strict=True):
        """Choose a voice for the next utterance ("" = system default).

        ``strict`` (a person's choice): it must be in the installed list now.
        Otherwise (a saved preference) it is kept and used only once listed.
        """
        if not isinstance(voice, str):
            raise ValueError("Voice must be text")
        if voice and strict and not any(v["id"] == voice for v in self.voices(wait=True)):
            raise ValueError("That voice is not installed on this computer; choose one from the list")
        if voice and not safe_voice_id(voice):
            if strict:
                raise ValueError("That voice is not installed on this computer; choose one from the list")
            voice = ""
        self._voice = voice

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
        rate = self._rate
        # A saved voice is used only once the backend lists it (bounded wait on first use).
        voice = self._effective_voice(wait=bool(self._voice))
        command = _command(backend, executable, rate, voice)
        kwargs = {"stdin": subprocess.PIPE}
        env = _environment(backend, rate, voice)
        if env is not None:
            kwargs["env"] = env
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
                process = subprocess.Popen(command, **kwargs)
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
