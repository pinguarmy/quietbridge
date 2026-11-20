"""Where each person's data lives. Shared model weights stay in the repo's models/; everything
learned from *you* (clips, phrases, personal models, settings) goes here.

Override with LIPFLOW_HOME (tests use a temp dir so they never touch your real data)."""
import os
import sys

WINDOWS = sys.platform == "win32"

# Shared, downloaded weights (setup.sh / setup.ps1): the repository's models/ folder. Defined here,
# not only in vsr.py, so `lipflow doctor` can check them without importing torch first.
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")



def default_home(platform=None, environ=None):
    """The per-user data directory for this platform (before any LIPFLOW_HOME override).

    Windows: %APPDATA%\Lipflow. macOS: ~/Library/Application Support/Lipflow.
    Elsewhere (Linux): $XDG_DATA_HOME/lipflow, by default ~/.local/share/lipflow.
    """
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    if platform == "win32":
        return os.path.join(environ.get("APPDATA") or os.path.expanduser("~"), "Lipflow")
    if platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/Lipflow")
    base = environ.get("XDG_DATA_HOME") or ""
    if not os.path.isabs(base):  # the XDG spec ignores a relative XDG_DATA_HOME
        base = os.path.expanduser("~/.local/share")
    return os.path.join(base, "lipflow")


_DEFAULT_HOME = default_home()
HOME = os.environ.get("LIPFLOW_HOME") or _DEFAULT_HOME
PERSONAL_MODELS = os.path.join(HOME, "models")
PERSONAL_VSR = os.path.join(PERSONAL_MODELS, "vsr_face.pth")
PERSONAL_LM = os.path.join(PERSONAL_MODELS, "lm_phrasing.pth")

# The app bundle's launcher sets LIPFLOW_APP=1: permissions then belong to "Lipflow", not the terminal.
# On Windows nothing is granted per app, so the name only shows up in messages.
WHO = "Lipflow" if os.environ.get("LIPFLOW_APP") or WINDOWS else "your terminal"
