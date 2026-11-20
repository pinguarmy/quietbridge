"""Where each person's data lives. Shared model weights live in models_dir() (the checkout's
models/ when running from source); everything
learned from *you* (clips, phrases, personal models, settings) goes here.

Override with LIPFLOW_HOME (tests use a temp dir so they never touch your real data)."""
import os
import sys

WINDOWS = sys.platform == "win32"

_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_source_checkout(parent):
    """True when ``parent`` is a source checkout holding the lipflow package.

    A checkout has pyproject.toml beside the package, or a models/ folder there
    already. An installed wheel's parent is site-packages: a stray models/ there
    (another distribution's package) must never be mistaken for model weights.
    """
    if os.path.isfile(os.path.join(parent, "pyproject.toml")):
        return True
    if os.path.basename(os.path.normpath(parent)).lower() in ("site-packages", "dist-packages"):
        return False
    return os.path.isdir(os.path.join(parent, "models"))


def models_dir(environ=None, package_parent=None, platform=None):
    """The one folder for the shared, downloaded stock weights (lip reading, face landmarker).

    1. ``LIPFLOW_MODELS`` (``~`` expanded) when set and non-empty.
    2. ``models/`` in the source checkout when running from one (unchanged for every
       existing checkout, setup.sh and setup.ps1).
    3. Otherwise (an installed wheel) ``stock-models`` in the per-user data directory
       (LIPFLOW_HOME, else the platform default). It is a sibling of the personal
       ``models`` folder, so stock and personal weights never mix.

    lipflow.vsr, lipflow.face, ``lipflow doctor`` and quietbridge.assets all use this.
    """
    environ = os.environ if environ is None else environ
    override = (environ.get("LIPFLOW_MODELS") or "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    parent = _PACKAGE_PARENT if package_parent is None else package_parent
    if _is_source_checkout(parent):
        return os.path.join(parent, "models")
    home = environ.get("LIPFLOW_HOME") or default_home(platform, environ)
    return os.path.join(home, "stock-models")



def default_home(platform=None, environ=None):
    r"""The per-user data directory for this platform (before any LIPFLOW_HOME override).

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
        base = os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "lipflow")


# Shared, downloaded weights (setup.sh / setup.ps1 / the QuietBridge page). Defined here, not only
# in vsr.py, so `lipflow doctor` can check them without importing torch first.
MODELS = models_dir()

_DEFAULT_HOME = default_home()
HOME = os.environ.get("LIPFLOW_HOME") or _DEFAULT_HOME
PERSONAL_MODELS = os.path.join(HOME, "models")
PERSONAL_VSR = os.path.join(PERSONAL_MODELS, "vsr_face.pth")
PERSONAL_LM = os.path.join(PERSONAL_MODELS, "lm_phrasing.pth")

# The app bundle's launcher sets LIPFLOW_APP=1: permissions then belong to "Lipflow", not the terminal.
# On Windows nothing is granted per app, so the name only shows up in messages.
WHO = "Lipflow" if os.environ.get("LIPFLOW_APP") or WINDOWS else "your terminal"
