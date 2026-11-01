"""Render the native settings view without a camera, hotkeys, or activating a window.

Usage: LIPFLOW_HOME=/tmp/lipflow-ui-check .venv/bin/python scripts/render_settings.py /tmp/settings.png
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from AppKit import NSApplication

from lipflow import camera, onboarding
from lipflow.cleanup import Cleaner
from lipflow.settings_window import Settings


def main():
    from AppKit import NSApplicationActivationPolicyAccessory
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    camera.list_cameras = lambda: []
    onboarding.saved_clips = lambda: []
    app = SimpleNamespace(
        settings={"cleanup_backend": "gemini", "cleanup_style": "polish"},
        opts=SimpleNamespace(key="right_option", backend="gemini", style="polish"),
        cleaner=Cleaner("basic", style="polish"),
    )
    settings = Settings.alloc().initWithApp_(app)
    if len(sys.argv) > 2:
        settings.build(sys.argv[2])
    else:
        settings.build()
    settings.win.orderFront_(None)  # accessory window, never activates the user's app
    settings.win.displayIfNeeded()
    from Foundation import NSDate, NSRunLoop
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.3))
    import subprocess
    subprocess.run(['/usr/sbin/screencapture', '-x', '-o', '-l', str(settings.win.windowNumber()), sys.argv[1]], check=True)
    settings.win.orderOut_(None)
    print(f"Rendered native settings: {sys.argv[1]}")


if __name__ == "__main__":
    main()
