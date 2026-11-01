"""Render native onboarding views without a camera, hotkeys, or activating a window.

Usage: LIPFLOW_HOME=/tmp/lipflow-ui-check .venv/bin/python scripts/render_onboarding.py <output_png> [page_name]
Pages supported: welcome, permissions, words, practice, train
"""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
from Foundation import NSDate, NSRunLoop

from lipflow import camera, onboarding
from lipflow.cleanup import Cleaner
from lipflow.onboarding import Onboarding


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Render native onboarding views without a camera, hotkeys, or activating a window."
    )
    parser.add_argument("output_png", help="Path to output PNG image")
    parser.add_argument(
        "page_name",
        nargs="?",
        default="welcome",
        choices=["welcome", "permissions", "words", "practice", "train"],
        help="Page to render: welcome, permissions, words, practice, train (default: welcome)",
    )
    args = parser.parse_args()
    output_path = args.output_png
    page_name = args.page_name

    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    camera.list_cameras = lambda: []

    # Mock HUD
    class DummyHUD:
        def hide(self):
            pass
        def show(self, *args, **kwargs):
            pass

    # Mock Camera
    class DummyCamera:
        def __init__(self):
            self.track_always = False
        def ensure_open(self):
            pass

    # Mock Cleaner / personal
    cleaner = Cleaner("basic", style="polish")
    app = SimpleNamespace(
        hud=DummyHUD(),
        camera=DummyCamera(),
        cleaner=cleaner,
        settings={"cleanup_backend": "gemini", "cleanup_style": "polish"},
        opts=SimpleNamespace(key="right_option", backend="gemini", style="polish"),
        onboarding=None,
        onboarding_text="",
        jobs=SimpleNamespace(put=lambda job: None),
    )

    ob = Onboarding.alloc().initWithApp_(app)

    if page_name == "welcome":
        ob.welcome()
    elif page_name == "permissions":
        ob.goPermissions_(None)
    elif page_name == "words":
        ob.goWords_(None)
    elif page_name == "practice":
        ob.goPractice_(None)
    elif page_name == "train":
        ob.goTrain_(None)
        ob.report(42.0, "正在训练面部口型模型 (42%)...")
    else:
        ob.welcome()

    ob.win.orderFront_(None)
    ob.win.displayIfNeeded()

    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.4))

    subprocess.run(
        ["/usr/sbin/screencapture", "-x", "-o", "-l", str(ob.win.windowNumber()), output_path],
        check=True,
    )
    ob.win.orderOut_(None)
    print(f"Rendered native onboarding ({page_name}): {output_path}")


if __name__ == "__main__":
    main()
