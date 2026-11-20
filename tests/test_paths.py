"""Where Lipflow keeps per-user data on each platform."""
import os

from lipflow import paths


def test_linux_uses_the_xdg_data_directory():
    home = os.path.expanduser("~")
    assert paths.default_home("linux", {}) == os.path.join(home, ".local", "share", "lipflow")
    assert paths.default_home("linux", {"XDG_DATA_HOME": "/data/x"}) == os.path.join("/data/x", "lipflow")


def test_relative_xdg_data_home_is_ignored_as_the_spec_requires():
    home = os.path.expanduser("~")
    assert paths.default_home("linux", {"XDG_DATA_HOME": "relative/dir"}) == os.path.join(home, ".local", "share", "lipflow")


def test_macos_and_windows_locations_are_unchanged():
    assert paths.default_home("darwin", {}) == os.path.expanduser("~/Library/Application Support/Lipflow")
    assert paths.default_home("win32", {"APPDATA": "C:/Users/a/AppData/Roaming"}) == os.path.join(
        "C:/Users/a/AppData/Roaming", "Lipflow")


def test_linux_no_longer_uses_the_macos_library_path():
    assert "Library" not in paths.default_home("linux", {})
