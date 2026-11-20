"""One models-directory rule for lipflow and quietbridge: env override, source checkout, per-user."""
import json
import os
import subprocess
import sys

from lipflow import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_env_override_wins_and_expands_user(tmp_path):
    env = {"LIPFLOW_MODELS": "~/my-models", "LIPFLOW_HOME": str(tmp_path)}
    assert paths.models_dir(env, package_parent=ROOT) == os.path.join(os.path.expanduser("~"), "my-models")
    assert paths.models_dir({"LIPFLOW_MODELS": "  "}, package_parent=ROOT) == os.path.join(ROOT, "models")


def test_source_checkout_uses_its_models_folder(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\n")
    assert paths.models_dir({}, package_parent=str(checkout)) == str(checkout / "models")
    # An existing models/ alone also marks a checkout (an older copy without pyproject.toml).
    older = tmp_path / "older"
    (older / "models").mkdir(parents=True)
    assert paths.models_dir({}, package_parent=str(older)) == str(older / "models")
    # This repository: unchanged for every existing checkout.
    assert paths.models_dir({}) == os.path.join(ROOT, "models")


def test_installed_wheel_uses_the_per_user_data_directory(tmp_path):
    site = tmp_path / "lib" / "site-packages"
    (site / "models").mkdir(parents=True)  # another distribution's package, not model weights
    assert paths.models_dir({"LIPFLOW_HOME": str(tmp_path / "home")}, package_parent=str(site)) == \
        str(tmp_path / "home" / "stock-models")
    expected = os.path.join(paths.default_home("linux", {"XDG_DATA_HOME": "/data/x"}), "stock-models")
    assert paths.models_dir({"XDG_DATA_HOME": "/data/x"}, package_parent=str(site), platform="linux") == expected
    assert expected == os.path.join("/data/x", "lipflow", "stock-models")
    # Never the personal-models folder (vsr_face.pth, lm_phrasing.pth live there).
    assert not expected.endswith(os.path.join("lipflow", "models"))


def test_every_consumer_agrees_on_one_folder(tmp_path):
    """doctor, the lip reader, the face tracker and the downloader read the same folder."""
    code = (
        "import json, os\n"
        "from lipflow import paths, doctor, face, vsr\n"
        "from quietbridge import assets\n"
        "print(json.dumps({'paths': paths.MODELS, 'doctor': doctor.MODELS, 'vsr': vsr.MODELS,\n"
        "  'face': os.path.dirname(face.DEFAULT_MODEL),\n"
        "  'assets': str(assets.LIPS_ASSETS[0].path().parent.parent)}))\n"
    )
    env = {**os.environ, "LIPFLOW_MODELS": str(tmp_path / "shared")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=300, check=True).stdout.strip().splitlines()[-1]
    folders = json.loads(out)
    assert set(folders.values()) == {str(tmp_path / "shared")}, folders
