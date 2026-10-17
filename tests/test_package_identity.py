"""The package is published and imported as `sheath`; nothing answers to the old name."""

import importlib
import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = "tether"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_imports_as_sheath():
    mod = importlib.import_module("sheath")
    assert mod.__name__ == "sheath"
    assert hasattr(mod, "Sheath")
    assert hasattr(mod, "SheathConfig")


def test_old_import_name_is_gone():
    assert importlib.util.find_spec(OLD) is None


def test_distribution_name_and_scripts():
    project = _pyproject()["project"]
    assert project["name"] == "sheath"
    assert project["scripts"] == {
        "sheath": "sheath.cli:main",
        "sheath-prefetch-docling": "sheath.tools.documents:_prefetch_main",
        "sheath-build-sandbox": "sheath.container_runtime:_build_sandbox_main",
    }


def test_build_config_ships_the_renamed_package():
    hatch = _pyproject()["tool"]["hatch"]
    assert hatch["version"]["path"] == "sheath/__init__.py"
    assert hatch["build"]["targets"]["wheel"]["packages"] == ["sheath"]
    assert hatch["build"]["targets"]["sdist"]["only-include"] == ["sheath"]


def test_on_disk_workspace_dir_and_env_prefix_follow_the_name():
    from sheath.workspace import PROTECTED_DIRS

    assert ".sheath" in PROTECTED_DIRS
    assert f".{OLD}" not in PROTECTED_DIRS
    import sheath.config as config

    assert "SHEATH_" in Path(config.__file__).read_text()
