"""Smoke tests for project scaffolding."""

from pathlib import Path

from seed_preview_cv import __version__
from seed_preview_cv.common.config import load_yaml, resolve_path
from seed_preview_cv.common.paths import CONFIGS_DIR, PROJECT_ROOT


def test_version():
    assert __version__ == "0.1.0"


def test_project_root_exists():
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_load_seed_selection_config():
    config = load_yaml(CONFIGS_DIR / "seed_selection.yaml")
    assert config["minecraft_version"] == "1.16.1"
    assert config["sampling"]["num_seeds"] == 100000


def test_resolve_path_relative_to_project_root():
    path = resolve_path("data/interim")
    assert path == (PROJECT_ROOT / "data/interim").resolve()
    assert path.is_dir()


def test_resolve_path_absolute():
    abs_path = Path("/tmp/test")
    assert resolve_path(abs_path) == abs_path
