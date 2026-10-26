from __future__ import annotations

import random
from pathlib import Path

from dl_toolbox.cli.main import main
from dl_toolbox.project.layout import PROJECT_MARKER_FILENAME

from _support import N_TRIALS, random_package_name, random_project_name


def test_cli_setup_project_uses_cwd(
    tmp_path: Path, rng: random.Random, monkeypatch, capsys
) -> None:
    for i in range(N_TRIALS):
        cwd = tmp_path / f"cwd_{i}"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        name = random_project_name(rng)
        code = main(["setup-project", name])
        assert code == 0
        captured = capsys.readouterr()
        root = cwd / name
        assert str(root.resolve()) in captured.out
        assert (root / PROJECT_MARKER_FILENAME).is_file()
        again = main(["setup-project", name])
        assert again == 1
        err = capsys.readouterr().err
        assert "already exists" in err


def test_cli_setup_project_with_package(
    tmp_path: Path, rng: random.Random, monkeypatch, capsys
) -> None:
    cwd = tmp_path / "pkg_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    name = random_project_name(rng)
    package = random_package_name(rng)
    code = main(["setup-project", name, "--package", package])
    assert code == 0
    capsys.readouterr()
    root = cwd / name
    assert not (root / "pyproject.toml").exists()
    assert (root / package / "pyproject.toml").is_file()
    src_dir = root / package / "src" / package
    assert sorted(path.name for path in src_dir.iterdir()) == ["__init__.py"]
