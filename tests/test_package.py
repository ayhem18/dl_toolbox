import subprocess
import sys

import dl_toolbox


def test_package_imports() -> None:
    assert dl_toolbox.__version__ == "0.1.0"


def test_parsing_is_importable_without_geometry() -> None:
    from dl_toolbox.parsing.parse_iterable import parse_mapping
    from dl_toolbox.parsing.parse_numerical import parse_int
    from dl_toolbox.parsing.parse_str import parse_non_empty_str
    from dl_toolbox.parsing.parse_utils import require_key

    assert parse_int(3, label="n") == 3
    assert parse_non_empty_str(" a ", label="s") == "a"
    assert parse_mapping({"k": 1}, label="row") == {"k": 1}
    assert require_key({"k": 1}, "k", label="row") == 1


def test_root_import_does_not_pull_torch_or_cv2() -> None:
    script = (
        "import dl_toolbox, sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'cv2' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", script])
