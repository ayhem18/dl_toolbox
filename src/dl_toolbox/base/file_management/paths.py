"""Path helpers used by loggers.

Callers pass absolute paths. This module does not use ``Path.cwd()`` or
``os.getcwd()``. Resolution is caller-owned, not process-cwd-owned.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

DEFAULT_ERROR_MESSAGE = "MAKE SURE THE passed path satisfies the condition passed with it"


def abs_path(path: Union[str, Path]) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        raise ValueError(f"path must be absolute (caller-owned), got {resolved}")
    return resolved


def process_path(
    save_path: Union[str, Path],
    *,
    dir_ok: bool = True,
    file_ok: bool = True,
    must_exist: bool = False,
    condition: Optional[Callable] = None,
    error_message: str = DEFAULT_ERROR_MESSAGE,
) -> Path:
    if save_path is None:
        raise ValueError("the save_path cannot be None")

    save_path = abs_path(save_path)

    if not save_path.exists():
        if must_exist:
            raise ValueError(
                "Passing a non-existing directory while setting the 'must_exist' argument to True"
            )

        if dir_ok and not file_ok:
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(
                "when passing a non-existing file, the parameters dir_ok and file_ok must be set to True and False respectively"
            )

    if (not file_ok and save_path.is_file()) or (not dir_ok and save_path.is_dir()):
        kind = "directory" if not dir_ok else "file"
        raise AssertionError(f"MAKE SURE NOT TO PASS A {kind}")

    if condition is not None:
        condition_output = condition(save_path)
        if not isinstance(condition_output, (tuple, bool)):
            raise TypeError(
                "The condition function must output either a tuple[bool, str] where 'str' can represent the error, or a 'bool' value. "
                f"Found: {condition_output}"
            )

        if isinstance(condition_output, Tuple):
            res, possible_error = condition_output
            if not (isinstance(res, bool) and isinstance(possible_error, str)):
                raise TypeError(
                    "if the condition function outputs a tuple. The first element must be a bool and the second a str. "
                    f"found: {condition_output}"
                )

            if not res:
                raise ValueError(
                    f"general error message: {error_message}\n specific error message: {possible_error}"
                )

        elif not condition_output:
            raise ValueError(error_message)

    return Path(save_path)
