from abc import ABC
import inspect

from dl_toolbox.run_manager import RunManager


def test_run_manager_is_abstract() -> None:
    assert issubclass(RunManager, ABC)
    assert inspect.isabstract(RunManager)
