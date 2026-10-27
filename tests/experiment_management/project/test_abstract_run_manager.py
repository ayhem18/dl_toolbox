from abc import ABC
import inspect

from dl_toolbox.experiment_management.project.abstract_run_manager import AbstractRunManager


def test_run_manager_is_abstract() -> None:
    assert issubclass(AbstractRunManager, ABC)
    assert inspect.isabstract(AbstractRunManager)
