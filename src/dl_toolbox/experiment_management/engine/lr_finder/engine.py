"""Default layout engine whose fit is a Leslie Smith LR range test."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dl_toolbox.base.input_output.json_io import write_json
from dl_toolbox.dl_engine.lr_finder import (
    LrFinderConfig,
    LrFinderError,
    LrFinderRequest,
    LrFinderRuntime,
    LrFinderStep,
    RangeTestInput,
    find_lr,
    parse_lr_finder_spec,
)
from dl_toolbox.experiment_management.engine.core.engine import (
    RESOLVED_CONFIG_FILENAME,
    _is_distributed_worker,
)
from dl_toolbox.experiment_management.engine.core.session import RunSession
from dl_toolbox.experiment_management.engine.default.engine import DefaultExpEngine


class LrFinderExpEngine(DefaultExpEngine):
    def train(
        self, recipe: Path | Mapping[str, Any], *, force: bool = False
    ) -> object:
        if _is_distributed_worker():
            raise LrFinderError(
                "LrFinderExpEngine is single-process; got a distributed worker env"
            )
        return super().train(recipe, force=force)

    def _prepare_experiment_manager(self, toolbox: object) -> None:
        devices = toolbox.trainer_config().devices
        if devices != 1:
            raise LrFinderError(
                "LrFinderExpEngine requires trainer_config.devices == 1, "
                f"got {devices!r}"
            )
        super()._prepare_experiment_manager(toolbox)

    def _fit(self, session: RunSession) -> object:
        if session.fit_config.accumulate_grad_batches != 1:
            raise LrFinderError(
                "LrFinderExpEngine requires accumulate_grad_batches == 1, "
                f"got {session.fit_config.accumulate_grad_batches}"
            )
        experiment = session.experiment
        if not _is_distributed_worker():
            write_json(
                session.run_dir / RESOLVED_CONFIG_FILENAME,
                experiment.run_config,
            )
        return find_lr(_request_from_session(session))


def _request_from_session(session: RunSession) -> LrFinderRequest:
    experiment = session.experiment
    trainer = session.fit_config.trainer
    spec = parse_lr_finder_spec(experiment.run_config)
    return LrFinderRequest(
        inputs=RangeTestInput(
            model=experiment.model,
            optimizer=experiment.optimizer,
            train_loader=experiment.train_loader,
            step=LrFinderStep(
                prepare_tensors_fn=experiment.prepare_tensors_fn,
                forward_fn=experiment.forward_fn,
                loss_fn=experiment.loss_fn,
            ),
            config=LrFinderConfig(
                spec=spec,
                runtime=LrFinderRuntime(
                    accelerator=trainer.accelerator,
                    precision=trainer.precision,
                ),
            ),
        ),
        logger=session.logger,
        run_dir=session.run_dir,
        run_config=experiment.run_config,
    )
