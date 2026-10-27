"""Default train entry. Unpacks ExpSignatureObj and calls fit."""

from __future__ import annotations

import lightning as L

from dl_toolbox.dl_engine.training.fit import fit_classification
from dl_toolbox.experiment_management.signature.exp_signature_obj import ExpSignatureObj


def run_experiment(signature: ExpSignatureObj) -> L.Trainer:
    experiment = signature.experiment
    return fit_classification(
        experiment.model,
        experiment.train_loader,
        experiment.val_loader,
        experiment.optimizer,
        prepare_tensors_fn=experiment.prepare_tensors_fn,
        forward_fn=experiment.forward_fn,
        loss_fn=experiment.loss_fn,
        build_metric_inputs_fn=experiment.build_metric_inputs_fn,
        train_metrics=experiment.train_metrics,
        val_metrics=experiment.val_metrics,
        config=signature.fit_config,
        checkpoint_manager=signature.checkpoint_manager,
        logger=signature.logger,
    )
