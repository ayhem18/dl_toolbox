"""Shared deep learning utilities.

Importable code belongs here only after a real project has used it and the
tests that justified the move live in this repository.

Heavy imports live in submodules so `import dl_toolbox` does not pull torch
or opencv:

- `dl_toolbox.seed.seed_everything`
- `dl_toolbox.loggers.base_logger.Logger`
- `dl_toolbox.loggers.factory.get_logger`
- `dl_toolbox.data_contract.hash_payload`
- `dl_toolbox.parsing.parse_numerical.parse_int`
- `dl_toolbox.geometry.contours.contours_bbox`
- `dl_toolbox.geometry.masks.rasterize_contours`
- `dl_toolbox.geometry.bboxes.generate_tile_boxes_xyxy`
- `dl_toolbox.dl_engine.training.fit.fit_classification`
- `dl_toolbox.dl_engine.training.config.FitConfig`
- `dl_toolbox.dl_engine.metrics.classification.classification.binary_classification_accuracy_metric`
"""

__version__ = "0.1.0"
