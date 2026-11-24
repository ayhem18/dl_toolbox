"""MAE patch masks at image resolution."""

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.mask.fcmae_mask_sampling import (
    random_patch_mask,
)

__all__ = ["random_patch_mask"]
