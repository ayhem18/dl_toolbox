"""Masked ConvNeXt V2 encoder and the FCMAE decoder."""

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_decoder import (
    DECODER_DEPTH,
    DECODER_EMBED_DIM,
    FcmaeConvNextV2,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_init import (
    CONVNEXTV2_VARIANTS,
    INIT_IMAGENET,
    INIT_KINDS,
    INIT_RANDOM,
    ConvNextV2VariantSpec,
    convnextv2_config,
    init_masked_convnextv2,
    variant_spec,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_masked_encoder import (
    MaskedConvNextV2,
    downsample_mask,
    mae_patch_size,
)
from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_weights import (
    backbone_state_dict,
    inject_unmasked_weights,
)

__all__ = [
    "CONVNEXTV2_VARIANTS",
    "DECODER_DEPTH",
    "DECODER_EMBED_DIM",
    "FcmaeConvNextV2",
    "INIT_IMAGENET",
    "INIT_KINDS",
    "INIT_RANDOM",
    "ConvNextV2VariantSpec",
    "MaskedConvNextV2",
    "backbone_state_dict",
    "convnextv2_config",
    "downsample_mask",
    "init_masked_convnextv2",
    "inject_unmasked_weights",
    "mae_patch_size",
    "variant_spec",
]
