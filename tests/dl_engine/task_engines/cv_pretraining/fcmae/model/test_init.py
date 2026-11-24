from __future__ import annotations

import pytest
import torch
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Model

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_init import (
    CONVNEXTV2_VARIANTS,
    INIT_IMAGENET,
    INIT_RANDOM,
    convnextv2_config,
    init_masked_convnextv2,
)

_ATTO_ENCODER_KEY = "encoder.stages.0.layers.0.dwconv.weight"


def test_tiny_random_matches_variant_table() -> None:
    spec = CONVNEXTV2_VARIANTS["tiny"]
    config = convnextv2_config("tiny", num_channels=1)
    assert tuple(config.depths) == spec.depths
    assert tuple(config.hidden_sizes) == spec.hidden_sizes
    assert config.num_channels == 1
    model = init_masked_convnextv2("tiny", init=INIT_RANDOM, num_channels=1)
    assert tuple(model.config.depths) == spec.depths
    assert tuple(model.config.hidden_sizes) == spec.hidden_sizes
    stem = model.embeddings.patch_embeddings
    assert tuple(stem.weight.shape) == (spec.hidden_sizes[0], 1, 4, 4)


def test_atto_imagenet_stem_and_encoder_copy() -> None:
    spec = CONVNEXTV2_VARIANTS["atto"]
    unmasked = ConvNextV2Model.from_pretrained(spec.imagenet_hub_id)
    expected_encoder = unmasked.state_dict()[_ATTO_ENCODER_KEY].detach().clone()
    masked = init_masked_convnextv2("atto", init=INIT_IMAGENET, num_channels=1)
    stem = masked.embeddings.patch_embeddings
    assert tuple(stem.weight.shape) == (40, 1, 4, 4)
    torch.testing.assert_close(
        masked.state_dict()[_ATTO_ENCODER_KEY],
        expected_encoder,
        atol=0.0,
        rtol=0.0,
    )


def test_unknown_variant_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ConvNeXt V2 variant"):
        convnextv2_config("not_a_variant")
