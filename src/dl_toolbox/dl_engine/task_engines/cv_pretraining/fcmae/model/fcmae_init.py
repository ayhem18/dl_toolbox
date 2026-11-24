"""Build a MaskedConvNextV2 from a named variant.

Random init uses the architecture table. ImageNet init loads the public
``facebook/convnextv2-*-1k-224`` ConvNextV2Model, optionally sums the RGB
stem into one grayscale channel, then copies embeddings and encoder into the
masked encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
from transformers.models.convnextv2.modeling_convnextv2 import (
    ConvNextV2Embeddings,
    ConvNextV2Model,
)

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_masked_encoder import (
    MaskedConvNextV2,
)

INIT_RANDOM = "random"
INIT_IMAGENET = "imagenet"
INIT_KINDS = (INIT_RANDOM, INIT_IMAGENET)

EXPECTED_RGB_CHANNELS = 3
GRAYSCALE_CHANNELS = 1
PATCH_SIZE = 4


@dataclass(frozen=True)
class ConvNextV2VariantSpec:
    depths: tuple[int, ...]
    hidden_sizes: tuple[int, ...]
    imagenet_hub_id: str


CONVNEXTV2_VARIANTS: dict[str, ConvNextV2VariantSpec] = {
    "atto": ConvNextV2VariantSpec(
        depths=(2, 2, 6, 2),
        hidden_sizes=(40, 80, 160, 320),
        imagenet_hub_id="facebook/convnextv2-atto-1k-224",
    ),
    "femto": ConvNextV2VariantSpec(
        depths=(2, 2, 6, 2),
        hidden_sizes=(48, 96, 192, 384),
        imagenet_hub_id="facebook/convnextv2-femto-1k-224",
    ),
    "pico": ConvNextV2VariantSpec(
        depths=(2, 2, 6, 2),
        hidden_sizes=(64, 128, 256, 512),
        imagenet_hub_id="facebook/convnextv2-pico-1k-224",
    ),
    "nano": ConvNextV2VariantSpec(
        depths=(2, 2, 8, 2),
        hidden_sizes=(80, 160, 320, 640),
        imagenet_hub_id="facebook/convnextv2-nano-1k-224",
    ),
    "tiny": ConvNextV2VariantSpec(
        depths=(3, 3, 9, 3),
        hidden_sizes=(96, 192, 384, 768),
        imagenet_hub_id="facebook/convnextv2-tiny-1k-224",
    ),
    "base": ConvNextV2VariantSpec(
        depths=(3, 3, 27, 3),
        hidden_sizes=(128, 256, 512, 1024),
        imagenet_hub_id="facebook/convnextv2-base-1k-224",
    ),
}


def variant_spec(variant: str) -> ConvNextV2VariantSpec:
    if variant not in CONVNEXTV2_VARIANTS:
        raise ValueError(
            f"unknown ConvNeXt V2 variant {variant!r}; "
            f"expected one of {tuple(CONVNEXTV2_VARIANTS)}"
        )
    return CONVNEXTV2_VARIANTS[variant]


def convnextv2_config(variant: str, *, num_channels: int = 1) -> ConvNextV2Config:
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    spec = variant_spec(variant)
    return ConvNextV2Config(
        num_channels=num_channels,
        patch_size=PATCH_SIZE,
        num_stages=len(spec.depths),
        hidden_sizes=spec.hidden_sizes,
        depths=spec.depths,
        drop_path_rate=0.0,
    )


def init_masked_convnextv2(
    variant: str,
    *,
    init: str,
    num_channels: int = 1,
) -> MaskedConvNextV2:
    if init not in INIT_KINDS:
        raise ValueError(f"init must be one of {INIT_KINDS}, got {init!r}")
    if init == INIT_RANDOM:
        return MaskedConvNextV2(convnextv2_config(variant, num_channels=num_channels))
    unmasked = _load_imagenet_unmasked(variant, num_channels=num_channels)
    return MaskedConvNextV2.from_unmasked(unmasked)


def _load_imagenet_unmasked(variant: str, *, num_channels: int) -> ConvNextV2Model:
    spec = variant_spec(variant)
    if num_channels not in (GRAYSCALE_CHANNELS, EXPECTED_RGB_CHANNELS):
        raise ValueError(
            "imagenet init supports num_channels="
            f"{GRAYSCALE_CHANNELS} or {EXPECTED_RGB_CHANNELS}, got {num_channels}"
        )
    unmasked = ConvNextV2Model.from_pretrained(spec.imagenet_hub_id)
    if not isinstance(unmasked, ConvNextV2Model):
        raise TypeError(
            "from_pretrained must return ConvNextV2Model, "
            f"got {type(unmasked)!r}"
        )
    if num_channels == GRAYSCALE_CHANNELS:
        _adapt_stem_to_single_channel(unmasked)
    return unmasked


def _embeddings(model: ConvNextV2Model) -> ConvNextV2Embeddings:
    stem = model.embeddings
    if not isinstance(stem, ConvNextV2Embeddings):
        raise TypeError(
            "ConvNextV2Model.embeddings must be ConvNextV2Embeddings, "
            f"got {type(stem)!r}"
        )
    return stem


def _stem_conv(model: ConvNextV2Model) -> nn.Conv2d:
    conv = _embeddings(model).patch_embeddings
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(
            "ConvNextV2Embeddings.patch_embeddings must be Conv2d, "
            f"got {type(conv)!r}"
        )
    return conv


def _adapt_stem_to_single_channel(model: ConvNextV2Model) -> nn.Conv2d:
    """Replace the 3-channel patch embedding with a 1-channel stem.

    New kernel = sum of the RGB kernels along ``in_channels``. With the same
    bias, ``stem(gray)`` equals ``stem(stack(gray, gray, gray))``.
    """
    stem = _embeddings(model)
    old_stem = _stem_conv(model)
    if old_stem.in_channels != EXPECTED_RGB_CHANNELS:
        raise ValueError(
            "pretrained stem must have in_channels="
            f"{EXPECTED_RGB_CHANNELS}, got {old_stem.in_channels}"
        )
    expected_kernel = (PATCH_SIZE, PATCH_SIZE)
    if old_stem.kernel_size != expected_kernel:
        raise ValueError(
            f"stem kernel_size must be {expected_kernel}, got {old_stem.kernel_size}"
        )
    if old_stem.stride != expected_kernel:
        raise ValueError(
            f"stem stride must be {expected_kernel}, got {old_stem.stride}"
        )
    new_stem = nn.Conv2d(
        in_channels=GRAYSCALE_CHANNELS,
        out_channels=old_stem.out_channels,
        kernel_size=old_stem.kernel_size,
        stride=old_stem.stride,
        padding=old_stem.padding,
        dilation=old_stem.dilation,
        groups=1,
        bias=old_stem.bias is not None,
        padding_mode=old_stem.padding_mode,
    )
    with torch.no_grad():
        new_stem.weight.copy_(old_stem.weight.sum(dim=1, keepdim=True))
        if old_stem.bias is not None:
            new_stem.bias.copy_(old_stem.bias)
    new_stem = new_stem.to(device=old_stem.weight.device, dtype=old_stem.weight.dtype)
    stem.patch_embeddings = new_stem
    stem.num_channels = GRAYSCALE_CHANNELS
    model.config.num_channels = GRAYSCALE_CHANNELS
    return new_stem
