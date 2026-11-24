"""Masked-dense ConvNeXt V2 encoder.

Local copy of Hugging Face ``transformers.models.convnextv2.modeling_convnextv2``,
with mask threading on spatial-mixing ops. Not a subclass of the HF modules.

``mask`` is 1 on removed pixels and 0 on visible pixels. It has the same
spatial size as the image. ``forward`` of ``MaskedConvNextV2`` takes ``x``
and this mask. The constructor is the same as ``ConvNextV2Model``:
``MaskedConvNextV2(config)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_weights import (
    inject_unmasked_weights,
)


def mae_patch_size(config: ConvNextV2Config) -> int:
    patch = config.patch_size
    if not isinstance(patch, int):
        raise TypeError(f"config.patch_size must be an int, got {patch!r}")
    return patch * (2 ** (config.num_stages - 1))


def _nchw_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask
    raise ValueError(
        "mask must have shape (N, H, W) or (N, 1, H, W), "
        f"got {tuple(mask.shape)}"
    )


def _visible(mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - mask


def _require_same_spatial(
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    label: str,
) -> None:
    feat_h, feat_w = features.shape[-2:]
    mask_h, mask_w = mask.shape[-2:]
    if feat_h != mask_h or feat_w != mask_w:
        raise ValueError(
            f"{label} mask spatial {(mask_h, mask_w)} does not match "
            f"features {(feat_h, feat_w)}"
        )


def downsample_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
    if stride < 1:
        raise ValueError(f"stride must be a positive int, got {stride!r}")
    if stride == 1:
        return mask
    _, _, height, width = mask.shape
    if height % stride != 0 or width % stride != 0:
        raise ValueError(
            f"mask spatial {(height, width)} is not divisible by stride {stride}"
        )
    return mask[:, :, ::stride, ::stride]


class ConvNextV2GRN(nn.Module):
    """GRN. Global L2 is taken over visible locations only."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        visible_nhwc = _visible(mask).permute(0, 2, 3, 1)
        gated = hidden_states * visible_nhwc
        global_features = torch.linalg.vector_norm(gated, ord=2, dim=(1, 2), keepdim=True)
        norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + 1e-6)
        return self.weight * (hidden_states * norm_features) + self.bias + hidden_states


class ConvNextV2LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, *, eps=1e-6, data_format="channels_last", **kwargs):
        super().__init__(normalized_shape, eps=eps, **kwargs)
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.data_format = data_format

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_first":
            features = features.permute(0, 2, 3, 1)
            features = super().forward(features)
            features = features.permute(0, 3, 1, 2)
        else:
            features = super().forward(features)
        return features


class ConvNextV2Embeddings(nn.Module):
    def __init__(self, config: ConvNextV2Config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels,
            config.hidden_sizes[0],
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.layernorm = ConvNextV2LayerNorm(
            config.hidden_sizes[0], eps=1e-6, data_format="channels_first"
        )
        self.num_channels = config.num_channels
        self.stride = int(config.patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match "
                "with the one set in the configuration."
            )
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings


class ConvNextV2DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prob = 1 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = torch.rand(
            shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        random_tensor = torch.floor(random_tensor + keep_prob)
        return hidden_states.div(keep_prob) * random_tensor

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class ConvNextV2Layer(nn.Module):
    """ConvNeXt V2 block with mask bracketing on the 7x7 depthwise conv."""

    def __init__(self, config: ConvNextV2Config, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = ConvNextV2LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = ACT2FN[config.hidden_act]
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = (
            ConvNextV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _require_same_spatial(features, mask, label="ConvNextV2Layer")
        residual = features
        visible = _visible(mask)
        features = self.dwconv(features * visible) * visible
        features = features.permute(0, 2, 3, 1)
        features = self.layernorm(features)
        features = self.pwconv1(features)
        features = self.act(features)
        features = self.grn(features, mask)
        features = self.pwconv2(features)
        features = features.permute(0, 3, 1, 2)
        features = residual + self.drop_path(features)
        return features


class ConvNextV2Stage(nn.Module):
    def __init__(
        self,
        config: ConvNextV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        depth: int = 2,
        drop_path_rates: list[float] | None = None,
    ):
        super().__init__()
        self.stride = stride
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.ModuleList(
                [
                    ConvNextV2LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=kernel_size, stride=stride
                    ),
                ]
            )
        else:
            self.downsampling_layer = nn.ModuleList()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.ModuleList(
            [
                ConvNextV2Layer(config, dim=out_channels, drop_path=drop_path_rates[j])
                for j in range(depth)
            ]
        )

    def _mask_for_layers(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        feat_h, feat_w = features.shape[-2:]
        mask_h, mask_w = mask.shape[-2:]
        if (mask_h, mask_w) == (feat_h, feat_w):
            return mask
        if (
            self.stride > 1
            and mask_h == feat_h * self.stride
            and mask_w == feat_w * self.stride
        ):
            return downsample_mask(mask, self.stride)
        raise ValueError(
            "ConvNextV2Stage mask spatial "
            f"{(mask_h, mask_w)} does not match features {(feat_h, feat_w)} "
            f"with stride {self.stride}"
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.downsampling_layer:
            features = layer(features)
        mask = self._mask_for_layers(features, mask)
        _require_same_spatial(features, mask, label="ConvNextV2Stage")
        for layer in self.layers:
            features = layer(features, mask)
        return features


class ConvNextV2Encoder(nn.Module):
    def __init__(self, config: ConvNextV2Config):
        super().__init__()
        self.stages = nn.ModuleList()
        drop_path_rates = [
            x.tolist()
            for x in torch.linspace(
                0, config.drop_path_rate, sum(config.depths), device="cpu"
            ).split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextV2Stage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def masks_for_stages(self, mask: torch.Tensor) -> list[torch.Tensor]:
        """One mask per stage, at that stage's post-downsample resolution."""
        current = _nchw_mask(mask)
        masks: list[torch.Tensor] = []
        for stage in self.stages:
            if stage.stride > 1:
                current = downsample_mask(current, stage.stride)
            masks.append(current)
        return masks

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        stage_masks = self.masks_for_stages(mask)
        for stage, stage_mask in zip(self.stages, stage_masks, strict=True):
            hidden_states = stage(hidden_states, stage_mask)
        return hidden_states


class MaskedConvNextV2(nn.Module):
    """ConvNeXt V2 encoder with masked-dense spatial mixing.

    Same constructor as Hugging Face ``ConvNextV2Model``: ``config`` only.
    ``forward(x, mask)`` returns the spatial last hidden state (no GAP).
    ``mask`` must match ``x`` spatially and be constant on non-overlapping
    MAE patches of size ``mae_patch_size(config)``.
    """

    def __init__(self, config: ConvNextV2Config):
        super().__init__()
        self.config = config
        self.mae_patch_size = mae_patch_size(config)
        self.embeddings = ConvNextV2Embeddings(config)
        self.encoder = ConvNextV2Encoder(config)
        self.register_buffer(
            "_mae_patch_ones",
            torch.ones(1, 1, self.mae_patch_size, self.mae_patch_size),
            persistent=False,
        )
        self._zero_grn()

    def _zero_grn(self) -> None:
        for module in self.modules():
            if isinstance(module, ConvNextV2GRN):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_image_and_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"image must have shape (N, C, H, W), got {tuple(x.shape)}")
        _, _, height, width = x.shape
        patch = self.mae_patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"image spatial {(height, width)} must be divisible by "
                f"{patch}x{patch}"
            )
        mask_nchw = _nchw_mask(mask)
        if mask_nchw.shape[0] != x.shape[0]:
            raise ValueError(
                f"mask batch {mask_nchw.shape[0]} does not match image batch {x.shape[0]}"
            )
        if mask_nchw.shape[-2] != height or mask_nchw.shape[-1] != width:
            raise ValueError(
                f"mask spatial {tuple(mask_nchw.shape[-2:])} must match "
                f"image spatial {(height, width)}"
            )
        mask_nchw = mask_nchw.to(device=x.device, dtype=x.dtype)
        patch_sums = F.conv2d(mask_nchw, self._mae_patch_ones, stride=patch)
        patch_area = float(patch * patch)
        aligned = (patch_sums == 0) | (patch_sums == patch_area)
        if not bool(torch.all(aligned)):
            raise ValueError(
                f"mask is not aligned to non-overlapping {patch}x{patch} patches: "
                "each conv(ones, stride=patch) bin must be 0 or "
                f"{patch * patch}"
            )
        return mask_nchw

    @classmethod
    def from_unmasked(cls, unmasked: nn.Module) -> MaskedConvNextV2:
        config = getattr(unmasked, "config", None)
        if config is None and hasattr(unmasked, "convnextv2"):
            config = unmasked.convnextv2.config
        if config is None:
            raise TypeError("unmasked model has no ConvNextV2Config")
        masked = cls(config)
        inject_unmasked_weights(masked, unmasked)
        return masked

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        image_mask = self._validate_image_and_mask(x, mask)
        hidden_states = self.embeddings(x)
        stem_mask = downsample_mask(image_mask, self.embeddings.stride)
        _require_same_spatial(hidden_states, stem_mask, label="stem")
        hidden_states = hidden_states * _visible(stem_mask)
        return self.encoder(hidden_states, stem_mask)
