"""Dense FCMAE: masked ConvNeXt V2 encoder plus a 1-block decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl_toolbox.dl_engine.task_engines.cv_pretraining.fcmae.model.fcmae_masked_encoder import (
    ConvNextV2Layer,
    MaskedConvNextV2,
    _nchw_mask,
    downsample_mask,
)

DECODER_DEPTH = 1
DECODER_EMBED_DIM = 512


class FcmaeConvNextV2(nn.Module):
    """Encoder last map -> 1x1 proj -> mask token -> ConvNeXt block -> pixel pred."""

    def __init__(
        self,
        encoder: MaskedConvNextV2,
        *,
        decoder_depth: int = DECODER_DEPTH,
        decoder_embed_dim: int = DECODER_EMBED_DIM,
    ) -> None:
        super().__init__()
        if decoder_depth < 1:
            raise ValueError(f"decoder_depth must be >= 1, got {decoder_depth}")
        if decoder_embed_dim < 1:
            raise ValueError(
                f"decoder_embed_dim must be >= 1, got {decoder_embed_dim}"
            )
        hidden = int(encoder.config.hidden_sizes[-1])
        channels = int(encoder.config.num_channels)
        patch = int(encoder.mae_patch_size)
        self.encoder = encoder
        self.mae_patch_size = patch
        self.num_channels = channels
        # Linear, not 1x1 Conv2d: cuDNN 1x1 weight grads are strided as
        # [Cout, Cin, 1, 1] with strides [Cin, 1, Cin, Cin], which DDP warns on.
        self.proj = nn.Linear(hidden, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        self.decoder = nn.ModuleList(
            [
                ConvNextV2Layer(encoder.config, dim=decoder_embed_dim, drop_path=0.0)
                for _ in range(decoder_depth)
            ]
        )
        self.pred = nn.Linear(decoder_embed_dim, patch * patch * channels)
        self._init_decoder()

    def _init_decoder(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for module in (self.proj, self.pred):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_decoder(self, latents: Tensor, image_mask: Tensor) -> Tensor:
        hidden = _linear_nchw(self.proj, latents)
        token_mask = downsample_mask(_nchw_mask(image_mask), self.mae_patch_size)
        if token_mask.shape[-2:] != hidden.shape[-2:]:
            raise ValueError(
                f"token mask spatial {tuple(token_mask.shape[-2:])} does not match "
                f"encoder map {tuple(hidden.shape[-2:])}"
            )
        hidden = hidden * (1.0 - token_mask) + self.mask_token * token_mask
        visible = torch.zeros_like(token_mask)
        n_layers = len(self.decoder)
        for i in range(n_layers):
            hidden = self.decoder[i](hidden, visible)
        return _linear_nchw(self.pred, hidden)

    def forward(self, images: Tensor, mask: Tensor) -> Tensor:
        latents = self.encoder(images, mask)
        return self.forward_decoder(latents, mask)


def _linear_nchw(linear: nn.Linear, features: Tensor) -> Tensor:
    """Apply a Linear to NCHW by moving channels last, then restore NCHW."""
    return linear(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
