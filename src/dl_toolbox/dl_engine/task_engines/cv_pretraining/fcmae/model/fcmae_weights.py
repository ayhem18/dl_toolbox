"""Copy Hugging Face ConvNeXt V2 backbone weights into a masked encoder."""

from __future__ import annotations

from torch import nn


def _unmasked_backbone(module: nn.Module) -> nn.Module:
    nested = getattr(module, "convnextv2", None)
    if nested is not None and hasattr(nested, "embeddings") and hasattr(nested, "encoder"):
        return nested
    if hasattr(module, "embeddings") and hasattr(module, "encoder"):
        return module
    raise TypeError(
        "expected a ConvNextV2Model (embeddings + encoder) or "
        "ConvNextV2ForImageClassification, "
        f"got {type(module)!r}"
    )


def backbone_state_dict(unmasked: nn.Module) -> dict[str, object]:
    """State dict keys that exist on ``MaskedConvNextV2``: embeddings and encoder."""
    backbone = _unmasked_backbone(unmasked)
    state = backbone.state_dict()
    return {
        key: value
        for key, value in state.items()
        if key.startswith("embeddings.") or key.startswith("encoder.")
    }


def inject_unmasked_weights(masked: nn.Module, unmasked: nn.Module) -> None:
    """Load unmasked HF backbone weights into ``MaskedConvNextV2``.

    Skips the HF global-pool ``layernorm``. Module names on embeddings and
    encoder match the Hugging Face tree, so this is a straight copy.
    """
    payload = backbone_state_dict(unmasked)
    missing, unexpected = masked.load_state_dict(payload, strict=False)
    if missing:
        raise ValueError(f"masked encoder is missing weights: {sorted(missing)}")
    if unexpected:
        raise ValueError(
            "unmasked backbone has unused masked-encoder keys: "
            f"{sorted(unexpected)}"
        )
