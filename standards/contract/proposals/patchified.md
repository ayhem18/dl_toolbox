# Patchified tensor / mask

Implemented. Living contracts:

- `.cursor/rules/dl_engine/dl_engine_utils/image_patchify/patchified_tensor.mdc`
- `.cursor/rules/dl_engine/dl_engine_utils/image_patchify/patchified_mask.mdc`

`SEQUENCE` → grid is `unflatten(Y_patches=..., X_patches=...)`. Grid is convert input, not spec.
