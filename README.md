# ComfyUI Precision Model Save

For users who merge models in ComfyUI and want the saved checkpoint to match the live merged model exactly.

A small diagnostic custom node for saving a **live merged MODEL** without using
ComfyUI's normal `ModelSave` materialisation path.

## What it does

1. Reads the live patch descriptions from `MODEL.get_key_patches()`.
2. Materialises every merged tensor on CPU in **FP32**.
3. Casts only the final tensor to your selected file dtype:
   - `fp32`
   - `bf16`
   - `fp16`
4. Saves a standalone `.safetensors` diffusion model.

The first test should be `fp32`. It will be roughly twice the size of BF16 and
can require a large amount of system RAM while saving.

## Install

Extract the folder into:

`ComfyUI/custom_nodes/ComfyUI-Precision-Model-Save/`

Restart ComfyUI.

The node appears under:

`model/merging -> Precision Model Save (FP32 Materialise)`

## Suggested test

- Feed the exact live merge that produces the expected image into this node.
- Select `fp32`.
- Keep `strip_diffusion_model_prefix = true`.
- Save, restart or fully unload models, then load the saved file with
  `Load Diffusion Model`.
- Render the same prompt, seed, sampler, scheduler, guidance and steps.

If FP32 matches the live merge while BF16/FP16 drift, the culprit is output
precision. If FP32 still differs, the issue is likely not merely the final
save dtype.

## Important

This is an experimental diagnostic node, not a universal saver. It is designed
for ordinary floating-point diffusion models and live ComfyUI model merges.
Quantised tensor formats may need architecture-specific handling.
