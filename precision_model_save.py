import json
import logging
import os
from contextlib import nullcontext

import torch

import comfy.lora
import comfy.model_management
import comfy.utils
import folder_paths
from comfy.cli_args import args


DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

MATERIALISE_MODES = {
    "match_live": (
        "Use ModelPatcher.patch_weight_to_device(return_weight=True), matching "
        "ComfyUI's live LoRA/patch computation, set_func handling, compute dtype, "
        "and deterministic stochastic rounding."
    ),
    "fp32_math": (
        "Rebuild patches manually on CPU in FP32 before the final save cast. "
        "This is the high-precision mathematical merge path from V1."
    ),
}


class PrecisionModelSaveV2:
    """
    Saves a standalone diffusion model from a live ComfyUI MODEL/ModelPatcher.

    Modes
    -----
    match_live:
        Materialises each tensor through ComfyUI's own live patching path using
        patch_weight_to_device(..., return_weight=True). This is intended to
        preserve the behaviour of a live model with ordinary attached LoRAs and
        model-merge patches as closely as possible.

    fp32_math:
        Reconstructs each patched tensor manually on CPU in float32, then casts
        once to the selected output dtype. This preserves V1's high-precision
        merge behaviour, but it may differ slightly from live inference because
        live ComfyUI can use device-dependent LoRA compute dtypes, custom
        set_func logic, and deterministic stochastic rounding.

    Notes
    -----
    Forward hooks, injections, bypass LoRAs, and weight-wrapper patches may not
    be representable as ordinary standalone checkpoint tensors. The node logs a
    warning when it detects those structures.
    """

    VERSION = "2"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": (
                    "STRING",
                    {"default": "diffusion_models/precision_merge_v2"},
                ),
                "materialise_mode": (
                    ["match_live", "fp32_math"],
                    {"default": "match_live"},
                ),
                "save_dtype": (
                    ["fp32", "bf16", "fp16"],
                    {"default": "bf16"},
                ),
                "strip_diffusion_model_prefix": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "model/merging"

    @staticmethod
    def _to_output_tensor(weight, output_dtype):
        if not isinstance(weight, torch.Tensor):
            raise TypeError(
                f"Materialised weight is {type(weight).__name__}, not Tensor"
            )
        return (
            weight.detach()
            .to(device="cpu", dtype=output_dtype, copy=True)
            .contiguous()
        )

    @classmethod
    def _materialise_fp32_math(
        cls,
        key,
        patch_description,
        output_dtype,
    ):
        """
        V1-compatible manual FP32 materialisation.

        patch_description follows ModelPatcher.get_key_patches():
            [(base_weight, convert_func), patch_1, patch_2, ...]
        """
        if not patch_description:
            raise RuntimeError(f"No patch description for {key}")

        base_weight, convert_func = patch_description[0]

        if not isinstance(base_weight, torch.Tensor):
            raise TypeError(
                f"{key}: unsupported base weight type "
                f"{type(base_weight).__name__}"
            )

        weight = base_weight.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )

        if convert_func is not None:
            try:
                weight = convert_func(weight, inplace=True)
            except TypeError:
                weight = convert_func(weight)

        patches = patch_description[1:]
        if patches:
            try:
                weight = comfy.lora.calculate_weight(
                    patches,
                    weight,
                    key,
                    intermediate_dtype=torch.float32,
                )
            except TypeError:
                # Compatibility fallback for older ComfyUI builds.
                weight = comfy.lora.calculate_weight(
                    patches,
                    weight,
                    key,
                )

        return cls._to_output_tensor(weight, output_dtype)

    @classmethod
    def _materialise_match_live(
        cls,
        model,
        key,
        output_dtype,
    ):
        """
        Materialise through ComfyUI's actual live ModelPatcher path.

        return_weight=True prevents the call from replacing the parameter in
        the live model while retaining ComfyUI's normal convert_func, set_func,
        LoRA compute dtype, and deterministic stochastic-rounding behaviour.
        """
        patch_method = getattr(model, "patch_weight_to_device", None)
        if patch_method is None:
            raise RuntimeError(
                "This ComfyUI build does not expose "
                "ModelPatcher.patch_weight_to_device(). "
                "Use fp32_math mode or update ComfyUI."
            )

        load_device = getattr(model, "load_device", None)
        if load_device is None:
            raise RuntimeError(
                f"{key}: MODEL has no load_device; cannot match live patching."
            )

        try:
            weight = patch_method(
                key,
                device_to=load_device,
                return_weight=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Your ComfyUI build has an incompatible "
                "patch_weight_to_device() signature. "
                "Use fp32_math mode or update ComfyUI."
            ) from exc

        return cls._to_output_tensor(weight, output_dtype)

    @staticmethod
    def _warn_about_non_bakeable_features(model):
        warnings = []

        hook_patches = getattr(model, "hook_patches", None)
        if hook_patches:
            warnings.append(
                f"{len(hook_patches)} hook-patch group(s)"
            )

        wrapper_patches = getattr(model, "weight_wrapper_patches", None)
        if wrapper_patches:
            warnings.append(
                f"{len(wrapper_patches)} weight-wrapper patch group(s)"
            )

        injections = getattr(model, "injections", None)
        if injections:
            warnings.append(
                f"{len(injections)} injection group(s)"
            )

        if warnings:
            logging.warning(
                "[PrecisionModelSaveV2] Detected %s. These can modify the "
                "forward pass and may not be fully bakeable into ordinary "
                "checkpoint tensors.",
                ", ".join(warnings),
            )

    def save(
        self,
        model,
        filename_prefix,
        materialise_mode,
        save_dtype,
        strip_diffusion_model_prefix,
        prompt=None,
        extra_pnginfo=None,
    ):
        if materialise_mode not in MATERIALISE_MODES:
            raise ValueError(
                f"Unknown materialise mode: {materialise_mode}"
            )

        output_dtype = DTYPES[save_dtype]

        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            resolved_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
        )
        os.makedirs(full_output_folder, exist_ok=True)

        output_name = f"{filename}_{counter:05}_.safetensors"
        output_path = os.path.join(
            full_output_folder,
            output_name,
        )

        logging.info(
            "[PrecisionModelSaveV2] Mode=%s; saving as %s to %s",
            materialise_mode,
            save_dtype,
            output_path,
        )
        logging.info(
            "[PrecisionModelSaveV2] %s",
            MATERIALISE_MODES[materialise_mode],
        )

        self._warn_about_non_bakeable_features(model)

        # Includes base diffusion tensors and ordinary live merge/LoRA patches.
        patch_map = model.get_key_patches("diffusion_model.")

        if not patch_map:
            raise RuntimeError(
                "No diffusion_model.* weights found. "
                "This node expects a ComfyUI MODEL/ModelPatcher."
            )

        state_dict = {}
        total = len(patch_map)

        # Eject model injections while reading ordinary patch tensors, matching
        # ComfyUI's own saving/state-dict behaviour where available.
        use_ejected = getattr(model, "use_ejected", None)
        context = use_ejected() if callable(use_ejected) else nullcontext()

        with context:
            for index, (
                internal_key,
                patch_description,
            ) in enumerate(
                patch_map.items(),
                start=1,
            ):
                comfy.model_management.throw_exception_if_processing_interrupted()

                save_key = internal_key
                if (
                    strip_diffusion_model_prefix
                    and save_key.startswith("diffusion_model.")
                ):
                    save_key = save_key[
                        len("diffusion_model.") :
                    ]

                if materialise_mode == "match_live":
                    tensor = self._materialise_match_live(
                        model,
                        internal_key,
                        output_dtype,
                    )
                else:
                    tensor = self._materialise_fp32_math(
                        internal_key,
                        patch_description,
                        output_dtype,
                    )

                state_dict[save_key] = tensor

                if (
                    index == 1
                    or index % 100 == 0
                    or index == total
                ):
                    logging.info(
                        "[PrecisionModelSaveV2] %d/%d tensors materialised",
                        index,
                        total,
                    )

        metadata = {
            "format": "pt",
            "comfy_precision_model_save": self.VERSION,
            "materialise_mode": materialise_mode,
            "saved_dtype": save_dtype,
            "stripped_diffusion_model_prefix": str(
                bool(strip_diffusion_model_prefix)
            ).lower(),
        }

        if materialise_mode == "fp32_math":
            metadata["materialise_dtype"] = "fp32"
        else:
            metadata["materialise_dtype"] = "comfy_live_path"

        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata[key] = json.dumps(value)

        comfy.utils.save_torch_file(
            state_dict,
            output_path,
            metadata=metadata,
        )

        del state_dict

        logging.info(
            "[PrecisionModelSaveV2] Saved: %s",
            output_path,
        )
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "PrecisionModelSaveV2": PrecisionModelSaveV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrecisionModelSaveV2": (
        "Precision Model Save V2 (Live Match / FP32 Math)"
    ),
}
