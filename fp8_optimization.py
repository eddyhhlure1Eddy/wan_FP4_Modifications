import torch
import torch.nn as nn
from .utils import log

# Try to import FP4 support (experimental)
try:
    # Check if torchao MX formats are available for FP4
    from torchao.prototype.mx_formats import MXLinear, f32_to_f4_unpacked, f4_unpacked_to_f32
    FP4_AVAILABLE = True
    log.info("FP4 microscaling support available (experimental)")
except ImportError:
    FP4_AVAILABLE = False
    log.info("FP4 microscaling not available (torchao not installed or incompatible)")

#based on ComfyUI's and MinusZoneAI's fp8_linear optimization
def fp8_linear_forward(cls, base_dtype, input):
    weight_dtype = cls.weight.dtype
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            input_shape = input.shape
            
            scale_weight = getattr(cls, 'scale_weight', None)
            if scale_weight is None:
                scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
            else:
                scale_weight = scale_weight.to(input.device)
            
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            input = torch.clamp(input, min=-448, max=448, out=input)
            inn = input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous() #always e4m3fn because e5m2 * e5m2 is not supported

            bias = cls.bias.to(base_dtype) if cls.bias is not None else None

            o = torch._scaled_mm(inn, cls.weight.t(), out_dtype=base_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)

            return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
        else:
            return cls.original_forward(input.to(base_dtype))
    else:
        return cls.original_forward(input)


@torch.compiler.disable()
def apply_lora(weight, lora, step=None):
    for lora_diff, lora_strength in zip(lora[0], lora[1]):
        if isinstance(lora_strength, list):
            lora_strength = lora_strength[step]
            if lora_strength == 0.0:
                continue
        elif lora_strength == 0.0:
            continue
        patch_diff = torch.mm(
            lora_diff[0].flatten(start_dim=1).to(weight.device),
            lora_diff[1].flatten(start_dim=1).to(weight.device)
        ).reshape(weight.shape)
        alpha = lora_diff[2] / lora_diff[1].shape[0] if lora_diff[2] is not None else 1.0
        scale = lora_strength * alpha
        weight = weight.add(patch_diff, alpha=scale)
    return weight

def convert_fp8_linear(module, base_dtype, params_to_keep={}, scale_weight_keys=None):
    log.info("FP8 matmul enabled")
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(submodule, nn.Linear):
                if scale_weight_keys is not None:
                    scale_key = f"{name}.scale_weight"
                    if scale_key in scale_weight_keys:
                        setattr(submodule, "scale_weight", scale_weight_keys[scale_key].float())
                original_forward = submodule.forward
                setattr(submodule, "original_forward", original_forward)
                setattr(submodule, "forward", lambda input, m=submodule: fp8_linear_forward(m, base_dtype, input))

def convert_fp4_linear(module, base_dtype, params_to_keep={}, scale_weight_keys=None):
    """
    Experimental FP4 linear layer optimization.

    Note: This is primarily a placeholder. True FP4 acceleration happens in attention layers
    via sageattn_3_fp4 mode, not in weight quantization.

    PyTorch doesn't have native FP4 dtype, so this function:
    1. Keeps weights in FP8 (if available) or FP16
    2. Supports scaled FP8 weights if scale_weight_keys provided
    3. Logs a warning that FP4 is for attention, not weights
    """
    log.info("FP4 quantization mode enabled")
    log.info("Note: FP4 is primarily used for attention computation, not weight storage.")
    log.info("Weights will use FP8 format, FP4 acceleration happens in attention layers")

    if scale_weight_keys is not None:
        log.info("Using scaled FP8 weights for improved precision")

    # Use FP8 for weights since PyTorch doesn't support FP4 weight storage
    # The actual FP4 acceleration happens in the attention layers
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(submodule, nn.Linear):
                # Add scale weights if available (for scaled models)
                if scale_weight_keys is not None:
                    scale_key = f"{name}.scale_weight"
                    if scale_key in scale_weight_keys:
                        setattr(submodule, "scale_weight", scale_weight_keys[scale_key].float())

                # Keep using FP8 for weights
                original_forward = submodule.forward
                setattr(submodule, "original_forward", original_forward)
                setattr(submodule, "forward", lambda input, m=submodule: fp8_linear_forward(m, base_dtype, input))

